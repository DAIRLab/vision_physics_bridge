import argparse
import os
import numpy as np
from file_utils import load_dataset_from_yaml, load_field_from_yaml, load_toss_time_from_yaml
from math_utils import trans_mat_to_pos_quat, transform_bundletrack_output, wxyz2xyzw, xyzw2wxyz
from rosbag_processor import extract_time_versus_poses, extract_timestamps
import rospy
import torch

from sync_data import Synchronizer
from scipy.spatial.transform import Rotation as R, RotationSpline
from scipy import signal
from scipy.interpolate import CubicSpline, interp1d
from pyquaternion import Quaternion
import matplotlib.pyplot as plt
import yaml
import pdb
import math

"""Class for generating and managing datasets for ContactNets.
"""

def rotvecfix(rv):
    for i in range(rv.shape[0]-1):
        rvi = rv[i,:]
        rvip1 = rv[i+1,:]
        theta = np.linalg.norm(rvip1)
        if theta > 0.0:
            rnew = rvip1*(1 - 2*math.pi/theta)
            if np.linalg.norm(rvi - rnew) < np.linalg.norm(rvi - rvip1):
                rv[i+1,:] = rnew
    return rv

def smooth_positions(positions, window_size=5):
    """
    Smooths positions using a moving average.
    :param positions: Nx3 array of positions.
    :param window_size: Size of the moving average window.
    :return: Smoothed Nx3 array of positions.
    """
    smoothed_positions = np.zeros_like(positions)
    half_window = window_size // 2

    for i in range(positions.shape[0]):
        start_idx = max(0, i - half_window)
        end_idx = min(positions.shape[0], i + half_window)
        smoothed_positions[i] = np.mean(positions[start_idx:end_idx], axis=0)

    return smoothed_positions

def smooth_quaternions_pyquat(quats, alpha=0.5):
    """
    Smooth quaternions using Slerp with pyquaternion.
    :param quats: Nx4 array of quaternions, xyzw
    :param alpha: Interpolation factor (0.0 <= alpha <= 1.0).
    :return: Smoothed Nx4 array of quaternions.
    """
    quats = xyzw2wxyz(quats)
    smoothed_quats = np.zeros_like(quats)
    smoothed_quats[0] = quats[0]

    for i in range(1, len(quats)):
        q0 = Quaternion(quats[i-1])
        q1 = Quaternion(quats[i])
        smoothed = Quaternion.slerp(q0, q1, alpha)
        smoothed_quats[i] = [smoothed.w, smoothed.x, smoothed.y, smoothed.z]
    smoothed_quats = wxyz2xyzw(smoothed_quats)
    return smoothed_quats #xyzw

class DatasetManagement:
    def __init__(self, frame_num, start_frame, end_frame, timestamps, toss_id, cam_trans, cam_axis_vec, frame_rate, plot=False, use_gt=False):
        self.frame_num = frame_num
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.timestamps = timestamps
        self.toss_id = toss_id-1
        self.positions = [] #(N, 3)
        self.quats = []  #(N, 4)
        self.interpolated_ang_vels = [] #(N,3)
        self.interpolated_lin_vels = [] #(N,3)
        self.rot_t = None
        ###### sophter ########
        self.t = None # N,
        self.q_t = None # 4,N, x,y,z,w
        self.p_t = None #3,N
        #######################
        self.plot = plot
        self.cam_trans = cam_trans
        self.cam_axis_vec = cam_axis_vec
        self.use_gt = use_gt
        self.frame_rate = frame_rate
        self.load_poses()
        
    def load_poses(self):
        p_t = []
        q_t = []
        t = []
        #####
        tagslam_data = np.loadtxt(GT_POSE_DIR+'tagslam.txt')
        #####
        for frame_id in range(self.frame_num):
            if frame_id < self.start_frame:
                continue
            if frame_id >= self.end_frame:
                break
            if frame_id == self.start_frame:
                t_start = self.timestamps[frame_id]
            if not self.use_gt:
                print('>>>>>>>>>>> Using bundlesdf results')
                pose = np.loadtxt(BUNDLESDF_POSE_DIR + "%04i.txt" % frame_id)
                pose = transform_bundletrack_output(pose, BUNDLESDF_POSE_DIR, ODOM_FILE_PATH, self.cam_trans, self.cam_axis_vec, to_world=True)
                q_t.append(R.from_matrix(pose[:3, :3]).as_quat()) #x,y,z,w
                p_t.append(pose[:3, 3])
            ######
            else:
                print(">>>>>>>>>>> Using ground-truth")
                tagslam_pose = tagslam_data[frame_id, 1:]
                q_t.append(tagslam_pose[3:])#xyzw
                p_t.append(tagslam_pose[:3])
            ######
            curr_t = self.timestamps[frame_id]
            t.append(curr_t - t_start)
        # q_t, p_t, t = self.force_landing(q_t, p_t, t) #force the final configuraton
        quats = np.array(q_t)
        positions = np.array(p_t)
        ts = np.array(t)
        print(f'quats: {quats.shape}, positions: {positions.shape}, ts: {ts.shape}')
        self.q_t = quats
        self.p_t = positions
        self.t = ts
        print(f'load_poses: self.q_t {self.q_t.shape}, self.p_t: {self.p_t.shape}, self.t: {self.t.shape}')
    
    # def force_landing(self, q_t, p_t, t):
    #     data = np.loadtxt(GT_POSE_DIR+'tagslam.txt')
    #     s = self.end_frame-10
    #     e = self.end_frame-1
    #     tagslam_final_poses = data[s:e, 2:]
    #     q_t.append(tagslam_final_poses[:, 3:])#xyzw
    #     p_t.append(tagslam_final_poses[:, :3])
    #     prev_tdiff = self.timestamps[self.end_frame-1] - self.timestamps[self.end_frame-2]
    #     t_start = self.timestamps[self.start_frame]
    #     for i in range(s,e):
    #         curr_t = self.timestamps[i] + prev_tdiff
    #         t.append(curr_t - t_start)
    #     print(len())
    #     return q_t, p_t, t

    def upsample(self, quats, positions, ts, frame_rate, original_frame_rate=30):
        """
        @quats: (N, 4), x,y,z,w
        @positions: (N, 3)
        @frame_rate: final frame rate of the trajectorys
        """
        quats = xyzw2wxyz(quats) #w,x,y,z
        quats = quats / np.linalg.norm(quats, axis=1)[:, np.newaxis]
        # upsample the trajectory
        N = quats.shape[0]
        for i in range(1, N):
            if np.dot(quats[0], quats[i]) < 0:
                quats[i] = -quats[i]

        # Upsample factor
        duration = (N-1)/original_frame_rate
        M = int(duration*frame_rate)+1
        new_times = np.linspace(0, 1, M)

        # Interpolate positions using linear interpolation
        interp_func_x = interp1d(np.linspace(0, 1, N), positions[:, 0], kind='linear')
        interp_func_y = interp1d(np.linspace(0, 1, N), positions[:, 1], kind='linear')
        interp_func_z = interp1d(np.linspace(0, 1, N), positions[:, 2], kind='linear')

        new_positions = np.vstack([
            interp_func_x(new_times),
            interp_func_y(new_times),
            interp_func_z(new_times)
        ]).T

        new_quaternions = np.zeros((M, 4))
        for i in range(N-1):  # Adjusted loop condition
            start_idx = i * (M // (N-1))
            end_idx = start_idx + (M // (N-1))
            t_array = np.linspace(0, 1, M//(N-1))
            q0 = Quaternion(quats[i, :])
            q1 = Quaternion(quats[i+1, :]) if i < N-2 else Quaternion(quats[-1, :])
            interpolated_quats = [Quaternion.slerp(q0, q1, amount=t).elements for t in t_array]
            new_quaternions[start_idx:end_idx, :] = np.array(interpolated_quats)
        if end_idx < M:
            new_quaternions[end_idx:] = quats[-1]
        
        interp_func_timestamps = interp1d(np.linspace(0, 1, N), ts, kind='linear')
        new_timestamps = interp_func_timestamps(new_times)
        new_quaternions = wxyz2xyzw(new_quaternions) #x,y,z,w
        return new_quaternions, new_positions, new_timestamps
            
    def do_process(self):
        """
        Generate contactnets trajectory.
        The PLL format is in:
	    [ quaternion  position  angular_velocity  linear_velocity ]
        where:
            - position:  		[x, y, z] in meters
            - quaternion: 		[qw, qx, qy, qz]
            - linear_velocity: 	[vx, vy, vz] in meters/second
            - angular_velocity:	[wx, wy, wz] in rad/second in body frame
        """
        rot_t = R.from_quat(self.q_t)
        filter_rot = True
        if filter_rot:
            rvecs = rotvecfix(rot_t.as_rotvec()).T
            for i in range(3):
                rvecs[i,:] = signal.medfilt(rvecs[i,:],kernel_size=3)
        rot_t = rot_t.from_rotvec(rvecs.T)
        quat_t = rot_t.as_quat()  #x,y,z,w
        
        filter_pos = True
        if filter_pos:
            # self.p_t = smooth_positions(self.p_t, window_size=5)
            print("self.p_t before smoothing:", self.p_t.shape) #N,3
            for i in range(3):
                self.p_t[:, i] = signal.savgol_filter(self.p_t[:, i], window_length=15, polyorder=3)
            print("self.p_t:", self.p_t.shape) #N,3
        ##### Upsample
        quat_t = quat_t / np.linalg.norm(quat_t, axis=1).reshape(-1,1)
        print(f"quat_t: {quat_t.shape}") #N,4
        # if quat_t.shape[0] >= 100: #if data long enough, skip upsampling
        #     self.q_t, self.p_t, self.t = quat_t, self.p_t, self.t
        # else:
        #     self.q_t, self.p_t, self.t = self.upsample(quat_t, self.p_t, self.t) #xyzw
        if self.frame_rate != 30:
            self.q_t, self.p_t, self.t = self.upsample(quat_t, self.p_t, self.t, frame_rate=self.frame_rate, original_frame_rate=30) #xyzw
            print(f'after upsample: {self.q_t.shape}, {self.p_t.shape}, {self.t.shape}')
        self.q_t = self.q_t / np.linalg.norm(self.q_t, axis=1).reshape(-1,1) #N,4
        rot_t = R.from_quat(self.q_t) #N,3,3
        #####
        adjust_pos = True
        # FINAL _Z = 0.05 #0.032  #0.050924062270897685 #0.025 #0.05126618331135579
        PLANK_HEIGHT = FINAL_Z - self.p_t[-1, -1]
        # PLANK_HEIGHT = 1.0
        print(f'PLANK_HEIGHT: {PLANK_HEIGHT}')
        if adjust_pos: # the franka is on a plank, need to add the plank height to z
            for i in range(self.p_t.shape[0]): #N,3
                self.p_t[i, -1] = self.p_t[i, -1] + PLANK_HEIGHT
                # if self.p_t[-1, -1] != 0.0: # force landing on the table
                #     self.p_t[-1, -1] = 0.0
                # if self.p_t[i, -1] < 0.0:
                #     self.p_t[i, -1] = 0.0
        print(f'z position: {self.p_t[-1, -1]}')

        self.p_t = self.p_t.T
        pdiff = self.p_t[:,1:] - self.p_t[:,:-1]
        tdiff = np.tile((self.t[1:] - self.t[:-1]).reshape([1,-1]), [3,1])
        dp_t = pdiff / tdiff
        dp_t = np.hstack((dp_t[:,[0]],dp_t))
        to_body = True
        if to_body:
            rot_diff = [rot_t[i+1] * rot_t[i].inv() for i in range(len(rot_t) - 1)]
            w_t = np.array([rd.as_rotvec() for rd in rot_diff]).T / tdiff
            w_t = np.hstack((w_t[:, [0]], w_t)).T
            w_t_body = np.zeros_like(w_t)
            for j in range(len(rot_t)):
                rotation_matrix = rot_t[j].as_matrix()
                w_t_body[j] = rotation_matrix.T @ w_t[j]
        w_t_body = w_t_body.T
        print(f'w_t_body: {w_t_body.shape}') #3, M
        print(f'dp_t: {dp_t.shape}') #3,M
        # butterworth filter of order 2 to smooth velocity states
        # sampling frequency
        fs = 148.

        # Cut-off frequency of angular velocity filter. < fs/2 (Nyquist)
        fc_w = 60.

        # Cut-off frequency of linear velocity filter. < fs/2 (Nyquist)
        fc_v = 45.

        # Cut-off frequency of linear accel filter. < fs/2 (Nyquist)
        fc_a = 45.
        filter_avel = True
        if filter_avel:
            # filter angular velocity
            w_w = np.clip((fc_w / (fs / 2)), a_min = 0.000001, a_max = 0.999999) # Normalize the frequency
            b, a = signal.butter(1, w_w, 'low')
            print("w_t_body before smoothing:", w_t_body.shape) #3,N
            for i in range(3):
                # w_t_body[i,:] = signal.medfilt(w_t_body[i,:],kernel_size=3)
                w_t_body[i,:] = signal.savgol_filter(w_t_body[i,:], window_length=15, polyorder=4)
            print("w_t_body after smoothing:", w_t_body.shape) #3,N
        filter_vel = True
        if filter_vel:
            # filter linear velocity
            w_v = np.clip((fc_v / (fs / 2)), a_min = 0.000001, a_max = 0.999999) # Normalize the frequency
            b, a = signal.butter(1, w_v, 'low')
            print("dp_t before smoothing:", dp_t.shape) #3,N
            for i in range(3):
                #dp_t[i,:] = signal.filtfilt(b, a, dp_t[i,:],padtype='odd',padlen=100)
                # dp_t[i,:] = signal.medfilt(dp_t[i,:],kernel_size=3)
                dp_t[i,:] = signal.savgol_filter(dp_t[i,:], window_length=10, polyorder=4)
            print("dp_t after smoothing:", dp_t.shape) #3,N
        quat_shuffle = xyzw2wxyz(self.q_t).T #4,N, wxyz
        print(f'quat_shuffle: {quat_shuffle.shape}')
        quat_shuffle = quat_shuffle / np.linalg.norm(quat_shuffle,axis=0)
        print(f'quat normal: {np.linalg.norm(quat_shuffle)}')
        data = np.concatenate((quat_shuffle, self.p_t, w_t_body, dp_t), axis=0)
        p_t = self.p_t.T
        quat_shuffle = quat_shuffle.T
        dp_t = dp_t.T
        w_t_body = w_t_body.T
        data = data.T
        print('data: ', data.shape) #N,13
        print(p_t.shape, quat_shuffle.shape, dp_t.shape, w_t_body.shape)
        torch.save(torch.tensor(data), CONTACTNETS_INPUT_DIR + "{}.pt".format(self.toss_id))
        print(f'file {self.toss_id}.pt saved at {CONTACTNETS_INPUT_DIR + "{}.pt".format(self.toss_id)}')
        fig, ax = plt.subplots(4, 3, figsize=(15, 15))
        ax[0, 0].plot(p_t[:, 0])
        ax[0, 0].set_title('X Position')
        ax[0, 1].plot(p_t[:, 1])
        ax[0, 1].set_title('Y Position')
        ax[0, 2].plot(p_t[:, 2])
        ax[0, 2].set_title('Z Position')

        ax[1, 0].plot(quat_shuffle[:, 0])
        ax[1, 0].set_title('Quaternion w')
        ax[1, 1].plot(quat_shuffle[:, 1])
        ax[1, 1].set_title('Quaternion x')
        ax[1, 2].plot(quat_shuffle[:, 2])
        ax[1, 2].set_title('Quaternion y')
        ax[2, 0].plot(quat_shuffle[:, 3])
        ax[2, 0].set_title('Quaternion z')

        ax[2, 1].plot(w_t_body[:, 0])
        ax[2, 1].set_title('Angular Velocity X')
        ax[2, 2].plot(w_t_body[:, 1])
        ax[2, 2].set_title('Angular Velocity Y')
        ax[3, 0].plot(w_t_body[:, 2])
        ax[3, 0].set_title('Angular Velocity Z')

        ax[3, 1].plot(dp_t[:, 0])
        ax[3, 1].set_title('Linear Velocity X')
        ax[3, 2].plot(dp_t[:, 1])
        ax[3, 2].set_title('Linear Velocity Y')
        plt.tight_layout()
        fig.suptitle('Generated from BundleSDF result')
        plt.savefig(f'bundlesdf_{TOSS_TYPE}_traj_{TOSS_ID}_tagslam.png')
        print(f'Saved fig bundlesdf_{TOSS_TYPE}_traj_{TOSS_ID}_tagslam.png')
        if self.plot:
            plt.show()
        
    def transform(self):
        self.p_t = self.p_t.T
        w_t = []
        dp_t = []
        ################
        filter_rot = True
        rot_t = R.from_quat(self.q_t.T)
        if filter_rot:
            rvecs = rotvecfix(rot_t.as_rotvec()).T
            for i in range(3):
                rvecs[i,:] = signal.medfilt(rvecs[i,:],kernel_size=3)
        rot_t = rot_t.from_rotvec(rvecs.T).as_matrix() #N,3,3
        i = 0
        ################
        for frame_id in range(1, self.frame_num-1):
            print(f'i = {i}, frame_id = {frame_id}')
            if frame_id < self.start_frame:
                continue
            if frame_id >= self.end_frame:
                break
            rotation = rot_t[i]
            rotation_ = rot_t[i+1]
            translation = self.p_t[i]
            translation_ = self.p_t[i+1]
            q = R.from_matrix(rotation).as_quat()
            q_shuffle = np.concatenate((q[3:4], q[0:3]), axis=0) #wxyz
            dt = self.t[i+1] - self.t[i]
            ang_velocity = self.get_angular_velocity(rotation, rotation_, dt)
            ang_velocity_body = rotation.T @ ang_velocity
            lin_velocity = self.get_linear_velocity(translation, translation_, dt)
            w_t.append(ang_velocity_body)
            dp_t.append(lin_velocity)
            ################ For Plotting ################
            self.positions.append(translation)
            self.quats.append(q_shuffle)
            ##############################################
            i += 1 

        # mid filter of order 2 to smooth velocity states
        w_t = np.array(w_t)
        dp_t = np.array(dp_t)
        
        # sampling frequency
        fs = 148.

        # Cut-off frequency of angular velocity filter. < fs/2 (Nyquist)
        fc_w = 60.

        # Cut-off frequency of linear velocity filter. < fs/2 (Nyquist)
        fc_v = 45.

        # Cut-off frequency of linear accel filter. < fs/2 (Nyquist)
        fc_a = 45.
        filter_avel = True
        if filter_avel:
            # filter angular velocity
            w_w = np.clip((fc_w / (fs / 2)), a_min = 0.000001, a_max = 0.999999) # Normalize the frequency
            b, a = signal.butter(1, w_w, 'low')
            for i in range(3):
                w_t[:, i] = signal.medfilt(w_t[:, i],kernel_size=3)
                # w_t[:, i] = signal.filtfilt(b, a, w_t[:, i],padtype='odd')
        
        filter_vel = True
        if filter_vel:
            # filter linear velocity
            w_v = np.clip((fc_v / (fs / 2)), a_min = 0.000001, a_max = 0.999999) # Normalize the frequency
            b, a = signal.butter(1, w_v, 'low')
            for i in range(3):
                # dp_t[:, i] = signal.filtfilt(b, a, dp_t[:, i],padtype='odd')
                dp_t[:, i] = signal.medfilt(dp_t[:, i],kernel_size=3)
        
        self.positions = np.array(self.positions)
        self.quats = np.array(self.quats) #wxyz
        self.interpolated_ang_vels = np.array(w_t)
        self.interpolated_lin_vels = np.array(dp_t)
        print(self.positions.shape, self.quats.shape, self.interpolated_ang_vels.shape, self.interpolated_lin_vels.shape)
        # data = np.concatenate((self.positions, self.quats, self.interpolated_lin_vels, self.interpolated_ang_vels), axis=1)
        data = np.concatenate((self.quats, self.positions, self.interpolated_ang_vels, self.interpolated_lin_vels), axis=1)
        print(f'traj size: {data.shape}')
        torch.save(torch.tensor(data), CONTACTNETS_INPUT_DIR + "{}.pt".format(self.toss_id))
        print(f'file {self.toss_id}.pt saved at {CONTACTNETS_INPUT_DIR + "{}.pt".format(self.toss_id)}')
        ################ For Plotting ################
        if self.plot:
            self.plot_data()
        ##############################################

    def get_angular_velocity(self, curr_rot, next_rot, dt):
        R_diff = next_rot @ curr_rot.T
        trace = np.trace(R_diff)
        theta = np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))
        if np.abs(theta) < 1e-6:
            k = np.array([1.0, 0.0, 0.0])
        else:
            k = np.array([R_diff[2, 1] - R_diff[1, 2],
                        R_diff[0, 2] - R_diff[2, 0],
                        R_diff[1, 0] - R_diff[0, 1]])
            k /= (2.0 * np.sin(theta))

        angular_velocity = (theta / max(dt, 1e-9)) * k
        return angular_velocity

    def get_linear_velocity(self, curr_trans, next_trans, dt):
        return (next_trans - curr_trans) / dt

    def plot_data(self):
        positions = np.array(self.positions)
        quats = np.array(self.quats) #wxyz
        angular_vels = np.array(self.interpolated_ang_vels)
        linear_vels = np.array(self.interpolated_lin_vels)
        fig, ax = plt.subplots(4, 3, figsize=(15, 15))
        ax[0, 0].plot(positions[:, 0])
        ax[0, 0].set_title('X Position')
        ax[0, 1].plot(positions[:, 1])
        ax[0, 1].set_title('Y Position')
        ax[0, 2].plot(positions[:, 2])
        ax[0, 2].set_title('Z Position')

        ax[1, 0].plot(quats[:, 0])
        ax[1, 0].set_title('Quaternion w')
        ax[1, 1].plot(quats[:, 1])
        ax[1, 1].set_title('Quaternion x')
        ax[1, 2].plot(quats[:, 2])
        ax[1, 2].set_title('Quaternion y')
        ax[2, 0].plot(quats[:, 3])
        ax[2, 0].set_title('Quaternion z')

        ax[2, 1].plot(angular_vels[:, 0])
        ax[2, 1].set_title('Angular Velocity X')
        ax[2, 2].plot(angular_vels[:, 1])
        ax[2, 2].set_title('Angular Velocity Y')
        ax[3, 0].plot(angular_vels[:, 2])
        ax[3, 0].set_title('Angular Velocity Z')

        ax[3, 1].plot(linear_vels[:, 0])
        ax[3, 1].set_title('Linear Velocity X')
        ax[3, 2].plot(linear_vels[:, 1])
        ax[3, 2].set_title('Linear Velocity Y')

        plt.tight_layout()
        fig.suptitle('Generated from BundleSDF result')
        plt.savefig(f'bundlesdf_{TOSS_TYPE}_traj_{TOSS_ID}.png')
        print(f'Saved fig bundlesdf_{TOSS_TYPE}_traj_{TOSS_ID}.png')
        plt.show()
 
#################### Plotting contactnets sample traj #################
def visualize_trajectory(trajectory_dir, fig_name):
    # data = torch.load(file_path)   #p_t, quat_shuffle, dp_t, w_t
    # print(data.shape) #N,13
    # positions = data[:, :3].numpy() #N,3
    # quats = data[:, 3:7].numpy() #N,4
    # linear_vels = data[:, 7:10].numpy() #N,3
    # angular_vels = data[:, 10:].numpy() #N,3
    # print(positions.shape, quats.shape, linear_vels.shape, angular_vels.shape)
    traj = torch.load(trajectory_dir) #q_t(wxyz), p_t, w_t, dp_t
    print(f'traj loaded: {traj.size()}') #N,13
    p_t = traj[:,4:7].numpy() #N,3
    q_t = traj[:,:4].numpy() #N,4, w,x,y,z
    q_t_shuffled = np.concatenate((q_t[:, 1:], q_t[:, 0].reshape(-1,1)), axis=1) ##N,4, x,y,z,w
    dp_t = traj[:,10:].numpy() #N,3
    w_t_body = traj[:,7:10].numpy() #N,3, in body frame
    fig, ax = plt.subplots(4, 3, figsize=(15, 15))

    # Plot positions
    ax[0, 0].plot(p_t[:, 0])
    ax[0, 0].set_title('X Position')
    ax[0, 1].plot(p_t[:, 1])
    ax[0, 1].set_title('Y Position')
    ax[0, 2].plot(p_t[:, 2])
    ax[0, 2].set_title('Z Position')

    # Plot Quaternion components
    ax[1, 0].plot(q_t[:, 0])
    ax[1, 0].set_title('Quaternion w')
    ax[1, 1].plot(q_t[:, 1])
    ax[1, 1].set_title('Quaternion x')
    ax[1, 2].plot(q_t[:, 2])
    ax[1, 2].set_title('Quaternion y')
    ax[2, 0].plot(q_t[:, 3])
    ax[2, 0].set_title('Quaternion z')

    # Plot angular velocities
    ax[2, 1].plot(w_t_body[:, 0])
    ax[2, 1].set_title('Angular Velocity X')
    ax[2, 2].plot(w_t_body[:, 1])
    ax[2, 2].set_title('Angular Velocity Y')
    ax[3, 0].plot(w_t_body[:, 2])
    ax[3, 0].set_title('Angular Velocity Z')

    # Plot linear velocities
    ax[3, 1].plot(dp_t[:, 0])
    ax[3, 1].set_title('Linear Velocity X')
    ax[3, 2].plot(dp_t[:, 1])
    ax[3, 2].set_title('Linear Velocity Y')

    plt.tight_layout()
    fig.suptitle('ContactNets cube trajectory')
    plt.savefig(fig_name)
    print(f'Saved to {fig_name}')
    plt.show()

#######################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--toss_id",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--type",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--use_gt",
        type=bool,
        required=False
    )
    parser.add_argument(
        "--zshift",
        type=float,
        default=0.05148739950625105
    )
    args = parser.parse_args()
    TOSS_ID = args.toss_id
    TOSS_TYPE = args.type
    USE_GT = args.use_gt
    FINAL_Z = args.zshift
    DATASET = f'{TOSS_TYPE}_{TOSS_ID}'
    YAML_PATH = './assets/config.yaml'
    ROSBAG = load_dataset_from_yaml(YAML_PATH, TOSS_TYPE, TOSS_ID)
    DEPTH_BAG_FILE = f"./rosbags/raw_{ROSBAG}.bag"
    ODOM_BAG_FILE = f"./rosbags/odom_{ROSBAG}.bag"
    # ODOM_BAG_FILE = f"./rosbags/adjusted_odom_{ROSBAG}.bag"
    DEPTH_ROS_TOPIC = "/camera/aligned_depth_to_color/image_raw"
    ODOM_ROS_TOPIC = f"/tagslam/odom/body_{TOSS_TYPE}"
    CAMERA_EXTRINSICS_FILE = f'./assets/realsense_pose_{TOSS_TYPE}.yaml'
    if TOSS_TYPE == 'milk' or TOSS_TYPE == 'prism':
        CAMERA_EXTRINSICS_FILE = f'./assets/realsense_pose_milk_prism.yaml'
    print(f'Processing toss {TOSS_TYPE}_{TOSS_ID} in raw_{ROSBAG}.bag')
    
    DEPTH_TOPIC = '/camera/aligned_depth_to_color/image_raw'
    BUNDLESDF_POSE_DIR = f"/home/cnets-vision/mengti_ws/BundleSDF/results/{DATASET}/ob_in_cam/"
    # CONTACTNETS_INPUT_DIR = f"/home/cnets-vision/mengti_ws/BundleSDF/dair_pll/assets/bundlesdf_{TOSS_TYPE}/"
    CONTACTNETS_INPUT_DIR = f"/home/cnets-vision/mengti_ws/dair_pll_latest/assets/bundlesdf_{TOSS_TYPE}/"
    ODOM_FILE_PATH = f"/home/cnets-vision/mengti_ws/BundleSDF/data/{DATASET}/annotated_poses/"
    GT_POSE_DIR = f"/home/cnets-vision/mengti_ws/robot_filter/dataset/{DATASET}/tagslam_poses/"
    # PLANK_HEIGHT = -0.05458#0.03428 #0.0145
    data = np.loadtxt(GT_POSE_DIR+'tagslam.txt')
    frame_num = len([name for name in os.listdir(BUNDLESDF_POSE_DIR)]) #data.shape[0]
    print(f'Total frame num: {frame_num}')
    cam = 'cam0' # realsense camera name
    with open(CAMERA_EXTRINSICS_FILE, 'r') as stream:
        data_loaded = yaml.safe_load(stream)
    cam_pos_dict = data_loaded[cam]['pose']['position']
    cam_trans = np.array([cam_pos_dict['x'], cam_pos_dict['y'], cam_pos_dict['z']]).reshape(-1, 1)
    cam_rot_dict = data_loaded[cam]['pose']['rotation']
    cam_axis_vec = np.array([cam_rot_dict['x'], cam_rot_dict['y'], cam_rot_dict['z']])
    
    start_time = load_toss_time_from_yaml(YAML_PATH, TOSS_TYPE, TOSS_ID, 'start_time')
    end_time = load_toss_time_from_yaml(YAML_PATH, TOSS_TYPE, TOSS_ID, 'end_time')
    start_frame = load_field_from_yaml(YAML_PATH, TOSS_TYPE, TOSS_ID, 'start_frame')
    end_frame = load_field_from_yaml(YAML_PATH, TOSS_TYPE, TOSS_ID, 'end_frame')
    # sync = Synchronizer(GT_POSE_DIR, frame_num, start_time, end_time, save=False)
    # bundletrack_time, gt_time = sync.bundletrack_time, sync.gt_time
    # print(len(bundletrack_time), len(gt_time))
    # bundletrack_time = extract_timestamps(rosbag, ros_topic, start_time, end_time)
    
    data = np.loadtxt(GT_POSE_DIR+'tagslam.txt')
    print('data loaded', data.shape)
    gt_time = data[:, 0] #N,
    bundletrack_time = extract_time_versus_poses(
        start_time,
        end_time,
        DEPTH_BAG_FILE,
        ODOM_BAG_FILE,
        DEPTH_ROS_TOPIC,
        ODOM_ROS_TOPIC,
        GT_POSE_DIR,
        save=True,
        # time_offset=125.19
    ).reshape(-1,)
    print(gt_time.shape, bundletrack_time.shape)
    dataset = DatasetManagement(frame_num, start_frame, end_frame, bundletrack_time, TOSS_ID, cam_trans, cam_axis_vec, frame_rate=30, plot=False, use_gt=USE_GT)
    dataset.do_process()
