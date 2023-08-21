import argparse
import os
import numpy as np
from file_utils import load_field_from_yaml, load_toss_time_from_yaml
from math_utils import trans_mat_to_pos_quat, transform_bundletrack_output
from rosbag_processor import extract_timestamps
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

class DatasetManagement:
    def __init__(self, frame_num, start_frame, end_frame, timestamps, toss_id, cam_trans, cam_axis_vec, plot=False):
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
        self.q_t = None # 4,N
        self.p_t = None #3,N
        #######################
        self.plot = plot
        self.cam_trans = cam_trans
        self.cam_axis_vec = cam_axis_vec
        self.load_poses()
        
    def load_poses(self):
        p_t = []
        q_t = []
        t = []
        for frame_id in range(1, self.frame_num+1):
            if frame_id < self.start_frame:
                continue
            if frame_id >= self.end_frame:
                break
            if frame_id == self.start_frame:
                t_start = self.timestamps[frame_id]
            pose = np.loadtxt(BUNDLESDF_POSE_DIR + "%04i.txt" % frame_id)
            pose = transform_bundletrack_output(pose, BUNDLESDF_POSE_DIR, ODOM_FILE_PATH, self.cam_trans, self.cam_axis_vec, to_world=True)
            q_t.append(R.from_matrix(pose[:3, :3]).as_quat()) #x,y,z,w
            p_t.append(pose[:3, 3])
            curr_t = self.timestamps[frame_id]
            t.append(curr_t - t_start)
        quats = np.array(q_t)
        positions = np.array(p_t)
        ts = np.array(t)
        print(f'quats: {quats.shape}, positions: {positions.shape}, ts: {ts.shape}')
        self.q_t = quats.T
        self.p_t = positions.T
        self.t = ts.T
        print(f'load_poses: self.q_t {self.q_t.shape}, self.p_t: {self.p_t.shape}, self.t: {self.t.shape}')
    
    def upsample(self, quats, positions, ts):
        quats = quats / np.linalg.norm(quats, axis=1)[:, np.newaxis]
        # upsample the trajectory
        N = quats.shape[0]
        for i in range(1, N):
            if np.dot(quats[0], quats[i]) < 0:
                quats[i] = -quats[i]

        # Upsample factor
        M = 100
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
        return new_quaternions.T, new_positions.T, new_timestamps.T
            
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
        rot_t = R.from_quat(self.q_t.T)
        if True:
            rvecs = rotvecfix(rot_t.as_rotvec()).T
            for i in range(3):
                rvecs[i,:] = signal.medfilt(rvecs[i,:],kernel_size=3)
        rot_t = rot_t.from_rotvec(rvecs.T)
        quat_t = rot_t.as_quat().T  #x,y,z,w
        print('before upsampling >>>>>>>>>')
        print(f'quat_t: {quat_t.shape}') #4,N
        print(f'self.p_t: {self.p_t.shape}') # 3,N
        print(f'self.t: {self.t.shape}') #N,
        ##### Upsample
        # quat_t = quat_t.T
        # self.q_t, self.p_t, self.t = self.upsample(quat_t, self.p_t.T, self.t)
        # rot_t = R.from_quat(self.q_t.T)
        # quat_t = rot_t.as_quat().T
        # print('After upsampling >>>>>>>>>')
        # print(f'quat_t: {quat_t.shape}') #4,M
        # print(f'self.p_t: {self.p_t.shape}') # 3,M
        # print(f'self.t: {self.t.shape}') #M,
        #####
        pdiff = self.p_t[:,1:] - self.p_t[:,:-1]
        tdiff = np.tile((self.t[1:] - self.t[:-1]).reshape([1,-1]), [3,1])
        dp_t = pdiff / tdiff
        dp_t = np.hstack((dp_t[:,[0]],dp_t))
        if True:
            rot_rel = rot_t[:-1].inv() * rot_t[1:]
            rel_vecs = rot_rel.as_rotvec()
            w_t = rel_vecs.T / tdiff
            w_t = np.hstack((w_t[:,[0]],w_t))
        print(f'w_t: {w_t.shape}') #3, N
        print(f'dp_t: {dp_t.shape}') #3,N
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
            for i in range(3):
                # w_t[i,:] = signal.medfilt(w_t[i,:],kernel_size=3)
                w_t[i,:] = signal.savgol_filter(w_t[i,:], window_length=15, polyorder=4)
        filter_vel = True
        if filter_vel:
            # filter linear velocity
            w_v = np.clip((fc_v / (fs / 2)), a_min = 0.000001, a_max = 0.999999) # Normalize the frequency
            b, a = signal.butter(1, w_v, 'low')
            for i in range(3):
                #dp_t[i,:] = signal.filtfilt(b, a, dp_t[i,:],padtype='odd',padlen=100)
                # dp_t[i,:] = signal.medfilt(dp_t[i,:],kernel_size=3)
                dp_t[i,:] = signal.savgol_filter(dp_t[i,:], window_length=10, polyorder=4)

        if True:
            # delta v = v' - v
            dpdiff = dp_t[:,1:] - dp_t[:,:-1]
            ddp_t = dpdiff / tdiff
            # repeat first column because want to compare \delta v with corner height of q'
            ddp_t = np.hstack((ddp_t[:,[0]],ddp_t))

        filter_acc = False
        if filter_acc:
            # filter linear velocity
            w_a = np.clip((fc_a / (fs / 2)), a_min = 0.000001, a_max = 0.999999) # Normalize the frequency
            b, a = signal.butter(1, w_a, 'low')
            for i in range(3):
                ddp_t[i,:] = signal.medfilt(ddp_t[i,:],kernel_size=3)
                #ddp_t[i,:] = signal.filtfilt(b, a, ddp_t[i,:],padtype='odd',padlen=100)
        
        quat_shuffle = np.concatenate((quat_t[3:4, :], quat_t[0:3, :]), axis=0) #w,x,y,z
        # data = np.concatenate((self.p_t, quat_shuffle, dp_t, w_t), axis=0)
        data = np.concatenate((quat_shuffle, self.p_t, w_t, dp_t), axis=0)
        p_t = self.p_t.T
        quat_shuffle = quat_shuffle.T
        dp_t = dp_t.T
        w_t = w_t.T
        data = data.T
        print('data: ', data.shape) #N,13
        print(p_t.shape, quat_shuffle.shape, dp_t.shape, w_t.shape)
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
        ax[1, 0].set_title('Quaternion q0')
        ax[1, 1].plot(quat_shuffle[:, 1])
        ax[1, 1].set_title('Quaternion q1')
        ax[1, 2].plot(quat_shuffle[:, 2])
        ax[1, 2].set_title('Quaternion q2')
        ax[2, 0].plot(quat_shuffle[:, 3])
        ax[2, 0].set_title('Quaternion q3')

        ax[2, 1].plot(w_t[:, 0])
        ax[2, 1].set_title('Angular Velocity X')
        ax[2, 2].plot(w_t[:, 1])
        ax[2, 2].set_title('Angular Velocity Y')
        ax[3, 0].plot(w_t[:, 2])
        ax[3, 0].set_title('Angular Velocity Z')

        ax[3, 1].plot(dp_t[:, 0])
        ax[3, 1].set_title('Linear Velocity X')
        ax[3, 2].plot(dp_t[:, 1])
        ax[3, 2].set_title('Linear Velocity Y')
        plt.tight_layout()
        fig.suptitle('Generated from BundleSDF result')
        plt.savefig(f'bundlesdf_{toss_type}_traj_{toss_id}.png')
        print(f'Saved fig bundlesdf_{toss_type}_traj_{toss_id}.png')
        plt.show()
        
    def transform(self):
        """
        State vector is 3 xyz position + 4 quaternions(w,x,y,z) + 3 linear velocity + 3 angular velocity
        """
        w_t = []
        dp_t = []
        ################
        filter_rot = True
        if filter_rot:
            rot_t = np.array([R.from_matrix(rot).as_rotvec() for rot in self.rot_t])
            print(f'rot_t: {rot_t.shape}')
            rvecs = rotvecfix(rot_t)
            print(f'rvecs: {rvecs.shape}')
            for i in range(3):
                rvecs[:, i] = signal.medfilt(rvecs[:, i],kernel_size=3)
            
            rot_t = R.from_rotvec(rvecs).as_matrix()
        ################
        for frame_id in range(1, self.frame_num-1):
            if frame_id < self.start_frame:
                continue
            if frame_id >= self.end_frame:
                break
            rotation = rot_t[frame_id]
            rotation_ = rot_t[frame_id+1]
            translation = self.p_t[frame_id]
            translation_ = self.p_t[frame_id+1]
            q = R.from_matrix(rotation).as_quat()
            q_shuffle = np.concatenate((q[3:4], q[0:3]), axis=0)
            dt = self.timestamps[frame_id+1].to_sec() - self.timestamps[frame_id].to_sec()
            ang_velocity = self.get_angular_velocity(rotation, rotation_, dt)
            ang_velocity_body = rotation.T @ ang_velocity
            lin_velocity = self.get_linear_velocity(translation, translation_, dt)
            w_t.append(ang_velocity_body)
            dp_t.append(lin_velocity)
            ################ For Plotting ################
            self.positions.append(translation)
            self.quats.append(q_shuffle)
            ##############################################

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
        # print("Diff w_t >>>>>>>>>>>>>")
        # diff_acceleration = np.diff(w_t_smoothed, axis=0)
        # threshold = 0.1
        # abrupt_changes = np.abs(diff_acceleration) > threshold
        # print(abrupt_changes)
        
        filter_vel = True
        if filter_vel:
            # filter linear velocity
            w_v = np.clip((fc_v / (fs / 2)), a_min = 0.000001, a_max = 0.999999) # Normalize the frequency
            b, a = signal.butter(1, w_v, 'low')
            for i in range(3):
                # dp_t[:, i] = signal.filtfilt(b, a, dp_t[:, i],padtype='odd')
                dp_t[:, i] = signal.medfilt(dp_t[:, i],kernel_size=3)
        
        # print("Diff dp_t >>>>>>>>>>>>>")
        # diff_acceleration = np.diff(dp_t_smoothed, axis=0)
        # threshold = 0.1
        # abrupt_changes = np.abs(diff_acceleration) > threshold
        # print(abrupt_changes)
        self.positions = np.array(self.positions)
        self.quats = np.array(self.quats)
        self.interpolated_ang_vels = np.array(w_t)
        self.interpolated_lin_vels = np.array(dp_t)
        print(self.positions.shape, self.quats.shape, self.interpolated_ang_vels.shape, self.interpolated_lin_vels.shape)
        data = np.concatenate((self.positions, self.quats, self.interpolated_lin_vels, self.interpolated_ang_vels), axis=1)
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
        quats = np.array(self.quats)
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
        ax[1, 0].set_title('Quaternion q0')
        ax[1, 1].plot(quats[:, 1])
        ax[1, 1].set_title('Quaternion q1')
        ax[1, 2].plot(quats[:, 2])
        ax[1, 2].set_title('Quaternion q2')
        ax[2, 0].plot(quats[:, 3])
        ax[2, 0].set_title('Quaternion q3')

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
        plt.savefig(f'bundlesdf_{toss_type}_traj_{toss_id}.png')
        print(f'Saved fig bundlesdf_{toss_type}_traj_{toss_id}.png')
        plt.show()

#################### Plotting contactnets sample traj #################
def visualize_trajectory(file_path, fig_name):
    data = torch.load(file_path)   #p_t, quat_shuffle, dp_t, w_t
    print(data.shape) #N,13
    positions = data[:, :3].numpy() #N,3
    quats = data[:, 3:7].numpy() #N,4
    linear_vels = data[:, 7:10].numpy() #N,3
    angular_vels = data[:, 10:].numpy() #N,3
    print(positions.shape, quats.shape, linear_vels.shape, angular_vels.shape)
    fig, ax = plt.subplots(4, 3, figsize=(15, 15))

    # Plot positions
    ax[0, 0].plot(positions[:, 0])
    ax[0, 0].set_title('X Position')
    ax[0, 1].plot(positions[:, 1])
    ax[0, 1].set_title('Y Position')
    ax[0, 2].plot(positions[:, 2])
    ax[0, 2].set_title('Z Position')

    # Plot Quaternion components
    ax[1, 0].plot(quats[:, 0])
    ax[1, 0].set_title('Quaternion q0')
    ax[1, 1].plot(quats[:, 1])
    ax[1, 1].set_title('Quaternion q1')
    ax[1, 2].plot(quats[:, 2])
    ax[1, 2].set_title('Quaternion q2')
    ax[2, 0].plot(quats[:, 3])
    ax[2, 0].set_title('Quaternion q3')

    # Plot angular velocities
    ax[2, 1].plot(angular_vels[:, 0])
    ax[2, 1].set_title('Angular Velocity X')
    ax[2, 2].plot(angular_vels[:, 1])
    ax[2, 2].set_title('Angular Velocity Y')
    ax[3, 0].plot(angular_vels[:, 2])
    ax[3, 0].set_title('Angular Velocity Z')

    # Plot linear velocities
    ax[3, 1].plot(linear_vels[:, 0])
    ax[3, 1].set_title('Linear Velocity X')
    ax[3, 2].plot(linear_vels[:, 1])
    ax[3, 2].set_title('Linear Velocity Y')

    plt.tight_layout()
    fig.suptitle('ContactNets cube trajectory')
    plt.savefig(fig_name)
    print(f'Saved to {fig_name}')
    plt.show()

#######################################################################
if __name__ == "__main__":
    # toss_id = 1
    # my_traj = f'/home/cnets-vision/mengti_ws/dair_pll_latest/assets/bundlesdf_cube/{toss_id-1}.pt'
    # visualize_trajectory(my_traj, f'my_traj_{toss_id}.png')
    # sample_traj = '/home/cnets-vision/mengti_ws/dair_pll_latest/assets/contactnets_cube/5.pt' #N,13
    # traj = torch.load(sample_traj)
    # print(traj.size())
    # visualize_trajectory(sample_traj, 'sample_traj_5.png')
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--toss_id",
        type=int,
        required=True,
    )
    args = parser.parse_args()
    toss_id = args.toss_id
    print(f'Processing toss {toss_id}')
    toss_type = 'cube'
    filename = f'old_toss_{toss_id}'
    rosbag = './rosbags/raw_10.bag'
    # toss_type = 'bottle'
    # filename = f'bottle_toss_{toss_id}'
    # rosbag = './rosbags/raw_43.bag'
    yaml_path = './assets/config.yaml'
    ros_topic = '/camera/aligned_depth_to_color/image_raw'
    CAMERA_EXTRINSICS_FILE = './assets/realsense_pose_cube_old.yaml'
    # CAMERA_EXTRINSICS_FILE = './assets/realsense_pose_bottle.yaml'
    BUNDLESDF_POSE_DIR = "/home/cnets-vision/mengti_ws/BundleSDF/results/"+filename+"/ob_in_cam/"
    CONTACTNETS_INPUT_DIR = f"/home/cnets-vision/mengti_ws/dair_pll_latest/assets/bundlesdf_upsample/"
    ODOM_FILE_PATH = "/home/cnets-vision/mengti_ws/BundleSDF/data/"+filename+"/annotated_poses/"
    GT_POSE_DIR = "/home/cnets-vision/mengti_ws/robot_filter/dataset/"+filename+"/tagslam_poses/"
    frame_num = len([name for name in os.listdir(BUNDLESDF_POSE_DIR)])
    print(f'Total frame num: {frame_num}')
    cam = 'cam0' # realsense camera name
    with open(CAMERA_EXTRINSICS_FILE, 'r') as stream:
        data_loaded = yaml.safe_load(stream)
    cam_pos_dict = data_loaded[cam]['pose']['position']
    cam_trans = np.array([cam_pos_dict['x'], cam_pos_dict['y'], cam_pos_dict['z']]).reshape(-1, 1)
    cam_rot_dict = data_loaded[cam]['pose']['rotation']
    cam_axis_vec = np.array([cam_rot_dict['x'], cam_rot_dict['y'], cam_rot_dict['z']])
    
    start_time = load_toss_time_from_yaml(yaml_path, toss_type, toss_id, 'start_time')
    end_time = load_toss_time_from_yaml(yaml_path, toss_type, toss_id, 'end_time')
    start_frame = load_field_from_yaml(yaml_path, toss_type, toss_id, 'start_frame')
    end_frame = load_field_from_yaml(yaml_path, toss_type, toss_id, 'end_frame')
    # sync = Synchronizer(GT_POSE_DIR, frame_num, start_time, end_time, save=False)
    # bundletrack_time, gt_time = sync.bundletrack_time, sync.gt_time
    # print(len(bundletrack_time), len(gt_time))
    
    # bundletrack_time = extract_timestamps(rosbag, ros_topic, start_time, end_time)
    
    data = np.loadtxt(GT_POSE_DIR+'tagslam.txt')
    print('data loaded', data.shape)
    bundletrack_time, gt_time = data[:, 0], data[:, 1] #N,
    dataset = DatasetManagement(frame_num, start_frame, end_frame, bundletrack_time, toss_id, cam_trans, cam_axis_vec, plot=True)
    # dataset.transform()
    dataset.do_process()
