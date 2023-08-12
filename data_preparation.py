import os
import numpy as np
from file_utils import load_field_from_yaml, load_toss_time_from_yaml
from math_utils import trans_mat_to_pos_quat, transform_bundletrack_output
import rospy
import torch

from sync_data import Synchronizer
from scipy.spatial.transform import Rotation as R, RotationSpline
from scipy import signal
from scipy.interpolate import CubicSpline

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
        self.t = []
        self.toss_id = toss_id-1
        self.positions = [] #(N, 3)
        self.quats = []  #(N, 4)
        self.interpolated_ang_vels = [] #(N,3)
        self.interpolated_lin_vels = [] #(N,3)
        ###### sophter ########
        self.rot_t = None #(N, 3)
        self.p_t = None #(3, N)
        #######################
        self.plot = plot
        self.cam_trans = cam_trans
        self.cam_axis_vec = cam_axis_vec
        self.load_poses()
    
    def load_poses(self):
        rot_t = []
        p_t = []
        for frame_id in range(1, self.frame_num+1):
            pose = np.loadtxt(BUNDLESDF_POSE_DIR + "%04i.txt" % frame_id)
            pose = transform_bundletrack_output(pose, BUNDLESDF_POSE_DIR, ODOM_FILE_PATH, self.cam_trans, self.cam_axis_vec, to_world=True)
            rot_t.append(pose[:3, :3])
            p_t.append(pose[:3, 3])
        self.rot_t = np.array(rot_t)
        self.p_t = np.array(p_t).T
        
    def do_process(self):
        """
        Reference:
            https://github.com/DAIRLab/SoPhTER/blob/master/contactnets/utils/processing/process_dynamics.py
        """
        rot_t = np.array([R.from_matrix(rot).as_rotvec() for rot in self.rot_t])
        rvecs = rotvecfix(rot_t).T
        for i in range(3):
            rvecs[i,:] = signal.medfilt(rvecs[i,:],kernel_size=3)
        
        rot_t = R.from_rotvec(rvecs.T)

        t = np.array([t.to_sec() for t in self.timestamps])
        t_start = t[0]
        t = t - t_start
        quat_t = rot_t.as_quat().T
        pdiff = self.p_t[:,1:] - self.p_t[:,:-1]
        tdiff = np.tile((t[1:] - t[:-1]).reshape([1,-1]), [3,1])
        dp_t = pdiff / tdiff
        dp_t = np.hstack((dp_t[:,[0]],dp_t))
        if True:
            rot_rel = rot_t[:-1].inv() * rot_t[1:]
            rel_vecs = rot_rel.as_rotvec()
            w_t = rel_vecs.T / tdiff
            w_t = np.hstack((w_t[:,[0]],w_t))

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
                w_t[i,:] = signal.medfilt(w_t[i,:],kernel_size=3)
        filter_vel = True
        if filter_vel:
            # filter linear velocity
            w_v = np.clip((fc_v / (fs / 2)), a_min = 0.000001, a_max = 0.999999) # Normalize the frequency
            b, a = signal.butter(1, w_v, 'low')
            for i in range(3):
                #dp_t[i,:] = signal.filtfilt(b, a, dp_t[i,:],padtype='odd',padlen=100)
                dp_t[i,:] = signal.medfilt(dp_t[i,:],kernel_size=3)

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
        
        data = np.concatenate((self.p_t, quat_t, dp_t, w_t), axis=0)
        p_t = self.p_t.T
        quat_t = quat_t.T
        dp_t = dp_t.T
        w_t = w_t.T
        data = data.T
        fig, ax = plt.subplots(4, 3, figsize=(15, 15))
        ax[0, 0].plot(p_t[:, 0])
        ax[0, 0].set_title('X Position')
        ax[0, 1].plot(p_t[:, 1])
        ax[0, 1].set_title('Y Position')
        ax[0, 2].plot(p_t[:, 2])
        ax[0, 2].set_title('Z Position')

        ax[1, 0].plot(quat_t[:, 0])
        ax[1, 0].set_title('Quaternion q0')
        ax[1, 1].plot(quat_t[:, 1])
        ax[1, 1].set_title('Quaternion q1')
        ax[1, 2].plot(quat_t[:, 2])
        ax[1, 2].set_title('Quaternion q2')
        ax[2, 0].plot(quat_t[:, 3])
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
        fig.suptitle('Generated from Sophter')
        plt.savefig('sophter.png')
        plt.show()

    def transform(self):
        """
        State vector is 3 xyz position + 4 quaternions(w,x,y,z) + 3 linear velocity + 3 angular velocity
        """
        w_t = []
        dp_t = []
        for frame_id in range(self.frame_num-1):
            if frame_id < self.start_frame:
                continue
            if frame_id >= self.end_frame:
                break
            pose = np.loadtxt(BUNDLESDF_POSE_DIR + "%04i.txt" % frame_id)
            pose_ = np.loadtxt(BUNDLESDF_POSE_DIR + "%04i.txt" % (frame_id+1))
            pose = transform_bundletrack_output(pose, BUNDLESDF_POSE_DIR, ODOM_FILE_PATH, self.cam_trans, self.cam_axis_vec)
            pose_ = transform_bundletrack_output(pose_, BUNDLESDF_POSE_DIR, ODOM_FILE_PATH, self.cam_trans, self.cam_axis_vec)
            rotation = pose[:3, :3]
            rotation_ = pose_[:3, :3]
            translation = pose[:3, 3]
            translation_ = pose_[:3, 3]
            q = R.from_matrix(rotation).as_quat()
            q_shuffle = np.concatenate((q[3:4], q[0:3]), axis=0)
            dt = self.timestamps[frame_id+1].to_sec() - self.timestamps[frame_id].to_sec()
            ang_velocity = self.get_angular_velocity(rotation, rotation_, dt)
            ang_velocity_body = rotation.T @ ang_velocity
            lin_velocity = self.get_linear_velocity(translation, translation_, dt)
            w_t.append(ang_velocity_body)
            dp_t.append(lin_velocity)
            self.t.append(self.timestamps[frame_id].to_sec())
            ################ For Plotting ################
            self.positions.append(translation)
            self.quats.append(q_shuffle)
            ##############################################
        filter_rot = False
        if filter_rot:
            rot_t = np.array([R.from_matrix(rot).as_rotvec() for rot in self.rot_t])
            rvecs = rotvecfix(rot_t)
            print('rvecs', rvecs.shape)
            for i in range(3):
                rvecs[i,:] = signal.medfilt(rvecs[i,:],kernel_size=3)
            rot_t = R.from_rotvec(rvecs)
            self.quats = rot_t.as_quat()

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
        self.interpolated_ang_vels = np.array(w_t)#w_t
        self.interpolated_lin_vels = np.array(dp_t)#dp_t
        print(self.positions.shape, self.quats.shape, self.interpolated_ang_vels.shape, self.interpolated_lin_vels.shape)
        data = np.concatenate((self.positions, self.quats, self.interpolated_lin_vels, self.interpolated_ang_vels), axis=1)
        print(f'traj size: {data.shape}')
        torch.save(torch.tensor(data), CONTACTNETS_INPUT_DIR + "{}.pt".format(self.toss_id))
        print(f'file {self.toss_id}.pt saved')
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
        plt.savefig('bundlesdf_interp.png')
        plt.show()

#################### Plotting contactnets sample traj #################
def visualize_trajectory(file_path):
    data = torch.load(file_path)
    # Assuming the data tensor has the format [N, 13]
    positions = data[:, 4:7].numpy()
    quats = data[:, 0:4].numpy()
    angular_vels = data[:, 7:10].numpy()
    linear_vels = data[:, 10:13].numpy()

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
    plt.savefig('sample_traj_100.png')
    # plt.show()

#######################################################################
if __name__ == "__main__":
    # visualize_trajectory('/home/cnets-vision/mengti_ws/dair_pll_latest/assets/contactnets_cube/250.pt')
    toss_id = 10
    toss_type = 'cube'
    filename = f'old_toss_{toss_id}'
    yaml_path = './assets/config.yaml'
    BUNDLESDF_POSE_DIR = "/home/cnets-vision/mengti_ws/BundleSDF/results/"+filename+"/ob_in_cam/"
    CONTACTNETS_INPUT_DIR = ("/home/cnets-vision/mengti_ws/dair_pll_latest/assets/bundlesdf/")
    ODOM_FILE_PATH = ("/home/cnets-vision/mengti_ws/BundleSDF/data/"+filename+"/annotated_poses/")
    GT_POSE_DIR = ("/home/cnets-vision/mengti_ws/robot_filter/dataset/"+filename+"/tagslam_poses/")
    frame_num = len([name for name in os.listdir(BUNDLESDF_POSE_DIR)])
    CAMERA_EXTRINSICS_FILE = './assets/realsense_pose_cube_old.yaml'
    cam = 'cam0' # realsense camera name
    with open(CAMERA_EXTRINSICS_FILE, 'r') as stream:
        data_loaded = yaml.safe_load(stream)
    print(data_loaded[cam]['pose']['position'])

    cam_pos_dict = data_loaded[cam]['pose']['position']
    cam_trans = np.array([cam_pos_dict['x'], cam_pos_dict['y'], cam_pos_dict['z']]).reshape(-1, 1)
    cam_rot_dict = data_loaded[cam]['pose']['rotation']
    cam_axis_vec = np.array([cam_rot_dict['x'], cam_rot_dict['y'], cam_rot_dict['z']])
    
    start_time = load_toss_time_from_yaml(yaml_path, toss_type, toss_id, 'start_time')
    end_time = load_toss_time_from_yaml(yaml_path, toss_type, toss_id, 'end_time')
    start_frame = load_field_from_yaml(yaml_path, toss_type, toss_id, 'start_frame')
    end_frame = load_field_from_yaml(yaml_path, toss_type, toss_id, 'end_frame')
    sync = Synchronizer(GT_POSE_DIR, frame_num, start_time, end_time, save=False)
    bundletrack_time, gt_time = sync.bundletrack_time, sync.gt_time
    print(len(bundletrack_time), len(gt_time))
    dataset = DatasetManagement(frame_num, start_frame, end_frame, bundletrack_time, toss_id, cam_trans, cam_axis_vec, plot=True)
    dataset.transform()
    # dataset.do_process()