import os
import numpy as np
from math_utils import trans_mat_to_pos_quat, transform_bundletrack_output
import rospy
import torch

from sync_data import Synchronizer
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
"""Class for generating and managing datasets for ContactNets.
"""

class DatasetManagement:
    def __init__(self, frame_num, start_frame, end_frame, timestamps, toss_id, plot=False):
        self.frame_num = frame_num
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.timestamps = timestamps
        self.toss_id = toss_id-1
        self.positions = []
        self.quats = []
        self.angular_vels = []
        self.linear_vels = []
        self.plot = plot

    def transform(self):
        """
        State vector is 4 quaternion + 3 xyz position + 3 angular velocity + 3 linear velocity.
        """
        poses = []
        for frame_id in range(self.frame_num-1):
            if frame_id < self.start_frame:
                continue
            if frame_id >= self.end_frame:
                break
            pose = np.loadtxt(BUNDLESDF_POSE_DIR + "%04i.txt" % frame_id)
            pose_ = np.loadtxt(BUNDLESDF_POSE_DIR + "%04i.txt" % (frame_id+1))
            pose = transform_bundletrack_output(pose, BUNDLESDF_POSE_DIR, ODOM_FILE_PATH)
            pose_ = transform_bundletrack_output(pose_, BUNDLESDF_POSE_DIR, ODOM_FILE_PATH)
            rotation = pose[:3, :3]
            rotation_ = pose_[:3, :3]
            
            translation = pose[:3, 3]
            translation_ = pose_[:3, 3]
            q = R.from_matrix(rotation).as_quat()
            dt = self.timestamps[frame_id+1].to_sec() - self.timestamps[frame_id].to_sec()
            ang_velocity = self.get_angular_velocity(rotation, rotation_, dt)
            ang_velocity_body = rotation.T @ ang_velocity
            lin_velocity = self.get_linear_velocity(translation, translation_, dt)
            # print(ang_velocity.shape, lin_velocity.shape, q.shape, translation.shape)
            ################ For Plotting ################
            self.positions.append(translation)
            self.quats.append(q)
            # self.angular_vels.append(ang_velocity)
            self.angular_vels.append(ang_velocity_body)
            self.linear_vels.append(lin_velocity)
            ##############################################
            curr_cnet_pose = np.hstack(
                (np.hstack((q, translation)), np.hstack((ang_velocity, lin_velocity)))
            )
            poses.append(torch.tensor(curr_cnet_pose))
        ################ For Plotting ################
        if self.plot:
            self.plot_data()
        ##############################################
        traj = torch.stack(poses, dim=0)
        print(f'traj size: {traj.size()}')
        # torch.save(torch.tensor(traj), CONTACTNETS_INPUT_DIR + "{}.pt".format(self.toss_id))
        # print(f'file {self.toss_id}.pt saved')

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
        angular_vels = np.array(self.angular_vels)
        linear_vels = np.array(self.linear_vels)
        print(positions.shape, quats.shape, angular_vels.shape, linear_vels.shape)
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
        plt.savefig('bundlesdf.png')
        # plt.show()

#################### Plotting contactnets sample traj #################
def visualize_trajectory(file_path):
    data = torch.load(file_path)
    
    # Assuming the data tensor has the format [N, 13]
    # with 4 elements for quaternions, 3 for positions, 3 for angular velocities, and 3 for linear velocities
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
    # visualize_trajectory('/home/cnets-vision/mengti_ws/dair_pll_latest/assets/contactnets_cube/100.pt')
    toss_id = 1
    filename = 'old_toss_1'
    BUNDLESDF_POSE_DIR = "/home/cnets-vision/mengti_ws/BundleSDF/results/"+filename+"/ob_in_cam/"
    CONTACTNETS_INPUT_DIR = ("/home/cnets-vision/mengti_ws/dair_pll/assets/bundlesdf/")
    ODOM_FILE_PATH = ("/home/cnets-vision/mengti_ws/BundleSDF/data/"+filename+"/annotated_poses/")
    GT_POSE_DIR = ("/home/cnets-vision/mengti_ws/robot_filter/dataset/"+filename+"/tagslam_poses/")
    frame_num = len([name for name in os.listdir(BUNDLESDF_POSE_DIR)])
    start_frame = 360 # toss 1
    # start_frame = 280 # toss 2
    # start_frame = 270 # toss 3
    # start_frame = 330 # toss 4
    
    end_frame = 449 # toss 1
    # end_frame = 310 # toss 2
    # end_frame = 301 # toss 3
    # end_frame = 381 # toss 4
    
    # Convert pose data to ContactNets format
    start_time = rospy.rostime.Time(secs=1655404893, nsecs=899137)  # toss 1
    # start_time = rospy.rostime.Time(secs=1655404908, nsecs=279948) # toss 2
    # start_time=rospy.rostime.Time(secs=1655404920, nsecs=470680) # toss 3
    # start_time=rospy.rostime.Time(secs=1655404932, nsecs=647236) # toss 4
    # start_time=rospy.rostime.Time(secs=1655404945, nsecs=387903) # toss 5
    # start_time=rospy.rostime.Time(secs=1655404955, nsecs=463919) # toss 6
    # start_time=rospy.rostime.Time(secs=1655404968, nsecs=399762) # toss 7
    
    end_time = rospy.rostime.Time(secs=1655404908, nsecs=279948) # toss 1
    # end_time = rospy.rostime.Time(secs=1655404920, nsecs=470680) # toss 2
    # end_time=rospy.rostime.Time(secs=1655404932, nsecs=647236) # toss 3
    # end_time=rospy.rostime.Time(secs=1655404945, nsecs=387903) # toss 4
    # end_time=rospy.rostime.Time(secs=1655404955, nsecs=463919) # toss 5
    # end_time=rospy.rostime.Time(secs=1655404968, nsecs=399762) # toss 6
    # end_time=rospy.rostime.Time(secs=1655404978, nsecs=579412) # toss 7
    sync = Synchronizer(GT_POSE_DIR, frame_num, start_time, end_time, save=False)
    bundletrack_time, gt_time = sync.bundletrack_time, sync.gt_time
    print(len(bundletrack_time), len(gt_time))
    dataset = DatasetManagement(frame_num, start_frame, end_frame, bundletrack_time, toss_id, plot=True)
    dataset.transform()