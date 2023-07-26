import os
import numpy as np
from math_utils import transform_bundletrack_output_to_world
import rospy
import torch

from sync_data import Synchronizer
from scipy.spatial.transform import Rotation as R

"""Class for generating and managing datasets for ContactNets.
"""

class DatasetManagement:
    def __init__(self, frame_num, start_frame, end_frame, timestamps, toss_id):
        self.frame_num = frame_num
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.timestamps = timestamps
        self.toss_id = toss_id-1

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
            pose = transform_bundletrack_output_to_world(pose, BUNDLESDF_POSE_DIR, ODOM_FILE_PATH)
            pose_ = transform_bundletrack_output_to_world(pose_, BUNDLESDF_POSE_DIR, ODOM_FILE_PATH)
            rotation = pose[:3, :3]
            rotation_ = pose_[:3, :3]
            
            translation = pose[:3, 3]
            translation_ = pose_[:3, 3]
            q = R.from_matrix(rotation).as_quat()
            dt = self.timestamps[frame_id+1].to_sec() -self.timestamps[frame_id].to_sec()
            ang_velocity = self.get_angular_velocity(rotation, rotation_, dt)
            lin_velocity = self.get_linear_velocity(translation, translation_, dt)
            print(ang_velocity.shape, lin_velocity.shape, q.shape, translation.shape)

            curr_cnet_pose = np.hstack(
                (np.hstack((q, translation)), np.hstack((ang_velocity, lin_velocity)))
            )
            poses.append(torch.tensor(curr_cnet_pose))
        traj = torch.stack(poses, dim=0)
        print(f'traj size: {traj.size()}')
        torch.save(torch.tensor(traj), CONTACTNETS_INPUT_DIR + "{}.pt".format(self.toss_id))
        print(f'file {self.toss_id}.pt saved')

    def get_angular_velocity(self, curr_state, next_state, dt):
        R_diff = next_state @ curr_state.T
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

    def get_linear_velocity(self, curr_state, next_state, dt):
        return (next_state - curr_state) / dt

if __name__ == "__main__":
    toss_id = 4
    filename = 'old_toss_4'
    BUNDLESDF_POSE_DIR = "/home/cnets-vision/mengti_ws/BundleSDF/results/"+filename+"/ob_in_cam/"
    CONTACTNETS_INPUT_DIR = ("/home/cnets-vision/mengti_ws/dair_pll/assets/bundlesdf/")
    ODOM_FILE_PATH = ("/home/cnets-vision/mengti_ws/BundleSDF/data/"+filename+"/annotated_poses/")
    GT_POSE_DIR = ("/home/cnets-vision/mengti_ws/robot_filter/dataset/"+filename+"/tagslam_poses/")
    frame_num = len([name for name in os.listdir(BUNDLESDF_POSE_DIR)])
    # start_frame = 360 # toss 1
    # start_frame = 280 # toss 2
    # start_frame = 270 # toss 3
    start_frame = 330 # toss 4
    
    # end_frame = 449 # toss 1
    # end_frame = 310 # toss 2
    # end_frame = 301 # toss 3
    end_frame = 381 # toss 4
    
    # Convert pose data to ContactNets format
    # start_time = rospy.rostime.Time(secs=1655404893, nsecs=899137)  # toss 1
    # start_time = rospy.rostime.Time(secs=1655404908, nsecs=279948) # toss 2
    # start_time=rospy.rostime.Time(secs=1655404920, nsecs=470680) # toss 3
    start_time=rospy.rostime.Time(secs=1655404932, nsecs=647236) # toss 4
    # start_time=rospy.rostime.Time(secs=1655404945, nsecs=387903) # toss 5
    # start_time=rospy.rostime.Time(secs=1655404955, nsecs=463919) # toss 6
    # start_time=rospy.rostime.Time(secs=1655404968, nsecs=399762) # toss 7
    
    # end_time = rospy.rostime.Time(secs=1655404908, nsecs=279948) # toss 1
    # end_time = rospy.rostime.Time(secs=1655404920, nsecs=470680) # toss 2
    # end_time=rospy.rostime.Time(secs=1655404932, nsecs=647236) # toss 3
    end_time=rospy.rostime.Time(secs=1655404945, nsecs=387903) # toss 4
    # end_time=rospy.rostime.Time(secs=1655404955, nsecs=463919) # toss 5
    # end_time=rospy.rostime.Time(secs=1655404968, nsecs=399762) # toss 6
    # end_time=rospy.rostime.Time(secs=1655404978, nsecs=579412) # toss 7
    sync = Synchronizer(GT_POSE_DIR, frame_num, start_time, end_time, save=False)
    bundletrack_time, gt_time = sync.bundletrack_time, sync.gt_time
    print(len(bundletrack_time), len(gt_time))
    dataset = DatasetManagement(frame_num, start_frame, end_frame, bundletrack_time, toss_id)
    dataset.transform()