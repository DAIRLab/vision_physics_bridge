from file_utils import load_toss_time_from_yaml
from math_utils import world_to_camera
import rospy
import message_filters
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
import numpy as np
from scipy.spatial.transform import Rotation as R
import yaml
import os

class Synchronizer:
    def __init__(self, tagslam_dir, data_length, start_time, end_time, save=False) -> None:
        rospy.init_node("listener", anonymous=True)
        self.tagslam_dir = tagslam_dir
        self.data_length = data_length
        self.start_time = start_time
        self.end_time = end_time
        self.save = save
        self.depth_sub = message_filters.Subscriber(
            "/camera/aligned_depth_to_color/image_raw", Image
        )
        self.odom_sub = message_filters.Subscriber("/tagslam/odom/body_bottle", Odometry)
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.depth_sub, self.odom_sub], 1000, 10
        )
        self.frame = 0
        self.bundletrack_time = []
        self.gt_time = []
        self.tagslam_poses = []
        self.ts.registerCallback(self.callback)
        rospy.spin()

    def callback(self, depth_msg, odom_msg):
        if self.start_time <= depth_msg.header.stamp < self.end_time:
            self.frame += 1
            print(self.frame)
            position = odom_msg.pose.pose.position
            Q = np.zeros((7,))
            Q[0] = position.x
            Q[1] = position.y
            Q[2] = position.z
            Q[3] = odom_msg.pose.pose.orientation.x
            Q[4] = odom_msg.pose.pose.orientation.y
            Q[5] = odom_msg.pose.pose.orientation.z
            Q[6] = odom_msg.pose.pose.orientation.w
            self.tagslam_poses.append(Q)
            btime = depth_msg.header.stamp.secs + depth_msg.header.stamp.nsecs * 1e-9
            gtime = odom_msg.header.stamp.secs + odom_msg.header.stamp.nsecs * 1e-9
            self.bundletrack_time.append(btime)
            self.gt_time.append(gtime)
        if self.frame == self.data_length:
            if self.save:
                self.bundletrack_time = np.expand_dims(self.bundletrack_time,axis=0)
                self.gt_time = np.expand_dims(self.gt_time,axis=0)
                self.tagslam_poses = np.array(self.tagslam_poses).T
                data = np.concatenate((self.bundletrack_time, self.gt_time, self.tagslam_poses), axis=0)
                data = data.T #N, 9
                np.savetxt(self.tagslam_dir + 'tagslam.txt', data)
                print(self.tagslam_dir + 'tagslam.txt saved!')
            rospy.signal_shutdown("Shutting down the node")


if __name__ == "__main__":
    # DATASET = "cube_hand_toss_60_3"
    # GT_POSE_DIR = f"/home/cnets-vision/mengti_ws/robot_filter/dataset/{DATASET}/tagslam_poses/"
    # BUNDLESDF_RGB_DIR = f"/home/cnets-vision/mengti_ws/BundleSDF/data/{DATASET}/rgb/"
    # frame_num = len(os.listdir(BUNDLESDF_RGB_DIR))
    # start_frame = 1
    # end_frame = 605
    # sync = Synchronizer(GT_POSE_DIR, frame_num, start_frame, end_frame, save=True)
    
    # DATASET = "new_toss_1"
    # GT_POSE_DIR = f"/home/cnets-vision/mengti_ws/robot_filter/dataset/{DATASET}/tagslam_poses/"
    # BUNDLESDF_RGB_DIR = f"/home/cnets-vision/mengti_ws/BundleSDF/data/{DATASET}/rgb/"
    # frame_num = len(os.listdir(BUNDLESDF_RGB_DIR))
    # yaml_path = "./assets/config.yaml"
    # toss_type = 'cube_new'
    # toss_id=1
    # start_time = load_toss_time_from_yaml(yaml_path, toss_type, toss_id, 'start_time')
    # end_time = load_toss_time_from_yaml(yaml_path, toss_type, toss_id, 'end_time')
    # sync = Synchronizer(GT_POSE_DIR, frame_num, start_time, end_time, save=True)

    DATASET = "napkin_1"
    GT_POSE_DIR = f"/home/cnets-vision/mengti_ws/robot_filter/dataset/{DATASET}/tagslam_poses/"
    BUNDLESDF_RGB_DIR = f"/home/cnets-vision/mengti_ws/BundleSDF/data/{DATASET}/rgb/"
    frame_num = len(os.listdir(BUNDLESDF_RGB_DIR))
    yaml_path = "./assets/config.yaml"
    toss_type = 'napkin'
    toss_id=1
    start_time = load_toss_time_from_yaml(yaml_path, toss_type, toss_id, 'start_time')
    end_time = load_toss_time_from_yaml(yaml_path, toss_type, toss_id, 'end_time')
    sync = Synchronizer(GT_POSE_DIR, frame_num, start_time, end_time, save=True)
    
