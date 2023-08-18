from math_utils import world_to_camera
import rospy
import message_filters
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
import numpy as np
from scipy.spatial.transform import Rotation as R
import yaml


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
        self.odom_sub = message_filters.Subscriber("/tagslam/odom/body_cube", Odometry)
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.depth_sub, self.odom_sub], 1000, 10
        )
        self.frame = 0
        self.bundletrack_time = []
        self.gt_time = []
        self.ts.registerCallback(self.callback)
        rospy.spin()

    def callback(self, depth_msg, odom_msg):
        if self.start_time <= depth_msg.header.stamp < self.end_time:
            self.frame += 1
            print(self.frame)
            # print(f"odom_msg.header.time {odom_msg.header.stamp}")
            position = odom_msg.pose.pose.position
            Q = np.zeros((7, 1))
            Q[0] = position.x
            Q[1] = position.y
            Q[2] = position.z
            Q[3] = odom_msg.pose.pose.orientation.x
            Q[4] = odom_msg.pose.pose.orientation.y
            Q[5] = odom_msg.pose.pose.orientation.z
            Q[6] = odom_msg.pose.pose.orientation.w
            if self.save:
                np.savetxt(self.tagslam_dir + "%04i.txt" % self.frame, Q)
            # if self.frame == self.start_frame:
            #     print("Saving!!!!!")
            #     quat = Q[3:].reshape(-1,)
            #     rotation_matrix = R.from_quat(quat).as_matrix()
            #     translation = np.array([[position.x], [position.y], [position.z]])
            #     result = np.vstack(
            #         (np.hstack((rotation_matrix, translation)), np.array([0, 0, 0, 1]))
            #     )
            #     result = world_to_camera(result, cam_trans, cam_axis_vec)
            #     np.savetxt('/home/cnets-vision/mengti_ws/robot_filter/' + "%04i.txt" % 0, result)
            self.bundletrack_time.append(depth_msg.header.stamp)
            self.gt_time.append(odom_msg.header.stamp)
        if self.frame == self.data_length:  # TODO
            rospy.signal_shutdown("Shutting down the node")


if __name__ == "__main__":
    # import os
    # CAMERA_EXTRINSICS_FILE = './assets/realsense_pose_bottle.yaml'

    # cam = 'cam0' # realsense camera name
    # with open(CAMERA_EXTRINSICS_FILE, 'r') as stream:
    #     data_loaded = yaml.safe_load(stream)
    # print(data_loaded[cam]['pose']['position'])

    # cam_pos_dict = data_loaded[cam]['pose']['position']
    # cam_trans = np.array([cam_pos_dict['x'], cam_pos_dict['y'], cam_pos_dict['z']]).reshape(-1, 1)
    # cam_rot_dict = data_loaded[cam]['pose']['rotation']
    # cam_axis_vec = np.array([cam_rot_dict['x'], cam_rot_dict['y'], cam_rot_dict['z']])

    # GT_POSE_DIR = (
    # "/home/cnets-vision/mengti_ws/robot_filter/dataset/bottle_toss/tagslam_poses/"
    # )
    # OUTPUT_POSE_DIR = "/home/cnets-vision/mengti_ws/BundleSDF/results/bottle_toss/ob_in_cam/"
    # frame_num = len([name for name in os.listdir(OUTPUT_POSE_DIR)])
    # # start time
    # start_time = rospy.rostime.Time(secs=1691457456, nsecs=641854)
    # # end time
    # end_time = rospy.rostime.Time(secs=1691457522, nsecs=271324)
    
    # start_frame = 1301
    # sync = Synchronizer(GT_POSE_DIR, frame_num, start_time, end_time, start_frame, save=False)
    pass