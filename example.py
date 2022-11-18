import argparse
import numpy as np
from tqdm import tqdm
from depth_filter import DepthFilter
from file_utils import (
    create_annotated_poses,
    denoise,
    generate_depth_img_without_robot,
    generate_rgb_image_without_robot,
    import_data,
    write_real_depth_as_txt,
)
from math_utils import axis_angle_to_rotation_matrix

from rosbag_processor import (
    bag_to_depth_images,
    extract_cube_pose,
    extract_gt_poses_from_tagslam,
    extract_poses_with_timestamps,
)
import rospy
from urdf_filter import dilate, run_urdf_filter
import os, os.path

"""Main file to process the box data.
"""

ROSBAG_NAME = "raw_19.bag"
ODOM_ROSBAG_NAME = "odom_19.bag"
DEPTH_ROS_TOPIC = "/camera/aligned_depth_to_color/image_raw"
JOINT_STATE_ROS_TOPIC = "/joint_states"
RGB_ROS_TOPIC = "/camera/color/image_raw"
ODOM_ROS_TOPIC = "/tagslam/odom/body_box"

POSITION_FILE_PATH = "./texts/joint_position.txt"
REAL_DEPTH_FILE = "./texts/real_depth_frame%04i.txt"
MASK_IAMGE_FILE = "./mask_data/%04i.png"
FILTERED_DEPTH_FILE = "./filtered_data/depth_without_robot_frame%04i.png"
FILTERED_RGB_FILE = "./filtered_data/rgb_without_robot_frame%04i.png"
ANNOTATED_POSES_DIR = "/home/cnets-vision/mengti_ws/BundleTrack/Data/YCBINEOAT/contact_nets/annotated_poses/"
TAGSLAM_POSES_DIR = "./tagslam_poses_tmp/"

CAMERA_CONFIG = {
    "old": {
        "translation": np.array([[1.14164360], [0.15815239], [0.66422200]]),
        "axis_vec": np.array([-1.57165949, -1.63112887, 1.07928078]),
    },
    "new": {
        "translation": np.array([[1.11076422], [-0.07966290], [0.67947702]]),
        "axis_vec": np.array([-1.61997882, -1.56988553, 0.86362178]),
    },
}
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth_dir", type=str, default="./depth_data")
    parser.add_argument("--rgb_dir", type=str, default="./rgb_data")
    parser.add_argument(
        "--start_time",
        type=rospy.rostime.Time,
        default=rospy.rostime.Time(secs=1667330345, nsecs=22710468),
    )  # the first timestamp of odombag
    parser.add_argument(
        "--end_time",
        type=rospy.rostime.Time,
        default=rospy.rostime.Time(secs=1667330357, nsecs=228116),
    )
    parser.add_argument(
        "--translation",
        type=np.array,
        default=CAMERA_CONFIG["new"]["translation"],
    )
    parser.add_argument(
        "--axis_vec",
        type=np.array,
        default=CAMERA_CONFIG["new"]["axis_vec"],
    )

    args = parser.parse_args()
    start_time = args.start_time
    end_time = args.end_time
    depth_dir = args.depth_dir
    rgb_dir = args.rgb_dir
    translation = args.translation
    axis_vec = args.axis_vec

    extract_cube_pose(
        start_time,
        end_time,
        ROSBAG_NAME,
        ODOM_ROSBAG_NAME,
        ODOM_ROS_TOPIC,
        ANNOTATED_POSES_DIR,
        translation,
        axis_vec,
    )  # Get the box pose for the first frame
    # bag_to_depth_images(ROSBAG_NAME, DEPTH_ROS_TOPIC, depth_dir, start_time, end_time)
    # print("Depth images generated")
    # extract_poses_with_timestamps(
    #     ROSBAG_NAME,
    #     DEPTH_ROS_TOPIC,
    #     RGB_ROS_TOPIC,
    #     JOINT_STATE_ROS_TOPIC,
    #     POSITION_FILE_PATH,
    #     rgb_dir,
    #     start_time,
    #     end_time,
    # )
    # frame_num = len([name for name in os.listdir(rgb_dir)])
    # print("There are %i frames in total!" % frame_num)
    # positions = import_data(POSITION_FILE_PATH)
    # write_real_depth_as_txt(
    #     start_frame=1, end_frame=frame_num
    # )  # TODO: The last frame is cropped
    # print("Finished writing %i real depth text files." % frame_num)
    # for frame_id in tqdm(range(1, frame_num + 1)):
    #     run_urdf_filter(frame_id, positions)
    #     dilate(frame_id)
    #     generate_depth_img_without_robot(
    #         REAL_DEPTH_FILE % frame_id,
    #         MASK_IAMGE_FILE % frame_id,
    #         FILTERED_DEPTH_FILE % frame_id,
    #     )
    #     # generate_rgb_image_without_robot(RGB_IMAGE_FILE, MASK_IAMGE_FILE, FILTERED_RGB_FILE) #optional, seems the point cloud looks fine with the unfiltered rgb data
    #     depth_filter = DepthFilter(frame_id)
    #     depth_filter.visualize_depth_image()
    #     denoise(frame_id)
    #     create_annotated_poses(output_dir=ANNOTATED_POSES_DIR, frame_id=frame_id)

    # extract_gt_poses_from_tagslam(
    #     start_time,
    #     end_time,
    #     depth_bag_file=ROSBAG_NAME,
    #     odom_bag_file=ODOM_ROSBAG_NAME,
    #     depth_topic=DEPTH_ROS_TOPIC,
    #     odom_topic=ODOM_ROS_TOPIC,
    #     output_dir=TAGSLAM_POSES_DIR,
    # )
