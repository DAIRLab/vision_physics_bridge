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
from pydrake.all import StartMeshcat
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
# ROS data
ROSBAG_NAME = "raw_19.bag"
ODOM_ROSBAG_NAME = "odom_19.bag"
DEPTH_ROS_TOPIC = "/camera/aligned_depth_to_color/image_raw"
JOINT_STATE_ROS_TOPIC = "/joint_states"
RGB_ROS_TOPIC = "/camera/color/image_raw"
ODOM_ROS_TOPIC = "/tagslam/odom/body_box"

ROOT_DIR = "./dataset/new_split/1/"
BUNDLETRACK_DATA_DIR = "./contact_nets_new_split/1/"
BUNDLETRACK_DIR = "/home/cnets-vision/mengti_ws/BundleTrack/Data/YCBINEOAT/"

# Create folders
if not os.path.exists(ROOT_DIR + "texts"):
    os.makedirs(ROOT_DIR + "texts")
if not os.path.exists(ROOT_DIR + "depth_data"):
    os.makedirs(ROOT_DIR + "depth_data")
if not os.path.exists(ROOT_DIR + "rgb_data"):
    os.makedirs(ROOT_DIR + "rgb_data")
if not os.path.exists(ROOT_DIR + "cube_data"):
    os.makedirs(ROOT_DIR + "cube_data")
if not os.path.exists(ROOT_DIR + "mask_data"):
    os.makedirs(ROOT_DIR + "mask_data")
if not os.path.exists(ROOT_DIR + "dilated_mask_data"):
    os.makedirs(ROOT_DIR + "dilated_mask_data")
if not os.path.exists(ROOT_DIR + "filtered_data"):
    os.makedirs(ROOT_DIR + "filtered_data")
if not os.path.exists(ROOT_DIR + "tagslam_poses"):
    os.makedirs(ROOT_DIR + "tagslam_poses")
if not os.path.exists(BUNDLETRACK_DIR + BUNDLETRACK_DATA_DIR + "annotated_poses"):
    os.makedirs(BUNDLETRACK_DIR + BUNDLETRACK_DATA_DIR + "annotated_poses")
if not os.path.exists(BUNDLETRACK_DIR + BUNDLETRACK_DATA_DIR + "depth"):
    os.makedirs(BUNDLETRACK_DIR + BUNDLETRACK_DATA_DIR + "depth")
if not os.path.exists(BUNDLETRACK_DIR + BUNDLETRACK_DATA_DIR + "masks"):
    os.makedirs(BUNDLETRACK_DIR + BUNDLETRACK_DATA_DIR + "masks")
if not os.path.exists(BUNDLETRACK_DIR + BUNDLETRACK_DATA_DIR + "rgb"):
    os.makedirs(BUNDLETRACK_DIR + BUNDLETRACK_DATA_DIR + "rgb")


POSITION_FILE_PATH = ROOT_DIR + "texts/joint_position.txt"
REAL_DEPTH_FILE = ROOT_DIR + "texts/real_depth_frame%04i.txt"
SIMULATED_DEPTH_FILE = ROOT_DIR + "texts/simulated_depth_frame%04i.txt"
IMAGE_TXT_PATH = ROOT_DIR + "texts/images.txt"  # depth image in the form of txt

DEPTH_DATA_DIR = ROOT_DIR + "depth_data/"
RGB_DATA_DIR = ROOT_DIR + "rgb_data/"
CUBE_SCREEN_DIR = ROOT_DIR + "cube_data/screen_image_frame%04i.png"
CUBE_DEPTH_DIR = ROOT_DIR + "cube_data/depth_image_frame%04i.png"
MASK_IAMGE_FILE = ROOT_DIR + "mask_data/%04i.png"
DILATED_MASK_FILE = ROOT_DIR + "dilated_mask_data/%04i.png"
FILTERED_DEPTH_FILE = ROOT_DIR + "filtered_data/depth_without_robot_frame%04i.png"
FILTERED_RGB_FILE = ROOT_DIR + "filtered_data/rgb_without_robot_frame%04i.png"
TAGSLAM_POSES_DIR = ROOT_DIR + "tagslam_poses/"

# BundleTrack data paths
DENOISE_MASK_DIR = (
    "/home/cnets-vision/mengti_ws/BundleTrack/Data/YCBINEOAT/"
    + BUNDLETRACK_DATA_DIR
    + "masks"
)
ANNOTATED_POSES_DIR = (
    "/home/cnets-vision/mengti_ws/BundleTrack/Data/YCBINEOAT/"
    + BUNDLETRACK_DATA_DIR
    + "annotated_poses"
)
BUNDLETRACK_DEPTH = (
    "/home/cnets-vision/mengti_ws/BundleTrack/Data/YCBINEOAT/"
    + BUNDLETRACK_DATA_DIR
    + "depth"
)
BUNDLETRACK_RGB = (
    "/home/cnets-vision/mengti_ws/BundleTrack/Data/YCBINEOAT/"
    + BUNDLETRACK_DATA_DIR
    + "rgb"
)

# For camera extrinsics
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
    parser.add_argument(
        "--start_time",
        type=rospy.rostime.Time,
        required=False,
        default=rospy.rostime.Time(secs=1667330342, nsecs=959312)  # toss 1
        # default = rospy.rostime.Time(secs=1667330357, nsecs=453744)  # toss 2
        # default = rospy.rostime.Time(secs=1667330379, nsecs=657584)  # toss 3
        # default = rospy.rostime.Time(secs=1667330401, nsecs=359995)  # toss 4
        # default = rospy.rostime.Time(secs=1667330422, nsecs=301875)  # toss 5
        # default = rospy.rostime.Time(secs=1667330442, nsecs=284764)  # toss 6
        # default = rospy.rostime.Time(secs=1667330465, nsecs=216068)  # toss 7
        # default = rospy.rostime.Time(secs=1667330490, nsecs=318756)  # toss 8
    )  # the first timestamp of odombag
    parser.add_argument(
        "--end_time",
        type=rospy.rostime.Time,
        required=False,
        default=rospy.rostime.Time(secs=1667330357, nsecs=453744)  # toss 1
        # default = rospy.rostime.Time(secs=1667330379, nsecs=657584)  # toss 2
        # default = rospy.rostime.Time(secs=1667330401, nsecs=359995)  # toss 3
        # default = rospy.rostime.Time(secs=1667330422, nsecs=301875)  # toss 4
        # default = rospy.rostime.Time(secs=1667330442, nsecs=284764)  # toss 5
        # default = rospy.rostime.Time(secs=1667330465, nsecs=216068)  # toss 6
        # default = rospy.rostime.Time(secs=1667330490, nsecs=318756)  # toss 7
        # default = rospy.rostime.Time(secs=1667330509, nsecs=187270)  # toss 8
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
    bag_to_depth_images(
        ROSBAG_NAME,
        DEPTH_ROS_TOPIC,
        DEPTH_DATA_DIR,
        start_time,
        end_time,
        img_dir=IMAGE_TXT_PATH,
        bundletrack_depth_dir=BUNDLETRACK_DEPTH,
    )
    print("Depth images generated")
    extract_poses_with_timestamps(
        ROSBAG_NAME,
        DEPTH_ROS_TOPIC,
        RGB_ROS_TOPIC,
        JOINT_STATE_ROS_TOPIC,
        POSITION_FILE_PATH,
        RGB_DATA_DIR,
        start_time,
        end_time,
        bundletrack_rgb_dir=BUNDLETRACK_RGB,
    )
    frame_num = len([name for name in os.listdir(RGB_DATA_DIR)])
    print("There are %i frames in total!" % frame_num)
    positions = import_data(POSITION_FILE_PATH)
    write_real_depth_as_txt(
        start_frame=1,
        end_frame=frame_num,
        img_dir=IMAGE_TXT_PATH,
        real_depth_dir=REAL_DEPTH_FILE,
    )  # TODO: The last frame is cropped
    print("Finished writing %i real depth text files." % frame_num)
    meshcat = StartMeshcat()
    for frame_id in tqdm(range(1, frame_num + 1)):
        run_urdf_filter(
            meshcat,
            frame_id,
            positions,
            MASK_IAMGE_FILE,
            SIMULATED_DEPTH_FILE,
            REAL_DEPTH_FILE,
            translation,
            axis_vec,
        )
        dilate(
            frame_id, mask_image_dir=MASK_IAMGE_FILE, dilated_mask_dir=DILATED_MASK_FILE
        )
        generate_depth_img_without_robot(
            REAL_DEPTH_FILE % frame_id,
            MASK_IAMGE_FILE % frame_id,
            FILTERED_DEPTH_FILE % frame_id,
        )
        # generate_rgb_image_without_robot(RGB_IMAGE_FILE, MASK_IAMGE_FILE, FILTERED_RGB_FILE) #optional, seems the point cloud looks fine with the unfiltered rgb data
        depth_filter = DepthFilter(
            frame_id,
            RGB_DATA_DIR + "%04i.png",
            FILTERED_DEPTH_FILE,
            CUBE_SCREEN_DIR,
            CUBE_DEPTH_DIR,
            translation,
            axis_vec,
        )
        depth_filter.visualize_depth_image()
        denoise(
            frame_id,
            img_dir=CUBE_DEPTH_DIR,
            denoise_mask_dir=DENOISE_MASK_DIR,
            region=(11, 11),
        )
        create_annotated_poses(output_dir=ANNOTATED_POSES_DIR, frame_id=frame_id)

    extract_gt_poses_from_tagslam(
        start_time,
        end_time,
        depth_bag_file=ROSBAG_NAME,
        odom_bag_file=ODOM_ROSBAG_NAME,
        depth_topic=DEPTH_ROS_TOPIC,
        odom_topic=ODOM_ROS_TOPIC,
        output_dir=TAGSLAM_POSES_DIR,
    )

# Problematic frames:
# 2384
# 3505 - 3547
# 4467
