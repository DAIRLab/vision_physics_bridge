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
    load_toss_time_from_yaml,
    write_real_depth_as_txt,
)
from math_utils import pos_quat_to_trans_mat, world_to_camera
from pydrake.all import StartMeshcat
from rosbag_processor import (
    bag_to_depth_images,
    bag_to_rgb_images,
    extract_cube_pose,
    extract_gt_poses_from_tagslam,
    extract_gt_poses_from_tagslam_with_quat,
    extract_poses_with_timestamps,
    extract_gt_poses_from_tagslam_with_missing_frames,
)
import rospy
from sync_data import Synchronizer
from urdf_filter import dilate, run_urdf_filter
import os, os.path
import yaml
from scipy.spatial.transform import Rotation as R
import shutil
"""Process the milk hand-tossing data.
"""
TOSS_TYPE = 'milk'
TOSS_ID=1
ROSBAG_NAME = "./rosbags/raw_66.bag"
ODOM_ROSBAG_NAME = "./rosbags/odom_66.bag"
DEPTH_ROS_TOPIC = "/camera/aligned_depth_to_color/image_raw"
ODOM_ROS_TOPIC = "/tagslam/odom/body_milk"
JOINT_STATE_ROS_TOPIC = "/joint_states"
RGB_ROS_TOPIC = "/camera/color/image_raw"

ROOT_DIR = f"./dataset/{TOSS_TYPE}_{TOSS_ID}/"
BUNDLETRACK_DATA_DIR = f"{TOSS_TYPE}_{TOSS_ID}/"
BUNDLETRACK_DIR = "/home/cnets-vision/mengti_ws/BundleSDF/data/"
CAMERA_EXTRINSICS_FILE = "./assets/realsense_pose_milk_prism.yaml"

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

TEXT_PATH = ROOT_DIR + "text"
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
DENOISE_MASK_DIR = BUNDLETRACK_DIR + BUNDLETRACK_DATA_DIR + "masks"
ANNOTATED_POSES_DIR = BUNDLETRACK_DIR + BUNDLETRACK_DATA_DIR + "annotated_poses"
BUNDLETRACK_DEPTH = BUNDLETRACK_DIR + BUNDLETRACK_DATA_DIR + "depth"
BUNDLETRACK_RGB = BUNDLETRACK_DIR + BUNDLETRACK_DATA_DIR + "rgb"

cam = 'cam0' # realsense camera name
with open(CAMERA_EXTRINSICS_FILE, 'r') as stream:
    data_loaded = yaml.safe_load(stream)
print(data_loaded[cam]['pose']['position'])

cam_pos_dict = data_loaded[cam]['pose']['position']
cam_trans = np.array([cam_pos_dict['x'], cam_pos_dict['y'], cam_pos_dict['z']]).reshape(-1, 1)
cam_rot_dict = data_loaded[cam]['pose']['rotation']
cam_axis_vec = np.array([cam_rot_dict['x'], cam_rot_dict['y'], cam_rot_dict['z']])

yaml_path = './assets/config.yaml'
start_time = load_toss_time_from_yaml(yaml_path, TOSS_TYPE, TOSS_ID, 'start_time')
end_time = load_toss_time_from_yaml(yaml_path, TOSS_TYPE, TOSS_ID, 'end_time')
print(f'start_time:{start_time.secs}.{start_time.nsecs}, end_time:{end_time.secs}.{end_time.nsecs}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--start_time",
        type=rospy.rostime.Time,
        required=False,
        default=start_time
    )
    parser.add_argument(
        "--end_time",
        type=rospy.rostime.Time,
        required=False,
        default=end_time
    )
    parser.add_argument(
        "--translation",
        type=np.array,
        default=cam_trans,
    )
    parser.add_argument(
        "--axis_vec",
        type=np.array,
        default=cam_axis_vec,
    )

    args = parser.parse_args()
    start_time = args.start_time
    end_time = args.end_time
    translation = args.translation
    axis_vec = args.axis_vec

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
    # Since we don't need the masks for this dataset, simply use the old version of rgb processor
    bag_to_rgb_images(ROSBAG_NAME, RGB_ROS_TOPIC, BUNDLETRACK_RGB, start_time, end_time)
    frame_num = len([name for name in os.listdir(BUNDLETRACK_RGB)])
    print(f'frame_num is {frame_num}')
    for frame_id in range(1, frame_num+1):
        create_annotated_poses(output_dir=ANNOTATED_POSES_DIR, frame_id=frame_id)
    # clean up
    try:
        shutil.rmtree(DEPTH_DATA_DIR)
        shutil.rmtree(RGB_DATA_DIR)
        shutil.rmtree(CUBE_SCREEN_DIR)
        shutil.rmtree(CUBE_DEPTH_DIR)
        shutil.rmtree(TEXT_PATH)
        print(f"Done clean up!")
    except Exception as e:
        print(f"Error occurred: {e}")