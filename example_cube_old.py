import argparse
import numpy as np
from tqdm import tqdm
from depth_filter import DepthFilter
from file_utils import (
    check_empty_img,
    create_annotated_poses,
    denoise,
    generate_depth_img_without_robot,
    import_data,
    load_toss_time_from_yaml,
    write_real_depth_as_txt,
)
from math_utils import pos_quat_to_trans_mat, world_to_camera
from pydrake.all import StartMeshcat
from rosbag_processor import (
    bag_to_depth_images,
    extract_cube_pose,
    extract_poses_with_timestamps,
)
import rospy
from sync_data import Synchronizer
from urdf_filter import dilate, run_urdf_filter
import os, os.path
import yaml

"""Process the cube data.
"""

parser = argparse.ArgumentParser()
parser.add_argument(
    "--toss_id",
    type=int,
    required=True,
)
args = parser.parse_args()
toss_id = args.toss_id
print(f'Processing toss {toss_id}')

ROSBAG_NAME = "./rosbags/raw_10.bag"
ODOM_ROSBAG_NAME = "./rosbags/odom_10.bag"
DEPTH_ROS_TOPIC = "/camera/aligned_depth_to_color/image_raw"
JOINT_STATE_ROS_TOPIC = "/joint_states"
RGB_ROS_TOPIC = "/camera/color/image_raw"
ODOM_ROS_TOPIC = "/tagslam/odom/body_cube"

ROOT_DIR = f"./dataset/old_toss_{toss_id}/"
BUNDLETRACK_DATA_DIR = f"old_toss_{toss_id}/"
BUNDLETRACK_DIR = "/home/cnets-vision/mengti_ws/BundleSDF/data/"

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
DENOISE_MASK_DIR = BUNDLETRACK_DIR + BUNDLETRACK_DATA_DIR + "masks"
ANNOTATED_POSES_DIR = BUNDLETRACK_DIR + BUNDLETRACK_DATA_DIR + "annotated_poses"
BUNDLETRACK_DEPTH = BUNDLETRACK_DIR + BUNDLETRACK_DATA_DIR + "depth"
BUNDLETRACK_RGB = BUNDLETRACK_DIR + BUNDLETRACK_DATA_DIR + "rgb"
CAMERA_EXTRINSICS_FILE = "./assets/realsense_pose_cube_old.yaml"

cam = 'cam0' # realsense camera name
with open(CAMERA_EXTRINSICS_FILE, 'r') as stream:
    data_loaded = yaml.safe_load(stream)
print(data_loaded[cam]['pose']['position'])

cam_pos_dict = data_loaded[cam]['pose']['position']
cam_trans = np.array([cam_pos_dict['x'], cam_pos_dict['y'], cam_pos_dict['z']]).reshape(-1, 1)
cam_rot_dict = data_loaded[cam]['pose']['rotation']
cam_axis_vec = np.array([cam_rot_dict['x'], cam_rot_dict['y'], cam_rot_dict['z']])
yaml_path = './assets/config.yaml'
toss_type = 'cube'
start_time = load_toss_time_from_yaml(yaml_path, toss_type, toss_id, 'start_time')
end_time = load_toss_time_from_yaml(yaml_path, toss_type, toss_id, 'end_time')

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
frame_num = len([name for name in os.listdir(BUNDLETRACK_RGB)])
print("There are %i frames in total!" % frame_num)
positions = import_data(POSITION_FILE_PATH)
write_real_depth_as_txt(
    start_frame=1,
    end_frame=frame_num,
    img_dir=IMAGE_TXT_PATH,
    real_depth_dir=REAL_DEPTH_FILE,
)
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
        cam_trans,
        cam_axis_vec,
    )
    dilate(
        frame_id, mask_image_dir=MASK_IAMGE_FILE, dilated_mask_dir=DILATED_MASK_FILE
    )
    generate_depth_img_without_robot(
        REAL_DEPTH_FILE % frame_id,
        MASK_IAMGE_FILE % frame_id,
        FILTERED_DEPTH_FILE % frame_id,
    )
    depth_filter = DepthFilter(
        frame_id,
        RGB_DATA_DIR + "%04i.png",
        FILTERED_DEPTH_FILE,
        CUBE_SCREEN_DIR,
        CUBE_DEPTH_DIR,
        cam_trans,
        cam_axis_vec,
    )
    depth_filter.visualize_depth_image()
    denoise(
        frame_id,
        img_dir=CUBE_DEPTH_DIR,
        denoise_mask_dir=DENOISE_MASK_DIR,
        region=(11, 11),
    )
    create_annotated_poses(output_dir=ANNOTATED_POSES_DIR, frame_id=frame_id)
check_empty_img(DENOISE_MASK_DIR)
# do this only once
sync = Synchronizer(TAGSLAM_POSES_DIR, frame_num, start_time, end_time, save=True)
data = np.loadtxt(TAGSLAM_POSES_DIR+'tagslam.txt')
init_pose = data[0, 2:]
init_pose_mat = pos_quat_to_trans_mat(init_pose.T)
init_pose_mat_cam = world_to_camera(init_pose_mat, cam_trans, cam_axis_vec)
np.savetxt(
    os.path.join(ANNOTATED_POSES_DIR, "%04i.txt" % 0), init_pose_mat_cam
)  # save init cube pose in camera frame
print(f'Saved init pose at {os.path.join(ANNOTATED_POSES_DIR, "%04i.txt" % 0)}')