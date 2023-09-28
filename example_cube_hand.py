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
from math_utils import world_to_camera
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
from urdf_filter import dilate, run_urdf_filter
import os, os.path
import yaml
from scipy.spatial.transform import Rotation as R
"""Process the cube hand-tossing data.
"""

ROSBAG_NAME = "./rosbags/raw_50.bag"
DEPTH_ROS_TOPIC = "/camera/aligned_depth_to_color/image_raw"
JOINT_STATE_ROS_TOPIC = "/joint_states"
RGB_ROS_TOPIC = "/camera/color/image_raw"

ROOT_DIR = "./dataset/cube_hand_toss_60_3/"
BUNDLETRACK_DATA_DIR = "cube_hand_toss_60_3/"
BUNDLETRACK_DIR = "/home/cnets-vision/mengti_ws/BundleSDF/data/"
CAMERA_EXTRINSICS_FILE = "./assets/realsense_pose_cube_hand_60_3.yaml"
# POSE_YAML = "./assets/poses_cube_hand_60.yaml"

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

cam = 'cam0' # realsense camera name
with open(CAMERA_EXTRINSICS_FILE, 'r') as stream:
    data_loaded = yaml.safe_load(stream)
print(data_loaded[cam]['pose']['position'])

cam_pos_dict = data_loaded[cam]['pose']['position']
cam_trans = np.array([cam_pos_dict['x'], cam_pos_dict['y'], cam_pos_dict['z']]).reshape(-1, 1)
cam_rot_dict = data_loaded[cam]['pose']['rotation']
cam_axis_vec = np.array([cam_rot_dict['x'], cam_rot_dict['y'], cam_rot_dict['z']])

# CUBE_HALF_LENGTH = 0.1048 / 2
# X_OFFSET = -CUBE_HALF_LENGTH
# Y_OFFSET = 0
# Z_OFFSET = 0#CUBE_HALF_LENGTH
# board = 'surface'
# with open(POSE_YAML, 'r') as stream:
#     data_loaded = yaml.safe_load(stream)
# board_pos = data_loaded['bodies'][1][board]['pose']['position']
# board_trans = np.array([board_pos['x']+X_OFFSET, board_pos['y']+Y_OFFSET, board_pos['z']+Z_OFFSET]).reshape(-1, 1)
# board_rot_dict = data_loaded['bodies'][1][board]['pose']['rotation']
# board_axis_vec = np.array([board_rot_dict['x'], board_rot_dict['y'], board_rot_dict['z']])

# cube_hand_toss_60
# cube_trans = np.array([0.183325781372, -0.0359087938626, 0.0252657499878]).reshape(-1, 1)
# cube_quat = np.array([0.00166957478468, -0.00188674789889, -0.00255240804615, 0.999993568937]) #xyzw

# cube_hand_toss_60_2
# cube_trans = np.array([0.183213660774, -0.0346361918548, 0.0252779084507]).reshape(-1,1)
# cube_quat = np.array([0.00162496697398, -0.00196408637707, 0.00452820745199, 0.999986498501])

# position: 
#       x: 0.22072389542360885
#       y: 0.00558024058296229
#       z: 0.023225527657574168
#     orientation: 
#       x: 0.006720446319847842
#       y: -0.008890129044248997
#       z: 0.0009710776280714613
#       w: 0.9999374271498586

# cube_hand_toss_60_3
cube_trans = np.array([0.22072389542360885, 0.00558024058296229, 0.023225527657574168]).reshape(-1,1)
cube_quat = np.array([0.006720446319847842, -0.008890129044248997, 0.0009710776280714613, 0.9999374271498586])

def save_init_pose(trans, quat, save_dir):
    rot = R.from_quat(quat).as_matrix()
    mat = np.vstack((np.hstack((rot, trans)), np.array([0,0,0,1])))
    mat_cam = world_to_camera(mat, cam_trans, cam_axis_vec)
    np.savetxt(os.path.join(save_dir, "%04i.txt" % 0), mat_cam)
    print(f'Initial pose saved to {save_dir}')

# def save_init_pose(trans, axis_vec, save_dir):
#     rot = R.from_rotvec(axis_vec).as_matrix()
#     mat = np.vstack(
#             (np.hstack((rot, trans)), np.array([0, 0, 0, 1]))
#         )
#     np.savetxt(os.path.join(save_dir, "%04i.txt" % 0), mat)
#     print(f'Initial pose saved to {save_dir}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--start_time",
        type=rospy.rostime.Time,
        required=False,
    )
    parser.add_argument(
        "--end_time",
        type=rospy.rostime.Time,
        required=False,
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

    # extract_cube_pose(
    #     start_time,
    #     end_time,
    #     ROSBAG_NAME,
    #     ODOM_ROSBAG_NAME,
    #     ODOM_ROS_TOPIC,
    #     ANNOTATED_POSES_DIR,
    #     translation,
    #     axis_vec,
    # )  # Get the cube pose for the first frame
    save_init_pose(cube_trans, cube_quat, ANNOTATED_POSES_DIR)
    # bag_to_depth_images(
    #     ROSBAG_NAME,
    #     DEPTH_ROS_TOPIC,
    #     DEPTH_DATA_DIR,
    #     start_time,
    #     end_time,
    #     img_dir=IMAGE_TXT_PATH,
    #     bundletrack_depth_dir=BUNDLETRACK_DEPTH,
    # )
    # print("Depth images generated")
    # # Since we don't need the masks for this dataset, simply use the old version of rgb processor
    # bag_to_rgb_images(ROSBAG_NAME, RGB_ROS_TOPIC, BUNDLETRACK_RGB, start_time, end_time)
    # frame_num = len([name for name in os.listdir(BUNDLETRACK_RGB)])
    # for frame_id in range(1, frame_num+1):
    #     create_annotated_poses(output_dir=ANNOTATED_POSES_DIR, frame_id=frame_id)