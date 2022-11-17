"""Process the cube data.
"""
import argparse

from tqdm import tqdm
from depth_filter import DepthFilter
from file_utils import create_annotated_poses, denoise, generate_depth_img_without_robot, generate_rgb_image_without_robot, import_data, write_real_depth_as_txt

from rosbag_processor import bag_to_depth_images, extract_cube_pose, extract_gt_poses_from_tagslam, extract_poses_with_timestamps
import rospy
from urdf_filter import dilate, run_urdf_filter
import os, os.path

ROSBAG_NAME = "raw_10.bag"
ODOM_ROSBAG_NAME = "odom_10.bag"
DEPTH_ROS_TOPIC = "/camera/aligned_depth_to_color/image_raw"
JOINT_STATE_ROS_TOPIC = "/joint_states"
RGB_ROS_TOPIC = "/camera/color/image_raw"
ODOM_ROS_TOPIC = "/tagslam/odom/body_cube"

POSITION_FILE_PATH = "./old/texts/joint_position.txt"
REAL_DEPTH_FILE = "./old/texts/real_depth_frame%04i.txt"
SIMULATED_DEPTH_FILE = "./old/texts/simulated_depth_frame%04i.txt"
IMAGE_TXT_PATH = "./old/texts/images.txt" #depth image in the form of txt
RGB_PATH = './old/rgb_data/%04i.png'
CUBE_SCREEN_DIR = "./old/cube_data/screen_image_frame%04i.png"
CUBE_DEPTH_DIR = "./old/cube_data/depth_image_frame%04i.png"
MASK_IAMGE_FILE = "./old/mask_data/%04i.png"
DILATED_MASK_FILE = "./old/dilated_mask_data/%04i.png"
FILTERED_DEPTH_FILE = "./old/filtered_data/depth_without_robot_frame%04i.png"
FILTERED_RGB_FILE = "./old/filtered_data/rgb_without_robot_frame%04i.png"
ANNOTATED_POSES_DIR = "/home/cnets-vision/mengti_ws/BundleTrack/Data/YCBINEOAT/contact_nets_old/annotated_poses/"
DENOISE_MASK_DIR = "/home/cnets-vision/mengti_ws/BundleTrack/Data/YCBINEOAT/contact_nets_old/masks/%04i.png"
BUNDLETRACK_DEPTH = "/home/cnets-vision/mengti_ws/BundleTrack/Data/YCBINEOAT/contact_nets_old/depth"
BUNDLETRACK_RGB = "/home/cnets-vision/mengti_ws/BundleTrack/Data/YCBINEOAT/contact_nets_old/rgb"
TAGSLAM_POSES_DIR = "./old/tagslam_poses/"

if __name__=='__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--depth_dir', type=str, default='./old/depth_data')
  parser.add_argument('--rgb_dir', type=str, default='./old/rgb_data')
  # parser.add_argument('--start_time', type=rospy.rostime.Time, default=rospy.rostime.Time(secs=1667330340, nsecs=140524))#The first toss
  parser.add_argument('--start_time', type=rospy.rostime.Time, default=rospy.rostime.Time(secs=1655404895, nsecs=508975))# the first timestamp of odombag
  parser.add_argument('--end_time', type=rospy.rostime.Time, default=rospy.rostime.Time(secs=1655404906, nsecs=972729))

  args = parser.parse_args()
  start_time = args.start_time
  end_time = args.end_time
  depth_dir = args.depth_dir
  rgb_dir = args.rgb_dir

  # extract_cube_pose(start_time, end_time, ROSBAG_NAME, ODOM_ROSBAG_NAME, ODOM_ROS_TOPIC, odom_file_path=ANNOTATED_POSES_DIR)#Get the box pose for the first frame
  # bag_to_depth_images(ROSBAG_NAME, DEPTH_ROS_TOPIC, depth_dir, start_time, end_time, img_dir=IMAGE_TXT_PATH, bundletrack_depth_dir=BUNDLETRACK_DEPTH)
  # print("Depth images generated")
  extract_poses_with_timestamps(ROSBAG_NAME, DEPTH_ROS_TOPIC, RGB_ROS_TOPIC, JOINT_STATE_ROS_TOPIC, POSITION_FILE_PATH, rgb_dir, start_time, end_time, bundletrack_rgb_dir=BUNDLETRACK_RGB)
  frame_num = len([name for name in os.listdir(rgb_dir)])
  print("There are %i frames in total!" % frame_num)
  positions = import_data(POSITION_FILE_PATH)
  # write_real_depth_as_txt(start_frame=1, end_frame=frame_num, img_dir=IMAGE_TXT_PATH, real_depth_dir=REAL_DEPTH_FILE)#TODO: The last frame is cropped
  print("Finished writing %i real depth text files." % frame_num)
  # for frame_id in tqdm(range(301, frame_num+1)):
  #   run_urdf_filter(frame_id, positions,mask_image_dir=MASK_IAMGE_FILE, simulated_depth_dir=SIMULATED_DEPTH_FILE, real_depth_dir=REAL_DEPTH_FILE)
  #   dilate(frame_id, mask_image_dir=MASK_IAMGE_FILE, dilated_mask_dir=DILATED_MASK_FILE)
  #   generate_depth_img_without_robot(REAL_DEPTH_FILE%frame_id, MASK_IAMGE_FILE%frame_id, FILTERED_DEPTH_FILE%frame_id)
  #   # generate_rgb_image_without_robot(RGB_IMAGE_FILE, MASK_IAMGE_FILE, FILTERED_RGB_FILE) #optional, seems the point cloud looks fine with the unfiltered rgb data
  #   depth_filter = DepthFilter(frame_id, color_image_dir=RGB_PATH, depth_image_dir=FILTERED_DEPTH_FILE, cube_screen_image_dir=CUBE_SCREEN_DIR, cube_depth_image_dir=CUBE_DEPTH_DIR)
  #   depth_filter.visualize_depth_image()
  #   denoise(frame_id, img_dir=CUBE_DEPTH_DIR, denoise_mask_dir=DENOISE_MASK_DIR)
  #   create_annotated_poses(output_dir=ANNOTATED_POSES_DIR, frame_id=frame_id)

  # extract_gt_poses_from_tagslam(start_time, end_time, depth_bag_file=ROSBAG_NAME, odom_bag_file=ODOM_ROSBAG_NAME, depth_topic=DEPTH_ROS_TOPIC, odom_topic=ODOM_ROS_TOPIC, output_dir=TAGSLAM_POSES_DIR)