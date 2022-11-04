import argparse

from tqdm import tqdm
from depth_filter import DepthFilter
from file_utils import denoise, generate_depth_img_without_robot, generate_rgb_image_without_robot, import_data, write_real_depth_as_txt

from rosbag_processor import bag_to_depth_images, extract_poses_with_timestamps
import rospy
from urdf_filter import dilate, run_urdf_filter
import os, os.path

ROSBAG_NAME = "raw_19.bag"
DEPTH_ROS_TOPIC = "/camera/aligned_depth_to_color/image_raw"
JOINT_STATE_ROS_TOPIC = "/joint_states"
RGB_ROS_TOPIC = "/camera/color/image_raw"

POSITION_FILE_PATH = "./texts/joint_position.txt"
REAL_DEPTH_FILE = ""
MASK_IAMGE_FILE = ""
FILTERED_DEPTH_FILE = ""
FILTERED_RGB_FILE = ""


if __name__=='__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--depth_dir', type=str, default='./depth_data')
  parser.add_argument('--rgb_dir', type=str, default='./rgb_data')
  parser.add_argument('--start_time', type=rospy.rostime.Time, default=rospy.rostime.Time(secs=1667330340, nsecs=140524))
  parser.add_argument('--end_time', type=rospy.rostime.Time, default=rospy.rostime.Time(secs=1667330357, nsecs=228116))


  args = parser.parse_args()
  start_time = args.start_time
  end_time = args.end_time
  depth_dir = args.depth_dir
  rgb_dir = args.rgb_dir
  
  # bag_to_depth_images(ROSBAG_NAME, DEPTH_ROS_TOPIC, depth_dir, start_time, end_time)
  # print("Depth images generated")
  # extract_poses_with_timestamps(ROSBAG_NAME, DEPTH_ROS_TOPIC, RGB_ROS_TOPIC, JOINT_STATE_ROS_TOPIC, POSITION_FILE_PATH, rgb_dir, start_time, end_time)
  # write_real_depth_as_txt(start_frame)
  frame_num = len([name for name in os.listdir(rgb_dir)])
  print("There are %i frames in total!" % frame_num)
  positions = import_data(POSITION_FILE_PATH)
  for frame_id in tqdm(range(1, frame_num+1)):
    run_urdf_filter(frame_id, positions)
    dilate(frame_id)
    # generate_depth_img_without_robot(REAL_DEPTH_FILE, MASK_IAMGE_FILE, FILTERED_DEPTH_FILE)
    # generate_rgb_image_without_robot(RGB_IMAGE_FILE, MASK_IAMGE_FILE, FILTERED_RGB_FILE) #optional, seems the point cloud looks fine with the unfiltered rgb data
    depth_filter = DepthFilter(frame_id)
    depth_filter.visualize()
    denoise(frame_id)
    break
  