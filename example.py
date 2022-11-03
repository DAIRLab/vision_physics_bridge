import argparse
from tracemalloc import start
from file_utils import generate_depth_img_without_robot, generate_rgb_image_without_robot, write_real_depth_as_txt

from rosbag_processor import bag_to_depth_images, extract_poses_with_timestamps
from urdf_filter import run_urdf_filter

ROSBAG_NAME = "raw10.bag"
DEPTH_ROS_TOPIC = "/camera/aligned_depth_to_color/image_raw"
JOINT_STATE_ROS_TOPIC = "/joint_states"
RGB_ROS_TOPIC = "/camera/color/image_raw"
POSITION_FILE_PATH = "./depth_data/joint_position.txt"
VELOCITY_FILE_PATH = "./depth_data/joint_velocity.txt"

if __name__=='__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--start_frame', type=int, default=1)
  parser.add_argument('--end_frame', type=int, default=4432)
  parser.add_argument('--rgb_dir', type=str, default='./rgb_data')
  # parser.add_argument('--rgb_img_file', type=str, default='rgb_image_file = "./rgb_data/%04i.png"'%frame_id)
  parser.add_argument()


  args = parser.parse_args()
  start_frame = args.start_frame
  end_frame = args.end_frame
  rgb_dir = args.rgb_dir
  
  bag_to_depth_images(start_frame)
  print("Depth images generated")
  extract_poses_with_timestamps(ROSBAG_NAME, DEPTH_ROS_TOPIC, RGB_ROS_TOPIC, JOINT_STATE_ROS_TOPIC, POSITION_FILE_PATH, VELOCITY_FILE_PATH, rgb_dir, start_frame)
  write_real_depth_as_txt(start_frame)
  run_urdf_filter()
  generate_depth_img_without_robot(real_depth_file, mask_image_file, filtered_depth_file)
  generate_rgb_image_without_robot(rgb_image_file, mask_image_file, filtered_rgb_file) #optional, seems the point cloud looks fine with the unfiltered rgb data
