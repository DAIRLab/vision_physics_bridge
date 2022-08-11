"""Extract images and poses from a rosbag.
"""

import os
from tracemalloc import start
import cv2
import rosbag
import rospy
# from sensor_msgs.msg import Image
from PIL import Image
from cv_bridge import CvBridge
import numpy as np
import matplotlib.pyplot as plt

def bag_to_rgb_images(bag_file, image_topic, output_dir, start_frame, end_frame):
    """Extract a folder of RGB images from a rosbag.
    """
    print("Extract images from %s on topic %s into %s" % (bag_file,
                                                          image_topic, 
                                                          output_dir))

    bag = rosbag.Bag(bag_file, "r")
    bridge = CvBridge()
    count = 0
    for topic, msg, t in bag.read_messages(topics=[image_topic]):
        count += 1
        if count < start_frame:
            continue
        cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        cv2.imwrite(os.path.join(output_dir, "frame%06i.png" % count), cv_img)
        # im = Image.fromarray(depth_array)
        # if im.mode != 'RGB':
        #     im = im.convert('RGB')
        
        print("Wrote image %i" % count)

        # Rosbag is too large, we only keep the first 10 images
        if count == end_frame:
            break 
    bag.close()
    return

def bag_to_depth_images(bag_file, image_topic, output_dir, start_frame, end_frame):
    """Extract a folder of depth images from a rosbag.
    Command to run: python3 bag_to_images.py raw_10.bag /data /camera/depth/image_rect_raw
    """
    print("Extract images from %s on topic %s into %s" % (bag_file,
                                                          image_topic, 
                                                          output_dir))

    bag = rosbag.Bag(bag_file, "r")
    bridge = CvBridge()
    count = 0
    arr = []
    for topic, msg, t in bag.read_messages(topics=[image_topic]):
        count += 1
        # Saving frame 100 - 110
        if count < start_frame:
            continue
        cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        depth_array = np.array(cv_img)*0.001
        print(depth_array[250,300])
        
        # This is temporarily not being used
        cv2.imwrite(os.path.join(output_dir, "frame%06i.png" % count), cv_img)
        
        # depth_array = np.array(cv_img, np.float32)
        arr.append(depth_array)
        # im = Image.fromarray(depth_array)
        # if im.mode != 'RGB':
        #     im = im.convert('RGB')
        
        # im.save(os.path.join(output_dir, "frame%06i.png" % count))
        print("Wrote image %i" % count)

        # Rosbag is too large, we only keep the first 10 images
        if count == end_frame:
            break 
    arr = np.array(arr)
    arr_reshaped = arr.reshape(arr.shape[0], -1)
    np.savetxt(IMAGE_FILE_PATH, arr_reshaped)
    loaded_arr = np.loadtxt(IMAGE_FILE_PATH)
    load_original_arr = loaded_arr.reshape(
        loaded_arr.shape[0], loaded_arr.shape[1] // arr.shape[2], arr.shape[2])
    # check the shapes
    print("shape of arr: ", arr.shape)
    print("shape of load_original_arr: ", load_original_arr.shape)
    bag.close()
    return


def bag_to_pose(bagfile, pose_topic, outfile_position, outfile_velocity, start_frame, end_frame):
    """
    Write /joint_states to a file
    """
    # To make sure we read images and poses approximately at the same rate
    offset = 32
    n = start_frame
    position = []
    velocity = []
    with rosbag.Bag(bagfile, 'r') as bag:
        prev = start_frame
        for (topic, msg, ts) in bag.read_messages(topics=str(pose_topic)):
            if msg.header.seq < n:
                continue
            if msg.header.seq - prev <= offset:
                continue
            prev = msg.header.seq
            print("recording frame: ", msg.header.seq)
            reordered_position = [0] * len(msg.position)
            reordered_position[:7] = msg.position[2:]
            reordered_position[-2:] = msg.position[:2]
            reordered_velocity = [0] * len(msg.velocity)
            reordered_velocity[:7] = msg.velocity[2:]
            reordered_velocity[-2:] = msg.velocity[:2]
            
            position.append(reordered_position)
            velocity.append(reordered_velocity)
            n += 1
            if n == end_frame:
                break
        position = np.array(position)
        velocity = np.array(velocity)
        np.savetxt(outfile_position, position)
        np.savetxt(outfile_velocity, velocity)
        # check the shapes
        loaded_position = np.loadtxt(outfile_position)
        loaded_velocity = np.loadtxt(outfile_velocity)
        print("shape of position: ", position.shape)
        print("shape of loaded_position: ", loaded_position.shape)
        print("shape of velocity: ", velocity.shape)
        print("shape of loaded_velocity: ", loaded_velocity.shape)
    print('wrote ' + str(n) + ' imu messages to the file: ' + outfile_position + ' and ' + outfile_velocity) 

if __name__ == '__main__':
    DEPTH_ROS_TOPIC = "/camera/aligned_depth_to_color/image_raw"
    JOINT_STATE_ROS_TOPIC = "/joint_states"
    RGB_ROS_TOPIC = "/camera/color/image_raw"

    IMAGE_FILE_PATH = "./aligned_data/images.txt"
    POSITION_FILE_PATH = "./aligned_data/joint_position.txt"
    VELOCITY_FILE_PATH = "./aligned_data/joint_velocity.txt"

    DEPTH_OUTPUT_DIR = "./aligned_data"
    RGB_OUTPUT_DIR = "./rgb_data"
    ROSBAG_NAME = "raw_10.bag"
    bag_to_rgb_images(ROSBAG_NAME, RGB_ROS_TOPIC, RGB_OUTPUT_DIR, 0, 10)
    # bag_to_depth_images(ROSBAG_NAME, DEPTH_ROS_TOPIC, DEPTH_OUTPUT_DIR, 0, 10)
    # bag_to_pose(ROSBAG_NAME, JOINT_STATE_ROS_TOPIC, POSITION_FILE_PATH, VELOCITY_FILE_PATH, 0, 10)