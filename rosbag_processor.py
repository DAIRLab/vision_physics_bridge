"""Extract images and poses from a rosbag.
"""

import os
import time
from tracemalloc import start
import cv2
import rosbag
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np

def bag_to_images(bag_file, image_topic, output_dir, start_frame, end_frame):
    """Extract a folder of images from a rosbag.
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

        cv2.imwrite(os.path.join(output_dir, "frame%06i.png" % count), cv_img)
        arr.append(np.array(cv_img))
        print("Wrote image %i" % count)

        
        # Rosbag is too large, we only keep the first 10 images
        if count == end_frame:
            break 
    arr = np.array(arr)
    arr_reshaped = arr.reshape(arr.shape[0], -1)
    np.savetxt("./data/images.txt", arr_reshaped)
    loaded_arr = np.loadtxt("./data/images.txt")
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
    # bag_to_images("raw_10.bag", "/camera/depth/image_rect_raw", "./data", 0, 10)
    bag_to_pose("raw_10.bag", "/joint_states", "./data/joint_position.txt", "./data/joint_velocity.txt", 0, 10)