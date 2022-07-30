"""Extract images and poses from a rosbag.
"""

import os
import argparse
import time
import cv2
import rosbag
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

def bag_to_images():
    """Extract a folder of images from a rosbag.
    Command to run: python3 bag_to_images.py raw_10.bag /data /camera/depth/image_rect_raw
    """
    parser = argparse.ArgumentParser(description="Extract images from a ROS bag.")
    parser.add_argument("bag_file", help="Input ROS bag.")
    parser.add_argument("output_dir", help="Output directory.")
    parser.add_argument("image_topic", help="Image topic.")
    args = parser.parse_args()

    print("Extract images from %s on topic %s into %s" % (args.bag_file,
                                                          args.image_topic, args.output_dir))

    bag = rosbag.Bag(args.bag_file, "r")
    bridge = CvBridge()
    count = 0
    for topic, msg, t in bag.read_messages(topics=[args.image_topic]):
        cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

        cv2.imwrite(os.path.join(args.output_dir, "frame%06i.png" % count), cv_img)
        print("Wrote image %i" % count)

        count += 1
        # Rosbag is too large, we only keep the first 10 images
        if count == 10:
            break 

    bag.close()

    return


def bag_to_pose(bagfile, pose_topic, out_filename):
    """
    Write /joint_states to a file
    """
    # To make sure we read images and poses approximately at the same rate
    offset = 32
    n = 0
    f = open(out_filename, 'w')
    f.write('# timestamp tx ty tz qx qy qz qw\n')
    with rosbag.Bag(bagfile, 'r') as bag:
        prev = 0
        for (topic, msg, ts) in bag.read_messages(topics=str(pose_topic)):
            if msg.header.seq - prev <= offset:
                continue
            prev = msg.header.seq
            print(prev)
            f.write('%.12f \n position: %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f \n \
            velocity: %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f \n' %
                    (msg.header.stamp.to_sec(),
                    msg.position[0],
                    msg.position[1],
                    msg.position[2],
                    msg.position[3],
                    msg.position[4],
                    msg.position[5],
                    msg.position[6],
                    msg.position[7],
                    msg.position[8],
                    msg.velocity[0],
                    msg.velocity[1],
                    msg.velocity[2],
                    msg.velocity[3],
                    msg.velocity[4],
                    msg.velocity[5],
                    msg.velocity[6],
                    msg.velocity[7],
                    msg.velocity[8]))
            n += 1
            if n == 10:
                break
    print('wrote ' + str(n) + ' imu messages to the file: ' + out_filename) 

if __name__ == '__main__':
    # bag_to_images()
    bag_to_pose("raw_10.bag", "/joint_states", "./data/joint_states.txt")