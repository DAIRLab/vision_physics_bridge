#!/usr/bin/env python

import rosbag
import rospy
from datetime import timedelta
import argparse 

parser = argparse.ArgumentParser()
parser.add_argument(
    "--offset",
    type=float,
    required=True,
)
args = parser.parse_args()
offset = args.offset
time_offset = offset

with rosbag.Bag('adjusted_odom_57.bag', 'w') as outbag:
    for topic, msg, t in rosbag.Bag('./rosbags/odom_57.bag').read_messages():
        
        # Adjust the timestamp by 20 seconds
        t_shifted = t + rospy.Duration(time_offset)

        # Write the modified message to the new bag
        outbag.write(topic, msg, t_shifted)

print("Timestamp adjustment complete!")