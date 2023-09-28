#!/usr/bin/env python

import rosbag
import rospy
from datetime import timedelta

time_offset = 120.18
with rosbag.Bag('adjusted_odom_50.bag', 'w') as outbag:
    for topic, msg, t in rosbag.Bag('./rosbags/odom_50.bag').read_messages():
        
        # Adjust the timestamp by 20 seconds
        t_shifted = t + rospy.Duration(time_offset)

        # Write the modified message to the new bag
        outbag.write(topic, msg, t_shifted)

print("Timestamp adjustment complete!")