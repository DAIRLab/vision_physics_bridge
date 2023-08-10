import rospy
import message_filters
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
import numpy as np


class Synchronizer:
    def __init__(self, tagslam_dir, data_length, start_time, end_time, save=False) -> None:
        rospy.init_node("listener", anonymous=True)
        self.tagslam_dir = tagslam_dir
        self.data_length = data_length
        self.start_time = start_time
        self.end_time = end_time
        self.save = save
        self.depth_sub = message_filters.Subscriber(
            "/camera/aligned_depth_to_color/image_raw", Image
        )
        self.odom_sub = message_filters.Subscriber("/tagslam/odom/body_bottle", Odometry)
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.depth_sub, self.odom_sub], 1000, 10
        )
        self.frame = 0
        self.bundletrack_time = []
        self.gt_time = []
        self.ts.registerCallback(self.callback)
        rospy.spin()

    def callback(self, depth_msg, odom_msg):
        if self.start_time <= depth_msg.header.stamp < self.end_time:
            self.frame += 1
            print(self.frame)
            # print(f"odom_msg.header.time {odom_msg.header.stamp}")
            position = odom_msg.pose.pose.position
            Q = np.zeros((7, 1))
            Q[0] = position.x
            Q[1] = position.y
            Q[2] = position.z
            Q[3] = odom_msg.pose.pose.orientation.x
            Q[4] = odom_msg.pose.pose.orientation.y
            Q[5] = odom_msg.pose.pose.orientation.z
            Q[6] = odom_msg.pose.pose.orientation.w
            if self.save:
                np.savetxt(self.tagslam_dir + "%04i.txt" % self.frame, Q)
            self.bundletrack_time.append(depth_msg.header.stamp)
            self.gt_time.append(odom_msg.header.stamp)
        if self.frame == self.data_length:  # TODO
            rospy.signal_shutdown("Shutting down the node")


if __name__ == "__main__":
    c = Synchronizer()
