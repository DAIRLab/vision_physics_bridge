import rospy
import message_filters
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
import numpy as np

start_time = rospy.rostime.Time(secs=1655404893, nsecs=899137)  # toss 1
end_time = rospy.rostime.Time(secs=1655404908, nsecs=279948)
tagslam_dir = "./dataset/old_dataset/tagslam_poses/"


class Synchronizer:
    def __init__(self) -> None:
        rospy.init_node("listener", anonymous=True)
        self.depth_sub = message_filters.Subscriber(
            "/camera/aligned_depth_to_color/image_raw", Image
        )
        self.odom_sub = message_filters.Subscriber("/tagslam/odom/body_cube", Odometry)
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.depth_sub, self.odom_sub], 1000, 10
        )
        self.frame = 0
        self.bundletrack_time = []
        self.gt_time = []
        self.ts.registerCallback(self.callback)
        rospy.spin()

    def callback(self, depth_msg, odom_msg):
        # print("Inside callback")
        if start_time <= depth_msg.header.stamp <= end_time:
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
            np.savetxt(tagslam_dir + "%04i.txt" % self.frame, Q)
            self.bundletrack_time.append(depth_msg.header.stamp.to_nsec())
            self.gt_time.append(odom_msg.header.stamp.to_nsec())
        if self.frame == 449:  # TODO
            rospy.signal_shutdown("Shutting down the node")


if __name__ == "__main__":
    c = Synchronizer()
