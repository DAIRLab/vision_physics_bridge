import numpy as np
import rospy
import torch

from math_utils import (
    get_angular_velocity,
    get_linear_velocity,
    rotation_matrix_to_quaternion,
)

"""
Classes for generating and managing datasets for ContactNets.
"""


DATA_DIR = "/home/cnets-vision/mengti_ws/BundleTrack/results/ycbineoat/contact-nets"
TIMESTAMP_FILE_PATH = (
    "/home/cnets-vision/mengti_ws/robot_filter/tagslam_poses/timestamps.txt"
)
CONTACTNETS_INPUT_DIR = (
    "/home/cnets-vision/mengti_ws/dair_pll/buddletrack/contactnets_cube/"
)


class DatasetManagement:
    def __init__(self, frame_id, next_frame_id):
        self.frame_id = frame_id
        self.next_frame_id = next_frame_id
        self.pose = np.loadtxt(DATA_DIR + "/poses/%04i.txt" % self.frame_id)
        self.pose_ = np.loadtxt(DATA_DIR + "/poses/%04i.txt" % self.next_frame_id)
        self.timestamps = np.loadtxt(TIMESTAMP_FILE_PATH)
        self.R = self.pose[:3, :3]
        self.R_ = self.pose_[:3, :3]
        self.translation = self.pose[:3, 3].reshape(1, -1)
        self.translation_ = self.pose_[:3, 3].reshape(1, -1)
        self.state = self.transform()

    def transform(self):
        """State vector is 4 quaternion + 3 xyz position + 3 angular velocity + 3 linear velocity."""
        q = rotation_matrix_to_quaternion(self.pose)
        q = q.reshape(1, -1)
        dt = self.get_duration()
        ang_velocity = get_angular_velocity(self.R, self.R_, dt)
        lin_velocity = get_linear_velocity(self.translation, self.translation_, dt)
        print(lin_velocity)
        return np.hstack(
            (np.hstack((q, self.translation)), np.hstack((ang_velocity, lin_velocity)))
        )

    def get_duration(self):
        curr_t, next_t = rospy.Time(), rospy.Time()
        curr_t.secs = int(self.timestamps[self.frame_id][1])
        curr_t.nsecs = int(self.timestamps[self.frame_id][2])
        next_t.secs = int(self.timestamps[self.next_frame_id][1])
        next_t.nsecs = int(self.timestamps[self.next_frame_id][2])
        duration = next_t - curr_t
        return duration.to_sec()


if __name__ == "__main__":
    # frame_id = 1
    # frame_id_ = 2
    state_vec = []
    for frame_id in range(1, 4431):
        frame_id_ = frame_id + 1
        dataset = DatasetManagement(frame_id, frame_id_)
        torch.save(
            torch.tensor(dataset.state),
            CONTACTNETS_INPUT_DIR + "{}.pt".format(frame_id),
        )
