import os
from example import CAMERA_CONFIG
import rospy
import numpy as np
import scipy.spatial as sp
import matplotlib.pyplot as plt

from math_utils import (
    camera_to_world,
    rotation_matrix_to_euler,
    transform_bundletrack_output_to_world,
)
from rosbag_processor import extract_time_versus_poses

GT_POSE_DIR = "/home/cnets-vision/mengti_ws/robot_filter/dataset/split/1/tagslam_poses/"
OUTPUT_POSE_DIR = "/home/cnets-vision/mengti_ws/results/poses_1/"
ODOM_FILE_PATH = "/home/cnets-vision/mengti_ws/BundleTrack/Data/YCBINEOAT/contact_nets_new_split/1/annotated_poses/"


def get_cosine_sim(frame_id):
    """Compare the output of BundleTrack with the ground-truth poses of tagslam."""
    gt_pose = np.loadtxt(GT_POSE_DIR + "%04i.txt" % frame_id)
    output_pose = np.loadtxt(OUTPUT_POSE_DIR + "%04i.txt" % frame_id)
    return 1 - sp.distance.cdist(gt_pose, output_pose, "cosine")


def get_angle(P, Q):
    R = np.dot(P, Q.T)
    theta = (np.trace(R) - 1) / 2
    return np.arccos(theta) * (180 / np.pi)


def plot_xyz(start_frame, end_frame):
    """Plot the x, y, z of BundleTrack output versus ground-truth poses of tagslam."""
    output_x, output_y, output_z = [], [], []  # translation
    gt_x, gt_y, gt_z = [], [], []
    angles = []

    for frame_id in range(start_frame, end_frame + 1):
        gt_frame = frame_id
        gt_pose = np.loadtxt(GT_POSE_DIR + "%04i.txt" % gt_frame)
        output_pose = np.loadtxt(OUTPUT_POSE_DIR + "%04i.txt" % frame_id)
        # output_pose = transform_to_camera(output_pose)
        output_pose = camera_to_world(output_pose)
        angle = get_angle(output_pose[:3, :3], gt_pose[:3, :3])
        angles.append(angle)
        output_x.append(output_pose[0, 3])
        output_y.append(output_pose[1, 3])
        output_z.append(output_pose[2, 3])

        gt_x.append(gt_pose[0, 3])
        gt_y.append(gt_pose[1, 3])
        gt_z.append(gt_pose[2, 3])
    output_x, output_y, output_z = (
        np.array(output_x),
        np.array(output_y),
        np.array(output_z),
    )
    gt_x, gt_y, gt_z = np.array(gt_x), np.array(gt_y), np.array(gt_z)

    x = np.arange(0, output_x.shape[0])
    plt.plot(x, output_x, label="Bundletrack")
    plt.plot(x, gt_x, label="ground-truth")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("Position x")
    plt.legend()
    plt.show()

    plt.plot(x, output_y, label="BundleTrack")
    plt.plot(x, gt_y, label="ground-truth")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("Position y")
    plt.legend()
    plt.show()

    plt.plot(x, output_z, label="BundleTrack")
    plt.plot(x, gt_z, label="ground-truth")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("Position z")
    plt.legend()
    plt.show()

    plt.plot(x, angles)
    plt.xlabel("X-axis")
    plt.ylabel("angle diff (degrees)")
    plt.title("Angle difference")
    plt.show()


def plot_ground_truth(start_frame, end_frame):
    gt_x, gt_y, gt_z = [], [], []
    for frame_id in range(start_frame, end_frame):
        gt_pose = np.loadtxt(GT_POSE_DIR + "%04i.txt" % frame_id)
        gt_x.append(gt_pose[0, 3])
        gt_y.append(gt_pose[1, 3])
        gt_z.append(gt_pose[2, 3])

    gt_x, gt_y, gt_z = np.array(gt_x), np.array(gt_y), np.array(gt_z)
    x = np.arange(0, gt_x.shape[0])
    plt.plot(x, gt_x, label="ground-truth")
    plt.xlabel("Frame")
    plt.ylabel("X axis")
    plt.title("Position x")
    plt.legend()
    plt.show()

    plt.plot(x, gt_y, label="ground-truth")
    plt.xlabel("Frame")
    plt.ylabel("Y axis")
    plt.title("Position y")
    plt.legend()
    plt.show()

    plt.plot(x, gt_z, label="ground-truth")
    plt.xlabel("Frame")
    plt.ylabel("Z axis")
    plt.title("Position z")
    plt.legend()
    plt.show()


def plot_with_time(bundletrack_time, gt_time, bundletrack_pose_dir, gt_pose_dir):
    """Plot the x, y, z of BundleTrack output versus ground-truth poses of tagslam."""
    bundletrack_time = np.array(bundletrack_time)
    # gt_time_pad = np.zeros_like(bundletrack_time)
    # gt_time_pad[:len(gt_time)] = gt_time
    output_x, output_y, output_z = [], [], []  # translation
    gt_x, gt_y, gt_z = [], [], []
    output_alpha, output_beta, output_gamma = [], [], []
    gt_alpha, gt_beta, gt_gamma = [], [], []
    frame_num = len([name for name in os.listdir(bundletrack_pose_dir)])
    for frame_id in range(1, frame_num + 1):
        output_pose = np.loadtxt(bundletrack_pose_dir + "%04i.txt" % frame_id)
        # output_pose = transform_to_camera(output_pose)
        output_pose = transform_bundletrack_output_to_world(
            output_pose,
            CAMERA_CONFIG["new"]["translation"],
            CAMERA_CONFIG["new"]["axis_vec"],
            bundletrack_pose_dir,
            ODOM_FILE_PATH,
        )
        output_x.append(output_pose[0, 3])
        output_y.append(output_pose[1, 3])
        output_z.append(output_pose[2, 3])
        alpha, beta, gamma = rotation_matrix_to_euler(output_pose[:3, :3])
        output_alpha.append(alpha)
        output_beta.append(beta)
        output_gamma.append(gamma)

    for frame_id in range(1, len(gt_time) + 1):
        gt_frame = frame_id
        gt_pose = np.loadtxt(gt_pose_dir + "%04i.txt" % gt_frame)
        gt_x.append(gt_pose[0, 3])
        gt_y.append(gt_pose[1, 3])
        gt_z.append(gt_pose[2, 3])
        alpha, beta, gamma = rotation_matrix_to_euler(gt_pose[:3, :3])
        gt_alpha.append(alpha)
        gt_beta.append(beta)
        gt_gamma.append(gamma)

    output_x, output_y, output_z = (
        np.array(output_x),
        np.array(output_y),
        np.array(output_z),
    )
    gt_x, gt_y, gt_z = np.array(gt_x), np.array(gt_y), np.array(gt_z)

    # fig = plt.figure()
    fig, axs = plt.subplots(2, 3)
    axs[0, 0].plot(bundletrack_time, output_x, label="BundleTrack")
    axs[0, 0].plot(gt_time, gt_x, label="ground-truth")
    axs[0, 0].set_title("Position x")
    axs[0, 0].legend()
    # plt.show()

    axs[0, 1].plot(bundletrack_time, output_y, label="BundleTrack")
    axs[0, 1].plot(gt_time, gt_y, label="ground-truth")
    axs[0, 1].set_title("Position y")
    axs[0, 1].legend()
    # plt.show()

    axs[0, 2].plot(bundletrack_time, output_z, label="BundleTrack")
    axs[0, 2].plot(gt_time, gt_z, label="ground-truth")
    axs[0, 2].set_title("Position z")
    axs[0, 2].legend()

    axs[1, 0].plot(bundletrack_time, output_alpha, label="BundleTrack")
    axs[1, 0].plot(gt_time, gt_alpha, label="ground-truth")
    axs[1, 0].set_title("Raw")

    axs[1, 1].plot(bundletrack_time, output_beta, label="BundleTrack")
    axs[1, 1].plot(gt_time, gt_beta, label="ground-truth")
    axs[1, 1].set_title("Pitch")

    axs[1, 2].plot(bundletrack_time, output_gamma, label="BundleTrack")
    axs[1, 2].plot(gt_time, gt_gamma, label="ground-truth")
    axs[1, 2].set_title("Yaw")

    spacing = 0.100
    fig.subplots_adjust(bottom=spacing)
    plt.show()


if __name__ == "__main__":
    # plot_xyz(1, 950)
    depth_bag_file = "./raw_19.bag"
    odom_bag_file = "./odom_19.bag"
    DEPTH_ROS_TOPIC = "/camera/aligned_depth_to_color/image_raw"
    ODOM_ROS_TOPIC = "/tagslam/odom/body_box"
    # start_time = rospy.rostime.Time(secs=1667330345, nsecs=22710468)
    # end_time = rospy.rostime.Time(secs=1667330357, nsecs=228116)
    start_time = rospy.rostime.Time(secs=1667330342, nsecs=959312)
    end_time = rospy.rostime.Time(secs=1667330357, nsecs=453744)
    bundletrack_time, gt_time = extract_time_versus_poses(
        start_time,
        end_time,
        depth_bag_file,
        odom_bag_file,
        DEPTH_ROS_TOPIC,
        ODOM_ROS_TOPIC,
        GT_POSE_DIR,
    )
    plot_with_time(bundletrack_time, gt_time, OUTPUT_POSE_DIR, GT_POSE_DIR)

    ###TEST
    # camera_pos = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]])
    # print(camera_to_world(camera_pos))

    # from PIL import Image
    # import imageio
    # img_dir = "./depth_data/0001.png"
    # # img_dir= "/home/cnets-vision/mengti_ws/BundleTrack/Data/YCBINEOAT/bleach0/depth/1578966042136484471.png"
    # # image = Image.open(img_dir)
    # # pixels = list(image.getdata())
    # image = imageio.imread(img_dir)
    # print(image[100][100])
    # print(image.dtype)
