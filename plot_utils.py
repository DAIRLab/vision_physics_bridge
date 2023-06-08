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

# TOSS_IDX = 2
# GT_POSE_DIR = (
#     "/home/cnets-vision/mengti_ws/robot_filter/dataset/old_split/%i/tagslam_poses/"
#     % TOSS_IDX
# )
# OUTPUT_POSE_DIR = "/home/cnets-vision/mengti_ws/old_results/poses_%i/" % TOSS_IDX
# ODOM_FILE_PATH = (
#     "/home/cnets-vision/mengti_ws/BundleTrack/Data/YCBINEOAT/contact_nets_old_split/%i/annotated_poses/"
#     % TOSS_IDX
# )
GT_POSE_DIR = (
    "/home/cnets-vision/mengti_ws/robot_filter/dataset/old_dataset/tagslam_poses/"
)
OUTPUT_POSE_DIR = "/home/cnets-vision/mengti_ws/poses_bundletrack2.0_latest/"
ODOM_FILE_PATH = (
    "/home/cnets-vision/mengti_ws/BundleTrack2.0/Data/old_dataset/annotated_poses/"
)


def get_cosine_sim(frame_id):
    """Compare the output of BundleTrack with the ground-truth poses of tagslam."""
    gt_pose = np.loadtxt(GT_POSE_DIR + "%04i.txt" % frame_id)
    output_pose = np.loadtxt(OUTPUT_POSE_DIR + "%04i.txt" % frame_id)
    return 1 - sp.distance.cdist(gt_pose, output_pose, "cosine")


def get_angle(P, Q):
    R = np.dot(P, Q.T)
    theta = (np.trace(R) - 1) / 2
    return np.arccos(theta) * (180 / np.pi)

def calculate_translation_error(estimated_pose, ground_truth_pose):
    est_translation = estimated_pose[:3, 3]
    gt_translation = ground_truth_pose[:3, 3]
    translation_error = np.linalg.norm(est_translation - gt_translation)
    return translation_error

def calculate_rotation_error(estimated_pose, ground_truth_pose):
    est_rotation = estimated_pose[:3, :3]
    gt_rotation = ground_truth_pose[:3, :3]
    rotation_error = np.arccos((np.trace(np.dot(est_rotation.T, gt_rotation)) - 1) / 2)
    return rotation_error

def calculate_success_rate(translation_errors, rotation_errors, translation_threshold, rotation_threshold):
    num_frames = len(translation_errors)
    success_count = sum(te <= translation_threshold and re <= rotation_threshold for te, re in zip(translation_errors, rotation_errors))
    success_rate = (success_count / num_frames) * 100
    return success_rate


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
    output_x, output_y, output_z = [], [], []  # translation
    gt_x, gt_y, gt_z = [], [], []
    output_alpha, output_beta, output_gamma = [], [], []
    gt_alpha, gt_beta, gt_gamma = [], [], []
    frame_num = len([name for name in os.listdir(bundletrack_pose_dir)])
    estimated_poses, ground_truth_poses = [], []
    for frame_id in range(1, frame_num + 1):
        output_pose = np.loadtxt(bundletrack_pose_dir + "%04i.txt" % frame_id)
        output_pose = transform_bundletrack_output_to_world(
            output_pose,
            CAMERA_CONFIG["old"]["translation"],
            CAMERA_CONFIG["old"]["axis_vec"],
            bundletrack_pose_dir,
            ODOM_FILE_PATH,
        )
        estimated_poses.append(output_pose)
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
        ground_truth_poses.append(gt_pose)
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

    # Evaluate based on 5deg5cm metric
    translation_errors = [calculate_translation_error(est_pose, gt_pose) for est_pose, gt_pose in zip(estimated_poses, ground_truth_poses)]
    rotation_errors = [calculate_rotation_error(est_pose, gt_pose) for est_pose, gt_pose in zip(estimated_poses, ground_truth_poses)]

    # Set error thresholds for successful pose estimation
    translation_threshold = 0.05
    rotation_threshold = 5.0
    
    # Calculate success rate
    success_rate = calculate_success_rate(translation_errors, rotation_errors, translation_threshold, np.radians(rotation_threshold))
    print(f"Translation Error: {np.mean(translation_errors):.4f}")
    print(f"Rotation Error: {np.degrees(np.mean(rotation_errors)):.4f} degrees")
    print(f"Success Rate: {success_rate:.2f}%")

    fig, axs = plt.subplots(2, 3)
    # fig.suptitle("Toss %i" % TOSS_IDX)
    axs[0, 0].plot(bundletrack_time, output_x, label="BundleTrack")
    axs[0, 0].plot(gt_time, gt_x, label="ground-truth")
    axs[0, 0].set_title("Position x")
    axs[0, 0].legend()

    axs[0, 1].plot(bundletrack_time, output_y, label="BundleTrack")
    axs[0, 1].plot(gt_time, gt_y, label="ground-truth")
    axs[0, 1].set_title("Position y")
    axs[0, 1].legend()

    axs[0, 2].plot(bundletrack_time, output_z, label="BundleTrack")
    axs[0, 2].plot(gt_time, gt_z, label="ground-truth")
    axs[0, 2].set_title("Position z")
    axs[0, 2].legend()

    axs[1, 0].plot(bundletrack_time, output_alpha, label="BundleTrack")
    axs[1, 0].plot(gt_time, gt_alpha, label="ground-truth")
    axs[1, 0].set_title("Roll")

    axs[1, 1].plot(bundletrack_time, output_beta, label="BundleTrack")
    axs[1, 1].plot(gt_time, gt_beta, label="ground-truth")
    axs[1, 1].set_title("Pitch")

    axs[1, 2].plot(bundletrack_time, output_gamma, label="BundleTrack")
    axs[1, 2].plot(gt_time, gt_gamma, label="ground-truth")
    axs[1, 2].set_title("Yaw")

    spacing = 0.100
    fig.subplots_adjust(bottom=spacing)
    plt.show()
    plt.savefig('result_bundletrack2.0_latest.png')


if __name__ == "__main__":
    # plot_xyz(1, 950)
    depth_bag_file = "./raw_10.bag"
    odom_bag_file = "./odom_10.bag"
    DEPTH_ROS_TOPIC = "/camera/aligned_depth_to_color/image_raw"
    ODOM_ROS_TOPIC = "/tagslam/odom/body_cube"
    # start_time = rospy.rostime.Time(secs=1667330345, nsecs=22710468)
    # end_time = rospy.rostime.Time(secs=1667330357, nsecs=228116)
    start_time = None
    end_time = None
    ##### NEW DATA ######
    # start time
    # start_time = rospy.rostime.Time(secs=1667330342, nsecs=959312)  # toss 1
    # start_time = rospy.rostime.Time(secs=1667330357, nsecs=453744)  # toss 2
    # start_time = rospy.rostime.Time(secs=1667330379, nsecs=657584)  # toss 3
    # start_time = rospy.rostime.Time(secs=1667330401, nsecs=359995)  # toss 4
    # start_time = rospy.rostime.Time(secs=1667330422, nsecs=301875)  # toss 5
    # start_time = rospy.rostime.Time(secs=1667330442, nsecs=284764)  # toss 6
    # start_time = rospy.rostime.Time(secs=1667330465, nsecs=216068)  # toss 7
    # start_time = rospy.rostime.Time(secs=1667330490, nsecs=318756)  # toss 8

    # end time
    # end_time = rospy.rostime.Time(secs=1667330357, nsecs=453744)  # toss 1
    # end_time = rospy.rostime.Time(secs=1667330379, nsecs=657584)  # toss 2
    # end_time = rospy.rostime.Time(secs=1667330401, nsecs=359995)  # toss 3
    # end_time = rospy.rostime.Time(secs=1667330422, nsecs=301875)  # toss 4
    # end_time = rospy.rostime.Time(secs=1667330442, nsecs=284764)  # toss 5
    # end_time = rospy.rostime.Time(secs=1667330465, nsecs=216068)  # toss 6
    # end_time = rospy.rostime.Time(secs=1667330490, nsecs=318756)  # toss 7
    # end_time = rospy.rostime.Time(secs=1667330509, nsecs=187270)  # toss 8

    #### OLD DATA #####
    # start time
    start_time = rospy.rostime.Time(secs=1655404893, nsecs=899137)  # toss 1
    # start_time = rospy.rostime.Time(secs=1655404906, nsecs=156495)  # toss 2
    # start_time = rospy.rostime.Time(secs=1655404918, nsecs=435741)  # toss 3
    # start_time = rospy.rostime.Time(secs=1655404930, nsecs=306306)  # toss 4
    # start_time = rospy.rostime.Time(secs=1655404944, nsecs=447914)  # toss 5
    # start_time=rospy.rostime.Time(secs=1655404955, nsecs=272877)  # toss 6
    # start_time=rospy.rostime.Time(secs=1655404966, nsecs=698458)  # toss 7
    # start_time=rospy.rostime.Time(secs=1655404977, nsecs=941630)  # toss 8
    # start_time=rospy.rostime.Time(secs=1655404991, nsecs=641514) # toss 9
    # start_time=rospy.rostime.Time(secs=1655405008, nsecs=720921) # toss 10

    # end time
    end_time = rospy.rostime.Time(secs=1655404908, nsecs=279948)
    # end_time = rospy.rostime.Time(secs=1655404906, nsecs=156495)  # toss 1
    # end_time = rospy.rostime.Time(secs=1655404918, nsecs=435741)  # toss 2
    # end_time = rospy.rostime.Time(secs=1655404930, nsecs=306306)  # toss 3
    # end_time = rospy.rostime.Time(secs=1655404944, nsecs=447914)  # toss 4
    # end_time = rospy.rostime.Time(secs=1655404955, nsecs=272877)  # toss 5
    # end_time=rospy.rostime.Time(secs=1655404966, nsecs=698458)  # toss 6
    # end_time=rospy.rostime.Time(secs=1655404977, nsecs=941630)  # toss 7
    # end_time=rospy.rostime.Time(secs=1655404991, nsecs=641514)  # toss 8
    # end_time=rospy.rostime.Time(secs=1655405008, nsecs=720921)  # toss 9
    # end_time=rospy.rostime.Time(secs=1655405022, nsecs=942549)  # toss 10
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
