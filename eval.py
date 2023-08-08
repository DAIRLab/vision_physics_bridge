import os
from example import CAMERA_CONFIG
import rospy
import numpy as np
import scipy.spatial as sp
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

from math_utils import (
    camera_to_world,
    pos_quat_to_trans_mat,
    trans_mat_to_pos_quat,
    world_to_camera,
    rotation_matrix_to_euler,
    transform_bundletrack_output,
)
from rosbag_processor import (
    extract_time_versus_poses,
    extract_gt_poses_from_tagslam_with_missing_frames,
)
from sync_data import Synchronizer
from data_preparation import DatasetManagement

GT_POSE_DIR = (
    "/home/cnets-vision/mengti_ws/robot_filter/dataset/old_toss_5/tagslam_poses/"
)
OUTPUT_POSE_DIR = "/home/cnets-vision/mengti_ws/BundleSDF/results/old_toss_5/ob_in_cam/"
ODOM_FILE_PATH = (
    "/home/cnets-vision/mengti_ws/BundleSDF/data/old_toss_5/annotated_poses/"
)
FIG_NAME = "result_poses_bundlesdf_test_transfer.png"


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
    cos_similarity = (np.trace(np.dot(est_rotation.T, gt_rotation)) - 1) / 2
    cos_similarity = np.clip(cos_similarity, -1, 1)
    rotation_error = np.arccos(cos_similarity)
    return rotation_error


def calculate_success_rate(
    translation_errors, rotation_errors, translation_threshold, rotation_threshold
):
    num_frames = len(translation_errors)
    success_count = sum(
        te <= translation_threshold and re <= rotation_threshold
        for te, re in zip(translation_errors, rotation_errors)
    )
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
    estimated_w, estimated_x, estimated_y, estimated_z = [], [], [], []
    ground_truth_w, ground_truth_x, ground_truth_y, ground_truth_z = [], [], [], []
    frame_num = len(bundletrack_time)
    estimated_poses, ground_truth_poses = [], []
    for frame_id in range(1, frame_num + 1):
        output_pose = np.loadtxt(bundletrack_pose_dir + "%04i.txt" % frame_id)
        output_pose = transform_bundletrack_output(
            output_pose,
            bundletrack_pose_dir,
            ODOM_FILE_PATH,
        )
        estimated_poses.append(output_pose)
        output_x.append(output_pose[0, 3])
        output_y.append(output_pose[1, 3])
        output_z.append(output_pose[2, 3])

        x, y, z, w = R.from_matrix(output_pose[:3, :3]).as_quat()
        estimated_w.append(w)
        estimated_x.append(x)
        estimated_y.append(y)
        estimated_z.append(z)

    for frame_id in range(1, len(gt_time) + 1):
        gt_frame = frame_id
        gt_pose = np.loadtxt(gt_pose_dir + "%04i.txt" % gt_frame)
        gt_pose_trans = pos_quat_to_trans_mat(gt_pose)
        gt_pose_trans_cam = world_to_camera(
            gt_pose_trans,
            CAMERA_CONFIG["old"]["translation"],
            CAMERA_CONFIG["old"]["axis_vec"],
        )
        gt_pos_quat_cam = trans_mat_to_pos_quat(gt_pose_trans_cam)
        gt_x.append(gt_pos_quat_cam[0])
        gt_y.append(gt_pos_quat_cam[1])
        gt_z.append(gt_pos_quat_cam[2])
        x_, y_, z_, w_ = (
            gt_pos_quat_cam[3],
            gt_pos_quat_cam[4],
            gt_pos_quat_cam[5],
            gt_pos_quat_cam[6],
        )
        ground_truth_poses.append(gt_pose_trans_cam)
        ground_truth_w.append(w_)
        ground_truth_x.append(x_)
        ground_truth_y.append(y_)
        ground_truth_z.append(z_)
        
        # x_, y_, z_, w_ = gt_pose_[3:]
        # print(gt_pose_[:3].shape)
        # ground_truth_pose = np.vstack(
        # (np.hstack((R.from_quat(gt_pose_[3:]).as_matrix(), gt_pose_[:3].reshape(-1, 1))), np.array([0, 0, 0, 1])))
        # ground_truth_poses.append(ground_truth_pose)
        # ground_truth_w.append(w_)
        # ground_truth_x.append(x_)
        # ground_truth_y.append(y_)
        # ground_truth_z.append(z_)
        # gt_x.append(gt_pose_[0])
        # gt_y.append(gt_pose_[1])
        # gt_z.append(gt_pose_[2])

    output_x, output_y, output_z = (
        np.array(output_x),
        np.array(output_y),
        np.array(output_z),
    )
    gt_x, gt_y, gt_z = np.array(gt_x), np.array(gt_y), np.array(gt_z)

    # Evaluate based on 5deg5cm metric
    translation_errors = [
        calculate_translation_error(est_pose, gt_pose)
        for est_pose, gt_pose in zip(estimated_poses, ground_truth_poses)
    ]
    rotation_errors = [
        calculate_rotation_error(est_pose, gt_pose)
        for est_pose, gt_pose in zip(estimated_poses, ground_truth_poses)
    ]

    # Set error thresholds for successful pose estimation
    translation_threshold = 0.05
    rotation_threshold = 5.0

    # Calculate success rate
    success_rate = calculate_success_rate(
        translation_errors,
        rotation_errors,
        translation_threshold,
        np.radians(rotation_threshold),
    )
    print(f"Translation Error: {np.mean(translation_errors):.4f}")
    print(f"Rotation Error: {np.degrees(np.mean(rotation_errors)):.4f} degrees")
    print(f"Success Rate: {success_rate:.2f}%")

    fig, axs = plt.subplots(2, 4)
    # fig.suptitle("Toss %i" % TOSS_IDX)
    axs[0, 0].plot(bundletrack_time, output_x, label="BundleTrack")
    axs[0, 0].plot(gt_time, gt_x, label="ground-truth")
    axs[0, 0].set_title("Position x")
    axs[0, 0].legend()

    axs[0, 1].plot(bundletrack_time, output_y, label="BundleTrack")
    axs[0, 1].plot(gt_time, gt_y, label="ground-truth")
    axs[0, 1].set_title("Position y")

    axs[0, 2].plot(bundletrack_time, output_z, label="BundleTrack")
    axs[0, 2].plot(gt_time, gt_z, label="ground-truth")
    axs[0, 2].set_title("Position z")

    axs[1, 0].plot(bundletrack_time, estimated_w, label="BundleTrack")
    axs[1, 0].plot(gt_time, ground_truth_w, label="ground-truth")
    axs[1, 0].set_title("w")

    axs[1, 1].plot(bundletrack_time, estimated_x, label="BundleTrack")
    axs[1, 1].plot(gt_time, ground_truth_x, label="ground-truth")
    axs[1, 1].set_title("x")

    axs[1, 2].plot(bundletrack_time, estimated_y, label="BundleTrack")
    axs[1, 2].plot(gt_time, ground_truth_y, label="ground-truth")
    axs[1, 2].set_title("y")

    axs[1, 3].plot(bundletrack_time, estimated_z, label="BundleTrack")
    axs[1, 3].plot(gt_time, ground_truth_z, label="ground-truth")
    axs[1, 3].set_title("z")

    spacing = 0.100
    fig.subplots_adjust(hspace=0.5)
    fig.subplots_adjust(bottom=spacing)
    plt.savefig(FIG_NAME)
    plt.show()


if __name__ == "__main__":
    # plot_xyz(1, 950)
    depth_bag_file = "./raw_10.bag"
    odom_bag_file = "./odom_10.bag"
    DEPTH_ROS_TOPIC = "/camera/aligned_depth_to_color/image_raw"
    ODOM_ROS_TOPIC = "/tagslam/odom/body_cube"
    start_time = None
    end_time = None
    #### OLD DATA #####
    # start time
    # start_time = rospy.rostime.Time(secs=1655404893, nsecs=899137)  # toss 1
    # start_time = rospy.rostime.Time(secs=1655404908, nsecs=279948)  # toss 2
    # start_time = rospy.rostime.Time(secs=1655404920, nsecs=470680)  # toss 3
    # start_time = rospy.rostime.Time(secs=1655404932, nsecs=647236)  # toss 4
    start_time = rospy.rostime.Time(secs=1655404945, nsecs=387903)  # toss 5
    # start_time=rospy.rostime.Time(secs=1655404955, nsecs=272877)  # toss 6
    # start_time=rospy.rostime.Time(secs=1655404966, nsecs=698458)  # toss 7
    # start_time=rospy.rostime.Time(secs=1655404977, nsecs=941630)  # toss 8
    # start_time=rospy.rostime.Time(secs=1655404991, nsecs=641514) # toss 9
    # start_time=rospy.rostime.Time(secs=1655405008, nsecs=720921) # toss 10

    # end time
    # end_time = rospy.rostime.Time(secs=1655404908, nsecs=279948) # toss 1
    # end_time = rospy.rostime.Time(secs=1655404920, nsecs=470680)  # toss 2
    # end_time = rospy.rostime.Time(secs=1655404932, nsecs=647236)  # toss 3
    # end_time = rospy.rostime.Time(secs=1655404945, nsecs=387903)  # toss 4
    end_time = rospy.rostime.Time(secs=1655404955, nsecs=463919)  # toss 5
    # end_time=rospy.rostime.Time(secs=1655404966, nsecs=698458)  # toss 6
    # end_time=rospy.rostime.Time(secs=1655404977, nsecs=941630)  # toss 7
    # end_time=rospy.rostime.Time(secs=1655404991, nsecs=641514)  # toss 8
    # end_time=rospy.rostime.Time(secs=1655405008, nsecs=720921)  # toss 9
    # end_time=rospy.rostime.Time(secs=1655405022, nsecs=942549)  # toss 10
    ######################### DEPRECATED ###########################
    # bundletrack_time, gt_time = extract_time_versus_poses(
    #     start_time,
    #     end_time,
    #     depth_bag_file,
    #     odom_bag_file,
    #     DEPTH_ROS_TOPIC,
    #     ODOM_ROS_TOPIC,
    #     GT_POSE_DIR,
    # )

    # gt_time_ = extract_gt_poses_from_tagslam_with_missing_frames(
    #     start_time,
    #     end_time,
    #     depth_bag_file=depth_bag_file,
    #     odom_bag_file=odom_bag_file,
    #     depth_topic=DEPTH_ROS_TOPIC,
    #     odom_topic=ODOM_ROS_TOPIC,
    #     output_dir=GT_POSE_DIR,
    #     write=False,
    # )
    ################################################################
    frame_num = len([name for name in os.listdir(OUTPUT_POSE_DIR)])
    print(f"there are {frame_num} frames")
    sync = Synchronizer(GT_POSE_DIR, frame_num, start_time, end_time)
    bundletrack_time, gt_time = sync.bundletrack_time, sync.gt_time
    print(len(bundletrack_time), len(gt_time))
    bundletrack_time = [t.to_sec() for t in bundletrack_time]
    gt_time = [t.to_sec() for t in gt_time]
    plot_with_time(bundletrack_time, gt_time, OUTPUT_POSE_DIR, GT_POSE_DIR)

    # ContactNets toss: from object leaving the gripper to the object landing on the table
    # start_time_toss = rospy.rostime.Time(secs=1655404905, nsecs=177325)  # toss 1
    # end_time_toss = rospy.rostime.Time(secs=1655404906, nsecs=183975)  # toss 1
    # Convert pose data to ContactNets format
    # dataset = DatasetManagement(frame_num, start_time_toss, end_time_toss, bundletrack_time)