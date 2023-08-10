import os
import rospy
import numpy as np
import scipy.spatial as sp
import matplotlib.pyplot as plt

from math_utils import (
    camera_to_world,
    rotation_matrix_to_euler,
    transform_bundletrack_output,
)
from rosbag_processor import extract_time_versus_poses, extract_gt_poses_from_tagslam_with_missing_frames

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
    "/home/cnets-vision/mengti_ws/BundleTrack/Data/YCBINEOAT/contact_nets_new_first_toss/tagslam_poses/"
)
OUTPUT_POSE_DIR = "/home/cnets-vision/mengti_ws/poses_new_normalized/"
ODOM_FILE_PATH = (
    "/home/cnets-vision/mengti_ws/BundleTrack/Data/YCBINEOAT/contact_nets_new_first_toss/annotated_poses/"
)
FIG_NAME = "result_poses_new_normalized.png"


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
    # print("est_translation: ", est_translation, "gt_translation: ", gt_translation)
    return translation_error

def calculate_rotation_error(estimated_pose, ground_truth_pose):
    est_rotation = estimated_pose[:3, :3]
    gt_rotation = ground_truth_pose[:3, :3]
    # print("est_rotation: ", est_rotation, "gt_rotation: ", gt_rotation)
    cos_similarity = (np.trace(np.dot(est_rotation.T, gt_rotation)) - 1) / 2
    cos_similarity = np.clip(cos_similarity, -1, 1)
    rotation_error = np.arccos(cos_similarity)
    return rotation_error

def calculate_success_rate(translation_errors, rotation_errors, translation_threshold, rotation_threshold):
    num_frames = len(translation_errors)
    success_count = sum(te <= translation_threshold and re <= rotation_threshold for te, re in zip(translation_errors, rotation_errors))
    success_rate = (success_count / num_frames) * 100
    return success_rate

def matrix_to_quaternion(matrix):
    q = np.empty((4,))
    trace = matrix[0, 0] + matrix[1, 1] + matrix[2, 2]

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        q[0] = 0.25 / s
        q[1] = (matrix[2, 1] - matrix[1, 2]) * s
        q[2] = (matrix[0, 2] - matrix[2, 0]) * s
        q[3] = (matrix[1, 0] - matrix[0, 1]) * s
    else:
        if matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
            s = 2.0 * np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])
            q[0] = (matrix[2, 1] - matrix[1, 2]) / s
            q[1] = 0.25 * s
            q[2] = (matrix[0, 1] + matrix[1, 0]) / s
            q[3] = (matrix[0, 2] + matrix[2, 0]) / s
        elif matrix[1, 1] > matrix[2, 2]:
            s = 2.0 * np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2])
            q[0] = (matrix[0, 2] - matrix[2, 0]) / s
            q[1] = (matrix[0, 1] + matrix[1, 0]) / s
            q[2] = 0.25 * s
            q[3] = (matrix[1, 2] + matrix[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1])
            q[0] = (matrix[1, 0] - matrix[0, 1]) / s
            q[1] = (matrix[0, 2] + matrix[2, 0]) / s
            q[2] = (matrix[1, 2] + matrix[2, 1]) / s
            q[3] = 0.25 * s
    return q

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
    output_x, output_y, output_z = [], [], []  # translation
    gt_x, gt_y, gt_z = [], [], []
    estimated_w, estimated_x, estimated_y, estimated_z = [], [], [], []
    ground_truth_w, ground_truth_x, ground_truth_y, ground_truth_z = [], [], [], []
    frame_num = len([name for name in os.listdir(bundletrack_pose_dir)])
    estimated_poses, ground_truth_poses = [], []
    # bundletrack_time = np.arange(frame_num)
    # bundletrack_time = np.array(bundletrack_time)
    problem_frames = set()
    for frame_id in range(1, len(gt_time) + 1):
        gt_frame = frame_id
        gt_pose = np.loadtxt(gt_pose_dir + "%04i.txt" % gt_frame)
        # gt pose is identity if no odometry data
        if np.array_equal(gt_pose, np.identity(gt_pose.shape[0])):
            problem_frames.add(frame_id)
            continue
        ground_truth_poses.append(gt_pose)
        gt_x.append(gt_pose[0, 3])
        gt_y.append(gt_pose[1, 3])
        gt_z.append(gt_pose[2, 3])
        w_, x_, y_, z_ = matrix_to_quaternion(gt_pose)
        ground_truth_w.append(w_) 
        ground_truth_x.append(x_)
        ground_truth_y.append(y_)
        ground_truth_z.append(z_)
        
    for frame_id in range(1, frame_num + 1):
        if frame_id in problem_frames:
            continue
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

        w, x, y, z = matrix_to_quaternion(output_pose)
        estimated_w.append(w)
        estimated_x.append(x)
        estimated_y.append(y)
        estimated_z.append(z)
    

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

    fig, axs = plt.subplots(2, 4)
    # bundletrack_time = np.arange(len(output_x))
    # gt_time = np.arange(len(gt_x))
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
    plt.show()
    plt.savefig(FIG_NAME)


if __name__ == "__main__":
    # plot_xyz(1, 950)
    depth_bag_file = "./raw_19.bag"
    odom_bag_file = "./odom_19.bag"
    DEPTH_ROS_TOPIC = "/camera/aligned_depth_to_color/image_raw"
    ODOM_ROS_TOPIC = "/tagslam/odom/body_box"
    # start_time = rospy.rostime.Time(secs=1667330345, nsecs=22710468)
    # end_time = rospy.rostime.Time(secs=1667330357, nsecs=228116)
    start_time = None
    end_time = None
    ##### NEW DATA ######
    # start time
    start_time = rospy.rostime.Time(secs=1667330342, nsecs=959312)  # toss 1
    # start_time = rospy.rostime.Time(secs=1667330357, nsecs=453744)  # toss 2
    # start_time = rospy.rostime.Time(secs=1667330379, nsecs=657584)  # toss 3
    # start_time = rospy.rostime.Time(secs=1667330401, nsecs=359995)  # toss 4
    # start_time = rospy.rostime.Time(secs=1667330422, nsecs=301875)  # toss 5
    # start_time = rospy.rostime.Time(secs=1667330442, nsecs=284764)  # toss 6
    # start_time = rospy.rostime.Time(secs=1667330465, nsecs=216068)  # toss 7
    # start_time = rospy.rostime.Time(secs=1667330490, nsecs=318756)  # toss 8

    # end time
    end_time = rospy.rostime.Time(secs=1667330357, nsecs=453744)  # toss 1
    # end_time = rospy.rostime.Time(secs=1667330379, nsecs=657584)  # toss 2
    # end_time = rospy.rostime.Time(secs=1667330401, nsecs=359995)  # toss 3
    # end_time = rospy.rostime.Time(secs=1667330422, nsecs=301875)  # toss 4
    # end_time = rospy.rostime.Time(secs=1667330442, nsecs=284764)  # toss 5
    # end_time = rospy.rostime.Time(secs=1667330465, nsecs=216068)  # toss 6
    # end_time = rospy.rostime.Time(secs=1667330490, nsecs=318756)  # toss 7
    # end_time = rospy.rostime.Time(secs=1667330509, nsecs=187270)  # toss 8

    bundletrack_time, gt_time = extract_time_versus_poses(
        start_time,
        end_time,
        depth_bag_file,
        odom_bag_file,
        DEPTH_ROS_TOPIC,
        ODOM_ROS_TOPIC,
        GT_POSE_DIR,
    )
    # gt_time = extract_gt_poses_from_tagslam_with_missing_frames(
    #     start_time,
    #     end_time,
    #     depth_bag_file=depth_bag_file,
    #     odom_bag_file=odom_bag_file,
    #     depth_topic=DEPTH_ROS_TOPIC,
    #     odom_topic=ODOM_ROS_TOPIC,
    #     output_dir=GT_POSE_DIR,
    # )
    
    print(len(bundletrack_time), len(gt_time))
    plot_with_time(bundletrack_time, gt_time, OUTPUT_POSE_DIR, GT_POSE_DIR)
