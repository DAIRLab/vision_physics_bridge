import argparse
import os
from file_utils import load_dataset_from_yaml, load_toss_time_from_yaml
from rosbag_processor import extract_time_versus_poses
import rospy
import numpy as np
import scipy.spatial as sp
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import cv2

from math_utils import pos_quat_to_trans_mat, trans_mat_to_pos_quat, transform_bundletrack_output, world_to_camera

from sync_data import Synchronizer
import yaml

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

def plot_with_time(bundletrack_time, gt_time, bundletrack_pose_dir, gt_poses):
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
            cam_trans,
            cam_axis_vec,
            to_world=True,
        ) # camera frame
        estimated_poses.append(output_pose)
        output_pose_ = trans_mat_to_pos_quat(output_pose)
        output_x.append(output_pose_[0])
        output_y.append(output_pose_[1])
        output_z.append(output_pose_[2])
        x, y, z, w = (
            output_pose_[3],
            output_pose_[4],
            output_pose_[5],
            output_pose_[6],
        )
        estimated_w.append(w)
        estimated_x.append(x)
        estimated_y.append(y)
        estimated_z.append(z)

        ########## Transform to world again ############
        # output_pose_world_test = camera_to_world(output_pose)
        # estimated_x_world_pose.append(output_pose_world_test[0, 3])
        # estimated_y_world_pose.append(output_pose_world_test[1, 3])
        # estimated_z_world_pose.append(output_pose_world_test[2, 3])

        # x_test, y_test, z_test, w_test = R.from_matrix(output_pose_world_test[:3, :3]).as_quat()
        # estimate_x_world_quat_list.append(x_test)
        # estimate_y_world_quat_list.append(y_test)
        # estimate_z_world_quat_list.append(z_test)
        # estimate_w_world_quat_list.append(w_test)
        ################################################

    for frame_id in range(frame_num):
        gt_frame = frame_id
        gt_pose = gt_poses[gt_frame]
        gt_x.append(gt_pose[0])
        gt_y.append(gt_pose[1])
        gt_z.append(gt_pose[2])
        x_, y_, z_, w_ = (
            gt_pose[3],
            gt_pose[4],
            gt_pose[5],
            gt_pose[6],
        )
        ground_truth_poses.append(pos_quat_to_trans_mat(gt_pose))
        ground_truth_w.append(w_)
        ground_truth_x.append(x_)
        ground_truth_y.append(y_)
        ground_truth_z.append(z_)
        ############# Transform to camera ##############
        # gt_pose_trans = pos_quat_to_trans_mat(gt_pose)
        # gt_pose_trans_cam = world_to_camera(
        #     gt_pose_trans,
        #     cam_trans,
        #     cam_axis_vec,
        # )
        # gt_pos_quat_cam = trans_mat_to_pos_quat(gt_pose_trans_cam)
        # gt_x.append(gt_pos_quat_cam[0])
        # gt_y.append(gt_pos_quat_cam[1])
        # gt_z.append(gt_pos_quat_cam[2])
        # x_, y_, z_, w_ = (
        #     gt_pos_quat_cam[3],
        #     gt_pos_quat_cam[4],
        #     gt_pos_quat_cam[5],
        #     gt_pos_quat_cam[6],
        # )
        # ground_truth_poses.append(gt_pose_trans_cam)
        # ground_truth_w.append(w_)
        # ground_truth_x.append(x_)
        # ground_truth_y.append(y_)
        # ground_truth_z.append(z_)
        
        ########### Transform to world again ############
        # gt_pose_world_test = camera_to_world(gt_pose_trans_cam)
        # gt_pose_world_quat = trans_mat_to_pos_quat(gt_pose_world_test)
        # print("After world2cam and cam2world transform", gt_pose_world_quat)
        # gt_x_quat_test, gt_y_quat_test, gt_z_quat_test, gt_w_quat_test = (gt_pose_world_quat[3], 
        #                                    gt_pose_world_quat[4], 
        #                                    gt_pose_world_quat[5], 
        #                                    gt_pose_world_quat[6]
        #                                 )
        # gt_x_world_pose.append(gt_pose_world_quat[0]) 
        # gt_y_world_pose.append(gt_pose_world_quat[1])
        # gt_z_world_pose.append(gt_pose_world_quat[2])
        # gt_x_world_test_list.append(gt_x_quat_test)
        # gt_y_world_test_list.append(gt_y_quat_test)
        # gt_z_world_test_list.append(gt_z_quat_test)
        # gt_w_world_test_list.append(gt_w_quat_test)
        ################################################
    output_x, output_y, output_z = (
        np.array(output_x),
        np.array(output_y),
        np.array(output_z),
    )
    gt_x, gt_y, gt_z = np.array(gt_x), np.array(gt_y), np.array(gt_z)

    # Evaluate based on 5deg5cm metric
    translation_errors = [
        calculate_translation_error(estimated_pose, grount_truth_pose)
        for estimated_pose, grount_truth_pose in zip(estimated_poses, ground_truth_poses)
    ]
    rotation_errors = [
        calculate_rotation_error(estimated_pose, grount_truth_pose)
        for estimated_pose, grount_truth_pose in zip(estimated_poses, ground_truth_poses)
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
    print(f'Figure saved to {FIG_NAME}')
    plt.show()

def draw_coords_to_image():
    """Draw rgb axes onto images to indicate the pose.
    """
    SEGMENT_IMAGE_DIR = f"/home/cnets-vision/mengti_ws/BundleSDF/results/cube_{toss_id}/color_segmented/"
    OUTPUT_SEGMENT_DIR = f"./dataset/cube_{toss_id}/color_segmented_coords/"
    if not os.path.exists(OUTPUT_SEGMENT_DIR):
        os.mkdir(OUTPUT_SEGMENT_DIR)
        print(f'Made dir {OUTPUT_SEGMENT_DIR}')
    axis_length = 0.1
    axis_points = np.float32([[0,0,0],
                            [axis_length,0,0], 
                            [0,axis_length,0], 
                            [0,0,axis_length]])
    intrinsic = np.loadtxt(f"/home/cnets-vision/mengti_ws/BundleSDF/data/cube_{toss_id}/cam_K.txt")
    # tagslam_data = np.loadtxt(GT_POSE_DIR+'tagslam.txt')
    for frame_id in range(1, frame_num):
        output_pose = np.loadtxt(OUTPUT_POSE_DIR + "%04i.txt" % frame_id)
        output_pose = transform_bundletrack_output(
            output_pose,
            OUTPUT_POSE_DIR,
            ODOM_FILE_PATH,
            cam_trans,
            cam_axis_vec,
        ) # camera frame
        rvec = output_pose[:3, :3]
        tvec = output_pose[:3, 3]
        # tagslam_pose = tagslam_data[frame_id, 2:]
        # tagslam_mat = pos_quat_to_trans_mat(tagslam_pose)
        # tagslam_mat = world_to_camera(tagslam_mat, cam_trans, cam_axis_vec)
        # rvec = tagslam_mat[:3, :3]
        # tvec = tagslam_mat[:3, 3]
        image_points, _ = cv2.projectPoints(axis_points, rvec, tvec, intrinsic, distCoeffs=np.zeros((5,1)))

        f = os.path.join(SEGMENT_IMAGE_DIR, "%04i.png" % frame_id)
        filename = "%04i.png" % frame_id
        print(f'file path: {f}')
        image_with_axes = cv2.imread(f)
        image_points = image_points.reshape(image_points.shape[0], image_points.shape[-1])
        image_points = image_points.astype(int)
        print(tuple(image_points[0].ravel()))
        print(tuple(image_points[1].ravel()))
        cv2.line(image_with_axes, tuple(image_points[0].ravel()), tuple(image_points[1].ravel()), (0,0,255), 5)  # X-axis (Red)
        cv2.line(image_with_axes, tuple(image_points[0].ravel()), tuple(image_points[2].ravel()), (0,255,0), 5)  # Y-axis (Green)
        cv2.line(image_with_axes, tuple(image_points[0].ravel()), tuple(image_points[3].ravel()), (255,0,0), 5)  # Z-axis (Blue)
        cv2.imwrite(OUTPUT_SEGMENT_DIR+filename, image_with_axes)
        print(f'Wrote image {OUTPUT_SEGMENT_DIR+filename}')

# TODO: This should get replaced in rosbag_processor.py or data_preparation.py
def save_init_pose():
    poses = np.loadtxt(os.path.join(GT_POSE_DIR, "tagslam.txt"))
    init_pose = poses[0, 1:]
    mat = pos_quat_to_trans_mat(init_pose)
    mat_cam = world_to_camera(mat, cam_trans, cam_axis_vec)
    np.savetxt(os.path.join(ODOM_FILE_PATH, "%04i.txt" % 0), mat_cam)
    print(f'Initial pose saved to {ODOM_FILE_PATH}')
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--toss_id",
        type=int,
        required=False,
    )
    args = parser.parse_args()
    toss_id = args.toss_id
    print(f'Processing toss {toss_id}')
    TOSS_TYPE = 'cube'
    yaml_path = './assets/config.yaml'
    bag_num = load_dataset_from_yaml(TOSS_TYPE, toss_id)
    print(f'bag num: {bag_num}')
    depth_bag_file = f"./rosbags/raw_{bag_num}.bag"
    odom_bag_file = f"./rosbags/odom_{bag_num}.bag"
    DEPTH_ROS_TOPIC = "/camera/aligned_depth_to_color/image_raw"
    ODOM_ROS_TOPIC = "/tagslam/odom/body_cube"
    DATASET=f"cube_{toss_id}"
    GT_POSE_DIR = f"/home/cnets-vision/mengti_ws/robot_filter/dataset/{DATASET}/tagslam_poses/"
    OUTPUT_POSE_DIR = f"/home/cnets-vision/mengti_ws/BundleSDF/results/{DATASET}/ob_in_cam/"
    #### Ablation 1 ####
    # OUTPUT_POSE_DIR = f"/home/cnets-vision/mengti_ws/robot_filter/ob_in_cams/ob_in_cam_exp_1/"
    
    #### Ablaton 2 ####
    # OUTPUT_POSE_DIR = f"/home/cnets-vision/mengti_ws/robot_filter/ob_in_cams/ob_in_cam_exp_2/"
    
    #### Ablation 3 ####
    # OUTPUT_POSE_DIR = f"/home/cnets-vision/mengti_ws/robot_filter/ob_in_cam_exp_1/"
    
    #### Ablation 4 ####
    # OUTPUT_POSE_DIR = f"/home/cnets-vision/mengti_ws/BundleSDF/results/old_toss_{toss_id}/ob_in_cam_exp_4/"
    
    #### Ablation 5 ####
    # OUTPUT_POSE_DIR = f"/home/cnets-vision/mengti_ws/robot_filter/ob_in_cams/ob_in_cam_exp_5/"
    
    #### Ablation 6 ####
    # OUTPUT_POSE_DIR = f"/home/cnets-vision/mengti_ws/robot_filter/ob_in_cams/ob_in_cam_exp_6/"
    
    ODOM_FILE_PATH = f"/home/cnets-vision/mengti_ws/BundleSDF/data/{DATASET}/annotated_poses/"
    FIG_NAME = f"result_poses_bundlesdf_{DATASET}.png"
    CAMERA_EXTRINSICS_FILE = "./assets/realsense_pose_cube.yaml"

    cam = 'cam0' # realsense camera name
    with open(CAMERA_EXTRINSICS_FILE, 'r') as stream:
        data_loaded = yaml.safe_load(stream)
    print(data_loaded[cam]['pose']['position'])
    cam_pos_dict = data_loaded[cam]['pose']['position']
    cam_trans = np.array([cam_pos_dict['x'], cam_pos_dict['y'], cam_pos_dict['z']]).reshape(-1, 1)
    cam_rot_dict = data_loaded[cam]['pose']['rotation']
    cam_axis_vec = np.array([cam_rot_dict['x'], cam_rot_dict['y'], cam_rot_dict['z']])
    
    yaml_path = './assets/config.yaml'
    toss_type = 'cube'
    frame_num = len([name for name in os.listdir(OUTPUT_POSE_DIR)])
    print(f"there are {frame_num} frames")
    start_time = load_toss_time_from_yaml(toss_type, toss_id, 'start_time')
    end_time = load_toss_time_from_yaml(toss_type, toss_id, 'end_time')
    bundletrack_time = extract_time_versus_poses(
        start_time,
        end_time,
        depth_bag_file,
        odom_bag_file,
        DEPTH_ROS_TOPIC,
        ODOM_ROS_TOPIC,
        GT_POSE_DIR,
        save=True
    ).reshape(-1,)
    save_init_pose()
    data = np.loadtxt(GT_POSE_DIR+'tagslam.txt')
    print('data', data.shape)
    gt_time = data[:, 0] #N,
    tagslam_poses = data[:, 1:] #N,7
    print(bundletrack_time.shape, gt_time.shape, tagslam_poses.shape)
    plot_with_time(bundletrack_time, gt_time, OUTPUT_POSE_DIR, tagslam_poses)
    # draw_coords_to_image()