"""This script is used to compute an effective offset to impose on the TagSLAM
poses in order to better align TagSLAM poses with depth readings.  While the
source of the error is likely on the depth camera side, we believe it is better
to do the accommodation on the TagSLAM side since more of the BundleSDF-PLL
pipeline uses the depth readings.  Keeping these at their native values seems
like a less invasive solution."""

import os
import os.path as op
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.cm as cmx
import matplotlib.pyplot as plt
import numpy as np
import pdb
import torch
from torch import Tensor
from typing import Tuple
import cv2
import rospy

import file_utils, math_utils, rosbag_processor

from compute_table_offsets import CUBE_CORNERS_IN_CUBE_FRAME, X_LIMS, Y_LIMS, \
    Z_LIMS


CUBE_EXTRA_BUFFER = 0.2
CUBE_CORNERS_PLOTTABLE_INDICES=[0, 1, 2, 3, 0, 4, 5, 6, 7, 4, 5, 1, 2, 6, 7, 3]

DEPTH_PLOT_HELP_PRINT = \
'''From past experience, the following offsets have looked reasonable for the
cube experiments (verified against cube_2, cube_3, and cube_9):
    update(-0.008, 0.001, -0.011)

Or, if want to test the different offset approaches, we found that a
distance of 12mm looks reasonable to back the camera readings up.  Can
try the following:

    dist = -0.012

    1) Offset the TagSLAM poses (translates TagSLAM poses):
    update(-dist*camera_z_normal[0], -dist*camera_z_normal[1],
           -dist*camera_z_normal[2])

    2) Offset the realsense camera pose (translates point cloud):
    camera_offset(dist)

    3) Offset the realsense depth readings (distorts point cloud):
    depth_offset(dist)

'''

DEPTH_PLOT_INSTRUCTIONS_PRINT = \
'''Use the following functions to help adjust the offset:
    update(x, y, z)       - Use a new x, y, z as the TagSLAM offset.
    depth_offset(offset)  - Apply a depth z offset to the depth image.
    camera_offset(offset) - Apply a camera z offset to the depth image.

Use these to adjust the plot view:
    top_view()            - View the plot from the top.
    side_view()           - View the plot from the side.
    original_view()       - View the plot from the original angle (this
                            looked right at the side of the cube_2 data).

Use this to save a TagSLAM offset:
    save_offset(x, y, z)  - Save the given offset to tagslam_offset.txt,
                            which gets used during dataset creation.
'''

# A (3, 2, 3) array where array[0] is the x-axis (from camera origin to a point
# along the position x-axis), array[1] is the y-axis, and array[2] is the
# z-axis.
AXIS_SCALING = 0.1
CAMERA_AXES_IN_CAMERA_FRAME = AXIS_SCALING * np.array([[[0, 0, 0], [1, 0, 0]],
                                                       [[0, 0, 0], [0, 1, 0]],
                                                       [[0, 0, 0], [0, 0, 1]]])

CAMERA_MARKER_COLORS = {'cam0': '#ff0000', 'cam1': '#00ff00', 'cam2': '#0000ff',
                        'realsense': '#ffff00'}

EE_RADIUS = 0.0195


def load_and_adjust_depth_readings_in_image(
        vision_asset: str, frame_num: int, z_axis_offset_meters: float
) -> np.ndarray:
    """Apply a depth reading offset to a depth image, which causes in a shrink
    or expansion of the resulting 3D point cloud."""
    object = '_'.join(vision_asset.split('_')[:-1])

    # Get the camera intrinsics and extrinsics.
    fx, fy, cx, cy = file_utils.load_camera_intrinsics(object)
    cam_trans, cam_rot_axis_angle = file_utils.load_camera_extrinsics(object)
    cam_trans = cam_trans.squeeze()
    cam_rot_axis_angle = cam_rot_axis_angle.squeeze()

    # Get the path of the desired depth image to load.
    depth_dir = file_utils.bundlesdf_video_depth_dir(vision_asset)
    depth_path = op.join(depth_dir, f"{frame_num:04d}.png")

    # Load the depth image.
    depth_image = math_utils.load_depth_image(depth_path)

    # Apply the z-offset, converting meters to millimeters.  Don't apply the
    # offset to non-returns (which have depth=0) so they still register as non-
    # returns.
    depth_image[depth_image != 0] += int(z_axis_offset_meters * 1000)

    return depth_image

def load_depth_image_as_points(vision_asset: str, frame_num: int,
                               z_axis_offset_meters: float = 0.0, smoothing: bool =False) -> np.ndarray:
    """Load a depth image from the BundlesDF dataset as a set of 3D points in
    world frame."""
    object = '_'.join(vision_asset.split('_')[:-1])

    # Get the camera intrinsics and extrinsics.
    fx, fy, cx, cy = file_utils.load_camera_intrinsics(object)
    cam_trans, cam_rot_axis_angle = file_utils.load_camera_extrinsics(object)

    # Load the depth image.
    depth_image = load_and_adjust_depth_readings_in_image(
        vision_asset, frame_num, z_axis_offset_meters)

    if smoothing:
        # filter the depth image using bilateral filter
        depth_image = cv2.bilateralFilter(depth_image.astype(np.float32), 20, 20, 10)   # diameter, sigmaColor, sigmaSpace

    # Convert the depth image to point cloud.
    points_camera = math_utils.convert_depth_image_to_points(
        depth_image, fx, fy, cx, cy)

    # Convert the points from camera frame to world frame.
    theta = np.linalg.norm(cam_rot_axis_angle)
    R_WC = math_utils.axis_angle_to_rotation_matrix(
        cam_rot_axis_angle.squeeze(), theta)
    points_world = (points_camera @ R_WC.T) + cam_trans.squeeze()

    return points_world

def crop_point_cloud(point_cloud: np.ndarray, cube_xyz: np.ndarray = None
                     ) -> Tuple[np.ndarray, np.ndarray]:
    if cube_xyz is None:
        mask = (point_cloud[:, 0] > X_LIMS[0]) & \
               (point_cloud[:, 0] < X_LIMS[1]) & \
               (point_cloud[:, 1] > Y_LIMS[0]) & \
               (point_cloud[:, 1] < Y_LIMS[1]) & \
               (point_cloud[:, 2] > Z_LIMS[0]) & \
               (point_cloud[:, 2] < Z_LIMS[1])
        return point_cloud[mask], mask

    mask = (point_cloud[:, 0] > cube_xyz[0] - CUBE_EXTRA_BUFFER) & \
           (point_cloud[:, 0] < cube_xyz[0] + CUBE_EXTRA_BUFFER) & \
           (point_cloud[:, 1] > cube_xyz[1] - CUBE_EXTRA_BUFFER) & \
           (point_cloud[:, 1] < cube_xyz[1] + CUBE_EXTRA_BUFFER) & \
           (point_cloud[:, 2] > cube_xyz[2] - CUBE_EXTRA_BUFFER) & \
           (point_cloud[:, 2] < cube_xyz[2] + CUBE_EXTRA_BUFFER)
    return point_cloud[mask], mask

def load_tagslam_pose(vision_asset: str, frame_num: int) -> np.ndarray:
    """Load a TagSLAM pose from the BundlesDF dataset."""
    tagslam_dir = file_utils.tagslam_pose_dir(vision_asset)
    tagslam_path = op.join(tagslam_dir, f"{frame_num:04d}.txt")
    return np.loadtxt(tagslam_path)

def load_poses_and_camera_images(vision_asset: str, frame_num: int):
    """Load the TagSLAM poses and camera images for a given frame."""
    object = '_'.join(vision_asset.split('_')[:-1])
    start_toss = int(vision_asset.split('_')[-1].split('-')[0])
    end_toss = start_toss if '-' not in vision_asset else \
        int(vision_asset.split('-')[1])

    start_time = file_utils.load_toss_time_from_yaml(
        object, start_toss, 'start_time', as_ros_time=True)
    end_time = file_utils.load_toss_time_from_yaml(
        object, end_toss, 'end_time', as_ros_time=True)

    rosbag_number = file_utils.load_rosbag_number_from_yaml(
        object, start_toss, second_toss_number=end_toss)
    raw_bag_file = file_utils.get_depth_bag_filename(rosbag_number)
    odom_bag_file = file_utils.get_odom_bag_filename(rosbag_number)

    odom_ros_topic = f"/tagslam/odom/body_{object}"

    pose_times, poses, image_times, image_msgs = \
        rosbag_processor.extract_camera_images_and_poses(
            start_time, end_time, raw_bag_file, odom_bag_file, odom_ros_topic
        )

    pose_t, pose = pose_times[frame_num], poses[frame_num]
    image_t = {key: image_times[key][frame_num] for key in image_times.keys()}
    image = {key: image_msgs[key][frame_num] for key in image_msgs.keys()}

    return pose_t, pose, image_t, image

def load_robot_poses_and_camera_images(vision_asset: str, frame_num: int):
    """Load the robot poses and camera images for a given frame."""
    assert vision_asset.startswith('robot')
    object = '_'.join(vision_asset.split('_')[:-1])
    start_toss = int(vision_asset.split('_')[-1].split('-')[0])
    end_toss = start_toss if '-' not in vision_asset else \
        int(vision_asset.split('-')[1])

    start_time = file_utils.load_toss_time_from_yaml(
        object, start_toss, 'start_time', as_ros_time=True)
    end_time = rospy.rostime.Time(secs=start_time.secs+1,
                                  nsecs=start_time.nsecs)

    rosbag_number = file_utils.load_rosbag_number_from_yaml(
        object, start_toss, second_toss_number=end_toss)
    robot_bag_file = file_utils.get_robot_bag_filename(rosbag_number)

    ee_times, ee_locations, image_times, image_msgs = \
        rosbag_processor.extract_camera_images_and_ee_locations(
            start_time, end_time, robot_bag_file)

    ee_t, ee_loc = ee_times[frame_num], ee_locations[frame_num]
    image_t = image_times[frame_num]
    image = image_msgs[frame_num]

    return ee_t, ee_loc, image_t, image

def inspect_tagslam_times(vision_asset: str, frame_num: int):
    """Make a plot of the message times to ensure they are synchronized."""
    pose_times, _poses, image_times, _image_msgs = \
        load_poses_and_camera_images(vision_asset, frame_num)

    t0 = pose_times[0]
    pose_times = np.array(pose_times) - t0
    cam0_times = np.array(image_times[0]) - t0
    cam1_times = np.array(image_times[1]) - t0
    cam2_times = np.array(image_times[2]) - t0

    plt.figure()
    plt.plot(pose_times, marker='o', markersize=10, label='TagSLAM Pose Times')
    plt.plot(cam0_times, marker='o', markersize=10, label='cam0 Times')
    plt.plot(cam1_times, marker='o', markersize=10, label='cam1 Times')
    plt.plot(cam2_times, marker='o', markersize=10, label='cam2 Times')
    plt.legend()
    plt.ylabel('Time (s)')
    plt.show()

def get_all_camera_intrinsics_extrinsics(vision_asset: str):
    """Get the camera intrinsics and extrinsics for all four cameras."""
    object = '_'.join(vision_asset.split('_')[:-1])
    start_toss = int(vision_asset.split('_')[-1].split('-')[0])
    end_toss = start_toss if '-' not in vision_asset else \
        int(vision_asset.split('-')[1])

    rosbag_number = file_utils.load_rosbag_number_from_yaml(
        object, start_toss, second_toss_number=end_toss)
    raw_bag_file = file_utils.get_depth_bag_filename(rosbag_number)
    odom_bag_file = file_utils.get_odom_bag_filename(rosbag_number)

    cam0_trans, cam0_axis_angle, cam1_trans, cam1_axis_angle, cam2_trans, \
        cam2_axis_angle = \
            rosbag_processor.get_tagslam_camera_extrinsics(odom_bag_file)

    cam_trans, cam_axis_angle = file_utils.load_camera_extrinsics(object)

    translations = {'cam0': cam0_trans, 'cam1': cam1_trans, 'cam2': cam2_trans,
                    'realsense': cam_trans.squeeze()}
    axis_angles = {'cam0': cam0_axis_angle, 'cam1': cam1_axis_angle,
                   'cam2': cam2_axis_angle,
                   'realsense': cam_axis_angle.squeeze()}

    intrinsics = rosbag_processor.get_all_camera_intrinsics(raw_bag_file)

    return intrinsics, translations, axis_angles

def get_robot_camera_intrinsics_extrinsics(vision_asset: str):
    """Get the realsense intrinsics and extrinsics."""
    assert vision_asset.startswith('robot')
    object = '_'.join(vision_asset.split('_')[:-1])
    start_toss = int(vision_asset.split('_')[-1].split('-')[0])
    end_toss = start_toss if '-' not in vision_asset else \
        int(vision_asset.split('-')[1])

    rosbag_number = file_utils.load_rosbag_number_from_yaml(
        object, start_toss, second_toss_number=end_toss)
    raw_bag_file = file_utils.get_robot_bag_filename(rosbag_number)

    cam_trans, cam_axis_angle = file_utils.load_camera_extrinsics(object)

    translations = {'realsense': cam_trans.squeeze()}
    axis_angles = {'realsense': cam_axis_angle.squeeze()}

    intrinsics = rosbag_processor.get_all_camera_intrinsics(raw_bag_file)
    intrinsics = {'realsense': intrinsics['realsense']}

    return intrinsics, translations, axis_angles

def inspect_camera_poses_and_images(vision_asset: str, frame_num: int = 1):
    """Plot the camera locations in 3D."""
    intrinsics, translations, axis_angles = \
        get_all_camera_intrinsics_extrinsics(vision_asset)

    world = np.array([0, 0, 0]).reshape(1, 3)
    cam0 = translations['cam0'].reshape(1, 3)
    cam1 = translations['cam1'].reshape(1, 3)
    cam2 = translations['cam2'].reshape(1, 3)
    realsense = translations['realsense'].reshape(1, 3)

    world_to_cam0 = np.concatenate((world, cam0), axis=0)
    world_to_cam1 = np.concatenate((world, cam1), axis=0)
    world_to_cam2 = np.concatenate((world, cam2), axis=0)
    world_to_realsense = np.concatenate((world, realsense), axis=0)

    pose_t, pose, image_t, image = load_poses_and_camera_images(
        vision_asset, frame_num)
    cube_corners_world = compute_cube_corners_in_world(pose)

    def plot_camera_triad(cam_trans, cam_axis_angle, label):
        pose = np.hstack((cam_trans,
                          math_utils.axis_angle_to_quat(cam_axis_angle)))
        cam_axes = math_utils.transform_point_coordinates_given_pose(
            CAMERA_AXES_IN_CAMERA_FRAME.reshape(6,3), pose).reshape(3,2,3)
        plt.plot(cam_axes[0, :, 0], cam_axes[0, :, 1], cam_axes[0, :, 2],
                 color='#ff0000', linewidth=5)
        plt.plot(cam_axes[1, :, 0], cam_axes[1, :, 1], cam_axes[1, :, 2],
                 color='#00ff00', linewidth=5)
        plt.plot(cam_axes[2, :, 0], cam_axes[2, :, 1], cam_axes[2, :, 2],
                 color='#0000ff', linewidth=5)
        plt.plot(cam_trans[0], cam_trans[1], cam_trans[2], marker='o',
                 color=CAMERA_MARKER_COLORS[label], markersize=10, label=label)

    # Be prepared to return all the figures.
    figs = {}

    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(world_to_cam0[:, 0], world_to_cam0[:, 1], world_to_cam0[:, 2])
    ax.plot(world_to_cam1[:, 0], world_to_cam1[:, 1], world_to_cam1[:, 2])
    ax.plot(world_to_cam2[:, 0], world_to_cam2[:, 1], world_to_cam2[:, 2])
    ax.plot(world_to_realsense[:, 0], world_to_realsense[:, 1],
            world_to_realsense[:, 2])
    for cam in ['cam0', 'cam1', 'cam2', 'realsense']:
        plot_camera_triad(translations[cam], axis_angles[cam], cam)

    ax.plot(cube_corners_world[CUBE_CORNERS_PLOTTABLE_INDICES, 0],
            cube_corners_world[CUBE_CORNERS_PLOTTABLE_INDICES, 1],
            cube_corners_world[CUBE_CORNERS_PLOTTABLE_INDICES, 2],
            c='r', label='TagSLAM Cube Edges')

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    plt.legend()

    ax.set_box_aspect([np.ptp(arr) for arr in \
                      [ax.get_xlim(), ax.get_ylim(), ax.get_zlim()]])
    figs['camera_poses'] = fig

    for cam in image.keys():
        # First compute the cube corners in camera frame.
        pose_cam = np.hstack((translations[cam],
                              math_utils.axis_angle_to_quat(axis_angles[cam])))
        cam_extrinsics = math_utils.pos_quat_to_trans_mat(pose_cam)
        cam_inv = np.linalg.inv(cam_extrinsics)
        pose_cam_inv = np.hstack((
            cam_inv[:3, 3],
            math_utils.rotation_matrix_to_quat(cam_inv[:3, :3])
        ))
        corners_in_cam = \
            math_utils.transform_point_coordinates_given_pose(
                cube_corners_world, pose_cam_inv)

        # Get the camera intrinsics.
        P_matrix = intrinsics[cam].reshape(3, 4)

        XYZ1 = np.hstack((corners_in_cam, np.ones((8, 1))))
        uvw = P_matrix @ XYZ1.T
        x_pixel = uvw[0] / uvw[2]
        y_pixel = uvw[1] / uvw[2]

        fig = plt.figure()
        if cam == 'realsense':
            plt.imshow(image[cam])
        else:
            plt.imshow(image[cam], cmap='gray', vmin=0, vmax=255)
        plt.scatter(x_pixel, y_pixel)
        plt.title(cam)
        figs[cam] = fig

    pdb.set_trace()
    return figs

def inspect_robot_pose_and_image(vision_asset: str, frame_num: int = 1):
    """Plot the camera locations in 3D."""
    intrinsics, translations, axis_angles = \
        get_robot_camera_intrinsics_extrinsics(vision_asset)

    world = np.array([0, 0, 0]).reshape(1, 3)
    realsense = translations['realsense'].reshape(1, 3)

    world_to_realsense = np.concatenate((world, realsense), axis=0)

    ee_t, ee_loc, image_t, image = load_robot_poses_and_camera_images(
        vision_asset, frame_num)

    def plot_camera_triad(cam_trans, cam_axis_angle, label):
        pose = np.hstack((cam_trans,
                          math_utils.axis_angle_to_quat(cam_axis_angle)))
        cam_axes = math_utils.transform_point_coordinates_given_pose(
            CAMERA_AXES_IN_CAMERA_FRAME.reshape(6,3), pose).reshape(3,2,3)
        plt.plot(cam_axes[0, :, 0], cam_axes[0, :, 1], cam_axes[0, :, 2],
                 color='#ff0000', linewidth=5)
        plt.plot(cam_axes[1, :, 0], cam_axes[1, :, 1], cam_axes[1, :, 2],
                 color='#00ff00', linewidth=5)
        plt.plot(cam_axes[2, :, 0], cam_axes[2, :, 1], cam_axes[2, :, 2],
                 color='#0000ff', linewidth=5)
        plt.plot(cam_trans[0], cam_trans[1], cam_trans[2], marker='o',
                 color=CAMERA_MARKER_COLORS[label], markersize=10, label=label)

    # Be prepared to return all the figures.
    figs = {}

    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(world_to_realsense[:, 0], world_to_realsense[:, 1],
            world_to_realsense[:, 2])
    plot_camera_triad(
        translations['realsense'], axis_angles['realsense'], 'realsense')
    ax.plot(ee_loc[0], ee_loc[1], ee_loc[2], c='r', label='EE Location')

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    plt.legend()

    ax.set_box_aspect([np.ptp(arr) for arr in \
                      [ax.get_xlim(), ax.get_ylim(), ax.get_zlim()]])
    figs['camera_poses'] = fig

    # First compute the EE location in camera frame.
    pose_cam = np.hstack((
        translations['realsense'],
        math_utils.axis_angle_to_quat(axis_angles['realsense'])))
    cam_extrinsics = math_utils.pos_quat_to_trans_mat(pose_cam)
    cam_inv = np.linalg.inv(cam_extrinsics)
    pose_cam_inv = np.hstack((
        cam_inv[:3, 3],
        math_utils.rotation_matrix_to_quat(cam_inv[:3, :3])
    ))
    ee_loc_in_cam = \
        math_utils.transform_point_coordinates_given_pose(
            ee_loc.reshape(-1, 3), pose_cam_inv)

    # Get the camera intrinsics.
    P_matrix = intrinsics['realsense'].reshape(3, 4)

    XYZ1 = np.hstack((ee_loc_in_cam, np.ones((1, 1))))
    uvw = P_matrix @ XYZ1.T
    x_pixel = uvw[0] / uvw[2]
    y_pixel = uvw[1] / uvw[2]

    ee_edges_in_world = get_ee_edges_in_3d(ee_loc)
    ee_edges_in_cam = math_utils.transform_point_coordinates_given_pose(
        ee_edges_in_world.reshape(-1, 3), pose_cam_inv)

    XYZ1 = np.hstack((ee_edges_in_cam, np.ones((ee_edges_in_cam.shape[0], 1))))
    uvw = P_matrix @ XYZ1.T
    xs_pixel = uvw[0] / uvw[2]
    ys_pixel = uvw[1] / uvw[2]

    fig = plt.figure()
    plt.imshow(image)
    plt.scatter(x_pixel, y_pixel)
    plt.scatter(xs_pixel, ys_pixel)
    plt.title('realsense')
    figs['realsense'] = fig

    pdb.set_trace()
    return figs

def get_ee_edges_in_3d(ee_loc):
    """Generate evenly spaced points on the surface of a 3D sphere.  Written
    with assistance from ChatGPT."""
    x, y, z = ee_loc[0], ee_loc[1], ee_loc[2]
    points = []
    phi = np.pi * (3. - np.sqrt(5.))  # Golden angle in radians

    for i in range(100):
        y = 1 - (i / float(100 - 1)) * 2  # y goes from 1 to -1
        radius_at_y = np.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = np.cos(theta) * radius_at_y
        z = np.sin(theta) * radius_at_y

        # Apply radius and center offset
        x = ee_loc[0] + EE_RADIUS * x
        y = ee_loc[1] + EE_RADIUS * y
        z = ee_loc[2] + EE_RADIUS * z

        points.append((x, y, z))

    return np.array(points)

def add_contact_visuals_to_figs(
        vision_asset: str, figs: dict, ps: Tensor, sdfs: Tensor, vs: Tensor,
        sdf_bounds: Tensor):
    """Add contact information to the realsense camera image, to verify the
    contact_in_cam conversion looks reasonable.

    TODO: Right now this just inspects ps and sdfs, not vs and sdf_bounds.
    """
    ps = np.array(ps)
    sdfs = np.array(sdfs)
    vs = np.array(vs)
    sdf_bounds = np.array(sdf_bounds)

    # Get the camera intrinsics.
    intrinsics, translations, axis_angles = \
        get_all_camera_intrinsics_extrinsics(vision_asset)

    # Do the realsense image.
    cam = 'realsense'
    P_matrix = intrinsics[cam].reshape(3, 4)

    XYZ1 = np.hstack((ps, np.ones((ps.shape[0], 1))))
    uvw = P_matrix @ XYZ1.T
    x_pixel = uvw[0] / uvw[2]
    y_pixel = uvw[1] / uvw[2]

    fig = figs[cam]
    ax = fig.gca()
    sdfs = ax.scatter(x_pixel, y_pixel, c=sdfs, cmap='coolwarm', vmin=-0.01,
                      vmax=0.01, label='SDF', s=10)
    cbar = fig.colorbar(sdfs)

def compute_cube_corners_in_world(pose):
    return math_utils.transform_point_coordinates_given_pose(
        CUBE_CORNERS_IN_CUBE_FRAME, pose)

def compute_camera_axes_in_world(cam_trans, cam_axis_angle):
    """"""
    xyzw = math_utils.axis_angle_to_quat(cam_axis_angle)
    pose = np.hstack((cam_trans, xyzw))
    return math_utils.transform_point_coordinates_given_pose(
        CAMERA_AXES_IN_CAMERA_FRAME, pose)

def compute_camera_z_normal(vision_asset: str) -> np.ndarray:
    """Compute the normal vector of the camera's z-axis in world frame."""
    object = vision_asset.split('_')[0]
    _cam_trans, cam_rot_axis_angle = file_utils.load_camera_extrinsics(object)
    cam_rot_axis_angle = cam_rot_axis_angle.squeeze()

    theta = np.linalg.norm(cam_rot_axis_angle)
    R_WC = math_utils.axis_angle_to_rotation_matrix(cam_rot_axis_angle, theta)
    return R_WC[:, 2]

def interactive_offset_adjustment(vision_asset: str, frame_num: int, smoothing: bool = False):
    """Plot the depth image and the cube corners in world frame."""
    print(DEPTH_PLOT_HELP_PRINT)

    # Load the depth image, TagSLAM pose, camera z direction.
    pose = load_tagslam_pose(vision_asset, frame_num)
    cube_corners_world = compute_cube_corners_in_world(pose)
    points_world = load_depth_image_as_points(vision_asset, frame_num, smoothing=smoothing)
    points_world, mask = crop_point_cloud(points_world, cube_xyz=pose[:3])
    camera_z_normal = compute_camera_z_normal(vision_asset)

    # Plot the depth image and cube corners.
    cm = plt.get_cmap('jet')
    cs = points_world[:, 2]
    cNorm = matplotlib.colors.Normalize(vmin=min(cs), vmax=max(cs))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)

    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    points = ax.scatter(points_world[:, 0], points_world[:, 1],
                        points_world[:, 2], s=1, c=scalarMap.to_rgba(cs),
                        label='Depth Image')
    ax.scatter(cube_corners_world[:, 0], cube_corners_world[:, 1],
               cube_corners_world[:, 2], c='r', s=10,
               label='Raw TagSLAM Cube Corners')
    edges = ax.plot(cube_corners_world[CUBE_CORNERS_PLOTTABLE_INDICES, 0],
                    cube_corners_world[CUBE_CORNERS_PLOTTABLE_INDICES, 1],
                    cube_corners_world[CUBE_CORNERS_PLOTTABLE_INDICES, 2],
                    c='r', label='Adjusted TagSLAM Cube Edges')[0]
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_xlim([pose[0]-CUBE_EXTRA_BUFFER, pose[0]+CUBE_EXTRA_BUFFER])
    ax.set_ylim([pose[1]-CUBE_EXTRA_BUFFER, pose[1]+CUBE_EXTRA_BUFFER])
    ax.set_zlim([pose[2]-CUBE_EXTRA_BUFFER, pose[2]+CUBE_EXTRA_BUFFER])
    ELEV, AZIM = -1, -75
    ax.view_init(elev=ELEV, azim=AZIM)
    plt.legend()

    def update_title(x, y, z, depth_offset, camera_offset, flush=True):
        plt.title(
            f'World TagSLAM Offset ({x:.03f}, {y:.03f}, {z:.03f})\n' + \
            f'Depth Z Offset {depth_offset:.03f}\n' + \
            f'Camera Z Offset {camera_offset:.03f}'
        )
        if flush:
            fig.canvas.draw()
            fig.canvas.flush_events()

    update_title(0, 0, 0, 0, 0)

    def reset(flush=True):
        points._offsets3d = (points_world[:, 0], points_world[:, 1],
                             points_world[:, 2])
        corners_plottable = cube_corners_world[CUBE_CORNERS_PLOTTABLE_INDICES]
        edges.set_data_3d(corners_plottable[:, 0], corners_plottable[:, 1],
                          corners_plottable[:, 2])
        update_title(0, 0, 0, 0, 0, flush=flush)

    def update(x, y, z):
        reset(flush=False)
        offset_corners = cube_corners_world + np.array([x, y, z])
        offset_plottable = offset_corners[CUBE_CORNERS_PLOTTABLE_INDICES]
        edges.set_data_3d(offset_plottable[:, 0], offset_plottable[:, 1],
                          offset_plottable[:, 2])
        update_title(x, y, z, 0, 0)

    def save_offset(x, y, z):
        update(x, y, z)
        offset = np.array([x, y, z])
        file_utils.save_tagslam_offset(offset)

    def depth_offset(offset):
        reset(flush=False)
        points_world_adj = load_depth_image_as_points(
            vision_asset, frame_num, z_axis_offset_meters=offset)
        points_world_adj = points_world_adj[mask]
        points._offsets3d = (points_world_adj[:, 0], points_world_adj[:, 1],
                             points_world_adj[:, 2])
        update_title(0, 0, 0, offset, 0)

    def camera_offset(offset):
        reset(flush=False)
        cam_offset_points = points_world + camera_z_normal * offset
        points._offsets3d = (cam_offset_points[:, 0], cam_offset_points[:, 1],
                             cam_offset_points[:, 2])
        update_title(0, 0, 0, 0, offset)

    def top_view():
        ax.view_init(elev=90, azim=-90)
        fig.canvas.draw()
        fig.canvas.flush_events()

    def side_view():
        ax.view_init(elev=0, azim=-90)
        fig.canvas.draw()
        fig.canvas.flush_events()

    def original_view():
        ax.view_init(elev=ELEV, azim=AZIM)
        fig.canvas.draw()
        fig.canvas.flush_events()

    print(DEPTH_PLOT_INSTRUCTIONS_PRINT)
    # plt.show()
    pdb.set_trace()

def load_tagslam_images(vision_asset: str, frame_num: int):
    """"""
    pass

def load_contact_information(
        vision_asset: str, cycle_iteration: int, pll_id: str):
    """Look for contact information stored in BundleSDF's geometry input
    directory."""
    geometry_dir = file_utils.bundlesdf_geometry_dir(
        dataset=vision_asset, cycle_iteration=cycle_iteration,
        pll_run_id=pll_id, create=False)
    if not op.exists(geometry_dir):
        print(f'No geometry found at {geometry_dir}.')
        return None

    from_mesh_dir = op.join(geometry_dir, 'contact_in_cam', 'from_mesh_surface')
    from_supports_dir = op.join(
        geometry_dir, 'contact_in_cam', 'from_support_points')

    # Iterate over all the frames with contact information.
    for frame_num in os.listdir(from_supports_dir):
        print(f'Loading frame {frame_num}...')
        ps = torch.load(op.join(from_supports_dir, frame_num, 'ps.pt'))
        sdfs = torch.load(op.join(from_supports_dir, frame_num, 'sdfs.pt'))

        vs = torch.load(op.join(from_mesh_dir, frame_num, 'vs.pt'))
        sdf_bounds = torch.load(
            op.join(from_mesh_dir, frame_num, 'sdf_bounds.pt'))

        figs = inspect_camera_poses_and_images(vision_asset, int(frame_num))

        add_contact_visuals_to_figs(
            vision_asset=vision_asset, figs=figs, ps=ps, sdfs=sdfs, vs=vs,
            sdf_bounds=sdf_bounds
        )
        pdb.set_trace()



if __name__ == '__main__':
    # interactive_offset_adjustment('cube_2', 1, smoothing=False)
    # inspect_camera_poses_and_images('cube_2')
    # inspect_tagslam_times("cube_2", 1)
    # load_contact_information('cube_2', 1, 'pll_id_p10')

    inspect_robot_pose_and_image('robot_bakingbox_sticky_A_1')
