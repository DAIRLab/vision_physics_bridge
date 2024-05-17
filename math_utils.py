import math
import numpy as np
import os.path as op
import rospy
from PIL import Image
from scipy.spatial.transform import Rotation as R
import sys
import torch
from torch import Tensor

DATA_GEN_DIR = op.dirname(op.realpath(__file__))
REPO_DIR = op.dirname(DATA_GEN_DIR)
PLL_DIR = op.join(REPO_DIR, 'dair_pll')

sys.path.append(PLL_DIR)    # For importing dair_pll.

from dair_pll import quaternion


def get_deep_support_query_directions() -> Tensor:
    """Get roughly evenly-spaced query directions."""
    linear_space = torch.linspace(-1, 1, steps=32)
    grid = torch.cartesian_prod(linear_space, linear_space, linear_space)
    points_on_box_surface = grid[grid.abs().max(dim=-1).values >= 1.0]
    surface = points_on_box_surface / points_on_box_surface.norm(
        dim=-1, keepdim=True)
    return surface.to(torch.float64)


def log_mean(a, b):
    """Compute the geometric mean of two numbers."""
    return np.exp((np.log(a) + np.log(b)) / 2)


def extract_floats_from_camk(lines):
    # Initialize an empty list to store the extracted floats
    extracted_floats = []

    # Iterate through each line
    for line in lines:
        # Split the line into individual elements using spaces
        elements = line.split()

        # Convert each element to a float and append to the list
        floats = [float(element) for element in elements]
        extracted_floats.append(floats)
    return extracted_floats


def convert_relative_frames_to_absolute(
        relative_frames: np.ndarray, full_times: np.ndarray,
        ros_times: rospy.rostime.Time) -> np.ndarray:
    """Converts frames relative to the start of a subsection of a longer
    trajectory to frames as absolute indices of the full trajectory.
    
    Args:
        relative_frames: 1D numpy array of relative frame indices.
        full_times: 1D numpy array of timestamps for the full trajectory.
        ros_times: 1D numpy array of ROS timestamps for the start of each
            subsection.

    Returns:
        absolute_frames: 1D numpy array of absolute frame indices.
    """
    start_times = ros_time_to_float(ros_times)
    absolute_frames = np.zeros_like(relative_frames)

    for i in range(len(relative_frames)):
        subsection_start_frame = np.argmin(np.abs(full_times - start_times[i]))
        absolute_frames[i] = subsection_start_frame + relative_frames[i]
    
    return absolute_frames


def convert_depth_image_to_points(
        depth_image: np.ndarray, fx: float, fy: float, cx: float, cy: float
) -> np.ndarray:
    """Convert a depth image to a point cloud represented in meters.  This
    conversion disregards all points that are non-returns (i.e. depth == 0)."""
    height, width = depth_image.shape

    # Generate pixel grid.
    x = np.arange(0, width)
    y = np.arange(0, height)
    xv, yv = np.meshgrid(x, y)

    # Calculate corresponding 3D coordinates.
    X = (xv - cx) * depth_image / fx
    Y = (yv - cy) * depth_image / fy
    Z = depth_image

    # Stack the coordinates and reshape.
    point_cloud = np.stack((X, Y, Z), axis=-1)
    point_cloud = point_cloud.reshape((-1, 3))

    # Filter out the non-returns and convert millimeters to meters.
    point_cloud = point_cloud[np.any(point_cloud != 0, axis=1)] / 1000.0

    return point_cloud


def load_depth_image(depth_image_path: str) -> np.ndarray:
    """Load a depth image at depth_image_path and return it as a numpy array."""
    return np.array(Image.open(depth_image_path))


def load_depth_image_to_points(
        depth_image_path: str, fx: float, fy: float, cx: float, cy: float
) -> np.ndarray:
    """Load a depth image at depth_image_path and convert it to a point cloud
    represented in meters.  This conversion disregards all points that are non-
    returns (i.e. depth == 0)."""
    depth_image = load_depth_image(depth_image_path)
    return convert_depth_image_to_points(depth_image, fx, fy, cx, cy)


def transform_point_coordinates_given_pose(
        points_in_A: np.ndarray, pose_A_in_B: np.ndarray) -> np.ndarray:
    """Transform a set of points represented in frame A to being represented in
    frame B, given the pose of frame A in frame B.  Interpret the pose as having
    order [x, y, z, qx, qy, qz, qw]."""
    assert points_in_A.ndim == 2 and points_in_A.shape[1] == 3
    assert pose_A_in_B.ndim == 1 and pose_A_in_B.shape[0] == 7

    xyz = pose_A_in_B[:3]
    quat_xyzw = pose_A_in_B[3:]

    rotation_matrix = R.from_quat(quat_xyzw).as_matrix()

    points_in_world = (rotation_matrix @ points_in_A.T).T + xyz
    return points_in_world


def ros_time_to_float(ros_times: np.ndarray) -> np.ndarray:
    """Converts ROS timestamps to floating point numbers.
    
    Args:
        ros_time: 1D numpy array of ROS timestamps.

    Returns:
        float_time: 1D numpy array of floating point timestamps.
    """
    float_time = np.zeros_like(ros_times)

    for i in range(len(ros_times)):
        float_time[i] = ros_times[i].secs + ros_times[i].nsecs * 1e-9
    
    return float_time


def ros_geometry_transform_to_camera_extrinsics(ros_tf):
    """Converts a ROS geometry transform message to a position-axisangle tuple.
    """
    cam_trans = np.array([ros_tf.translation.x,
                    ros_tf.translation.y,
                    ros_tf.translation.z])
    quat = np.array([ros_tf.rotation.x,
                     ros_tf.rotation.y,
                     ros_tf.rotation.z,
                     ros_tf.rotation.w])
    cam_rot_axis_angle = quat_to_axis_angle(quat)
    return cam_trans, cam_rot_axis_angle


def axis_angle_to_rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array(
        [
            [aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
            [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
            [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc],
        ]
    )


def rotation_matrix_to_euler(R):
    # beta = -np.arcsin(R[2, 0])
    # alpha = np.arctan2(R[2, 1] / np.cos(beta), R[2, 2] / np.cos(beta))
    # gamma = np.arctan2(R[1, 0] / np.cos(beta), R[0, 0] / np.cos(beta))
    # return np.array((alpha, beta, gamma))
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])


def world_to_image(point, K, R, T):
    """
    :param point: 3*1 array.
    :param K: 3*3 intrinsic matrix.
    :param R: 3*3 rotation matrix.
    :param T: 3*1 translation matrix.
    """
    world_coord = K @ R @ point + K @ T
    print("world", world_coord.shape)
    return world_coord[0] / world_coord[2], world_coord[1] / world_coord[2]


def world_to_point_cloud(X, Y, Z, K, R, T, depth_scale=1000):
    """
    World coordinate = (X, Y, Z). In world coordinate, d is in -X.
    u = 1/Z * [fx, 0, cz] @ R0 * X
    v = 1/Z * [0, fy, cy] @ R1 * Y
    Transform depth image to point cloud (in the image frame)
    z = d / depth_scale = -X / depth_scale
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    """
    # u = 1/Z * (K[0] @ R[:, 0]) * X
    # v = 1/Z * (K[1] @ R[:, 1]) * Y
    u, v = world_to_image(np.array([[X], [Y], [Z]]), K, R, T)
    z = -X / depth_scale
    x = (u - K[0][1]) * z / K[0][0]
    y = (v - K[1][2]) * z / K[1][1]
    z = -X / depth_scale
    return x, y, z


def world_to_camera(m, translation, axis_vec):
    """
    :param m: 4*4 transformation matrix in world frame
    :param translation: camera translation in world frame
    :param axis_vec: camera axis vector in world frame
    """
    extrinsic = extrinsics_T_WC(translation, axis_vec)
    return extrinsic @ m


def camera_to_world(m, translation, axis_vec):
    """
    Transform camera coordinates to world coordinates.
    :param m: 4*4 transformation matrix in camera frame
    :param translation: camera translation in world frame
    :param axis_vec: camera axis vector in world frame
    """
    extrinsic = extrinsics_T_CW(translation, axis_vec)
    return extrinsic @ m


def transform_bundletrack_origin_to_tagslam_origin(
    pred_pose, bsdf_output_pose_dir, annotated_poses_dir, translation, axis_vec,
    to_world=False
):
    """This function uses the equation from the below BundleTrack issue:

    https://github.com/wenbowen123/BundleTrack/issues/38

    The expression involves the below transformations:
        - camera_T_Bn (pred_pose):  BundleSDF's reported pose of the BundleSDF
            origin at the nth time stamp, in camera frame.
        - camera_T_B1 (init_pose):  BundleSDF's reported pose of the BundleSDF
            origin at the first time stamp, in camera frame.
        - camera_T_T0 (init_pose_new):  TagSLAM's reported pose of the TagSLAM
            origin at the 0th time stamp, in camera frame.
        - camera_T_Tn (pred_new):  The desired pose of the TagSLAM origin at the
            nth time stamp, in camera frame.
        - world_T_Tn (pred_new_world):  The desired pose of the TagSLAM origin
            at the nth time stamp, in world frame.
        - world_T_camera:  Required camera extrinsics.

    The first desired result is camera_T_Tn, which is obtained using the
    following identity:

        Bn_T_Tn = B0_T_T0  <-- due to rigid body, relative origin offset fixed.

    Thus camera_T_Tn is obtained via:

        camera_T_Tn = camera_T_Bn * Bn_T_Tn
                    = camera_T_Bn * B0_T_T0
                    = camera_T_Bn * B0_T_camera * camera_T_T0
                    = camera_T_Bn * inv(camera_T_B0) * camera_T_T0

    Or:    pred_new =    pred     *  inv(init_pose)  * init_pose_new
    """
    # This is camera_T_B1, the BundleSDF origin wrt camera at 1st timestamp.
    init_pose = np.loadtxt(op.join(bsdf_output_pose_dir, "0001.txt"))

    # This is camera_T_T0, the TagSLAM origin wrt camera at 0th timestamp.
    init_pose_new = np.loadtxt(op.join(annotated_poses_dir, "0000.txt"))

    # This converts camera_T_Bn to camera_T_Tn, switching from reporting pose of
    # BundleSDF origin to TagSLAM origin.
    pred_new = (pred_pose @ np.linalg.inv(init_pose)) @ init_pose_new

    if to_world:
        # This yields world_T_Tn.
        pred_new_world = camera_to_world(pred_new, translation, axis_vec)
        return pred_new_world
    
    return pred_new


def transform_points_wrt_tagslam_origin_to_bundletrack_origin(
        points_wrt_T, bsdf_output_pose_dir, annotated_poses_dir
):
    """This function aims to obtain B_T_p from T_T_p where B is the BundleSDF
    origin of the body, T is the TagSLAM origin of the body, and p is a point on
    the surface of the body.  The point p is represented as a 4x4 transformation
    matrix, where the orientation does not matter since it's a point.

        B_T_p = B_T_T * T_T_p
              = (B_T_camera * camera_T_T) * T_T_p
              = (inv(camera_T_B0 * camera_T_T0)) * T_T_p

    Or:  points_wrt_B = inv(init_pose) * init_pose_new * points_wrt_T

    Args:
        points_wrt_T (*, 4, 4):  possibly batched 4x4 transformation matrices
            representing points expressed in TagSLAM's body frame.
        bsdf_output_pose_dir:  the output directory of the BundleSDF run that
            the PLL run used.  0001.txt in this directory is the first BundleSDF
            origin pose wrt the camera.
        annotated_poses_dir:  the annotated poses directory associated with the
            BundleSDF run.  0000.txt in this directory is the first TagSLAM
            origin pose wrt the camera.

    Output:
        points_wrt_B (*, 4, 4):  possibly batched 4x4 transformation matrices
            representing points expressed in BundleSDF's body frame.  Is
            returned as the same shape as the input.
    """
    original_shape = points_wrt_T.shape

    if points_wrt_T.ndim == 2:
        points_wrt_T = points_wrt_T.reshape(1, 4, 4)
    assert points_wrt_T.shape[1:] == (4, 4), f'Expecting (N, 4, 4) shape ' + \
        f'but found {points_wrt_T.shape=}.'

    # This is camera_T_B1, the BundleSDF origin wrt camera at 1st timestamp.
    init_pose = np.loadtxt(op.join(bsdf_output_pose_dir, "0001.txt"))

    # This is camera_T_T0, the TagSLAM origin wrt camera at 0th timestamp.
    init_pose_new = np.loadtxt(op.join(annotated_poses_dir, "0000.txt"))

    points_wrt_B = (np.linalg.inv(init_pose) @ init_pose_new) @ points_wrt_T
    return points_wrt_B.reshape(original_shape)


def extrinsics_T_WC(translation, axis_vec):
    """
    Convert translation and axis-angle representation to extrinsic matrix.

    Parameters:
    - translation: 3x1 numpy array, translation vector.
    - axis_vec: 3x1 numpy array, rotation represented in axis-angle (rodriques) form.

    Returns:
    - 4x4 numpy array, extrinsic matrix.
    """
    rotation_matrix = R.from_rotvec(axis_vec.ravel()).as_matrix()
    rotation_inverse = rotation_matrix.T
    translation_inverse = -rotation_inverse @ translation.ravel()
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = rotation_inverse
    extrinsic[:3, 3] = translation_inverse

    return extrinsic


def extrinsics_T_CW(translation, axis_vec):
    """
    Convert translation and axis-angle representation to extrinsic matrix.
    
    Parameters:
    - translation: 3x1 numpy array, translation vector.
    - axis_vec: 3x1 numpy array, rotation represented in axis-angle (rodriques) form.

    Returns:
    - 4x4 numpy array, extrinsic matrix.
    """
    rotation_matrix = R.from_rotvec(axis_vec.ravel()).as_matrix()
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = rotation_matrix
    extrinsic[:3, 3] = translation.squeeze()
    
    return extrinsic


def inverse_homogeneous_transformation(transform: Tensor) -> Tensor:
    """Produce the inverse of a homogeneous transform.  If a homogeneous
    transformation matrix is of the form:

        T = [ R  d ]
            [ 0  1 ]

    for T (4,4), R (3,3), and d (3,1), then the inverse is:
    
        inv(T) = [ R^T  -R^T*d ]
                 [  0      1   ]
    """
    assert transform.shape == (4, 4)

    # Split into rotation and translation components.
    rot_mat = transform[:3, :3]
    translation = transform[:3, 3]

    # Build the inverse.
    inv_transform = np.zeros((4, 4))
    inv_transform[:3, :3] = rot_mat.T
    inv_transform[:3, 3] = -rot_mat.T @ translation
    inv_transform[3, 3] = 1

    return inv_transform


def axis_angle_to_quat(axis_angle):
    """Convert axis-angle to quaternion.  Returns xyzw ordering."""
    return R.from_rotvec(axis_angle).as_quat()


def quat_to_rotation_matrix(quat):
    """Convert quaternion to rotation matrix.  Assumes xyzw ordering."""
    return R.from_quat(quat).as_matrix()


def rotation_matrix_to_quat(rot_mat):
    """Convert rotation matrix to quaternion.  Returns xyzw ordering."""
    return R.from_matrix(rot_mat).as_quat()


def quat_to_axis_angle(quat):
    """Convert quaternion to axis-angle.  Assumes xyzw ordering."""
    return R.from_quat(quat).as_rotvec()


def pos_quat_to_trans_mat(pos_quat):
    """Converts position-quaternion to transformation matrix.  Assumes
    quaternion is in xyzw ordering."""
    quat = pos_quat[3:]
    rot = quat_to_rotation_matrix(quat)
    trans = pos_quat[:3].reshape(-1, 1)
    return np.vstack((np.hstack((rot, trans)), np.array([0, 0, 0, 1])))


def trans_mat_to_pos_quat(trans):
    """Converts transformation matrix to position-quaternion.  Returns
    quaternion in xyzw ordering."""
    q = R.from_matrix(trans[:3, :3]).as_quat().reshape(-1, 1)
    magnitude = np.linalg.norm(q)
    q /= magnitude
    if ((q[3] < -0.01)
        or (q[3] == 0 and q[0] < 0)
        or (q[3] == 0 and q[0] == 0 and q[1] < 0)
        or (q[3] == 0 and q[0] == 0 and q[1] == 0 and q[2] < 0)):
        q[0] *= -1.0
        q[1] *= -1.0
        q[2] *= -1.0
        q[3] *= -1.0
    pos = trans[:3, 3].reshape(-1, 1)
    return np.vstack((pos, q))

def slerp(q0, q1, t_array):
    """Spherical linear interpolation between two quaternions."""
    dot = np.dot(q0, q1)
    
    # If the dot product is negative, slerp won't take the shorter path.
    if dot < 0.0:
        q1 = -q1
        dot = -dot
        
    DOT_THRESHOLD = 0.9995
    if dot > DOT_THRESHOLD:
        # If the inputs are too close for comfort, linearly interpolate and normalize the result.
        result = q0 + t_array[:, np.newaxis] * (q1 - q0)
        return result / np.linalg.norm(result, axis=1)[:, np.newaxis]
    
    # Compute the quaternion of the rotation angle
    theta_0 = np.arccos(dot)
    theta = theta_0 * t_array
    q2 = q1 - q0 * dot
    q2 /= np.linalg.norm(q2)
    
    return np.cos(theta)[:, np.newaxis] * q0 + np.sin(theta)[:, np.newaxis] * q2

def wxyz2xyzw(quat_wxyz):
    """
    quat_wxyz: (N,4) or (4,)
    """
    original_shape = quat_wxyz.shape

    if len(original_shape) == 1:
        assert original_shape[0] == 4
        quat_wxyz = quat_wxyz.reshape(1, 4)

    w = quat_wxyz[:, 0:1]
    xyz = quat_wxyz[:, 1:]
    return np.concatenate((xyz, w), axis=1).reshape(original_shape)

def xyzw2wxyz(quat_xyzw):
    """
    quat_xyzw: (N,4) or (4,)
    """
    original_shape = quat_xyzw.shape

    if len(original_shape) == 1:
        assert original_shape[0] == 4
        quat_xyzw = quat_xyzw.reshape(1, 4)

    xyz = quat_xyzw[:, 0:3]
    w = quat_xyzw[:, 3:4]
    return np.concatenate((w, xyz), axis=1).reshape(original_shape)


def quaternion_error(quat1_wxyz, quat2_wxyz):
    """Input quaternions must be in wxyz format.  Returns the angular error in
    radians between the two quaternions over time.  Inputs can be (N, 4) or
    (4,)."""
    # Check inputs.
    assert quat1_wxyz.shape == quat2_wxyz.shape
    if quat1_wxyz.ndim == 1:
        assert quat1_wxyz.shape[0] == 4
        quat1_wxyz = quat1_wxyz.reshape(1, 4)
        quat2_wxyz = quat2_wxyz.reshape(1, 4)

    quat1_wxyz = torch.Tensor(quat1_wxyz)
    quat2_wxyz = torch.Tensor(quat2_wxyz)

    quat_shift = quaternion.multiply(quaternion.inverse(quat1_wxyz), quat2_wxyz)
    rot = quaternion.log(quat_shift)

    return torch.sqrt((rot**2).sum(dim=-1))
