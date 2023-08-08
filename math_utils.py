import numpy as np
import math
import tf.transformations as tr
from scipy.spatial.transform import Rotation as R


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

def rotation_matrix_to_quaternion(R):
    """
    :param R: 4*4 transformation matrix.
    """
    return tr.quaternion_from_matrix(R)

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
    extrinsic = setup_extrinsic(translation, axis_vec)
    Rw2c = extrinsic[:3, :3]
    R_w = m[:3, :3]
    R_c = Rw2c @ R_w

    pos_world_vector = m[:, 3]
    pos_camera_vector = extrinsic @ pos_world_vector
    T_c = pos_camera_vector[:3]
    T_c = T_c.reshape(-1, 1)

    return np.vstack((np.hstack((R_c, T_c)), np.array([0, 0, 0, 1])))


def camera_to_world(m, translation, axis_vec):
    """
    Transform camera coordinates to world coordinates.
    :param m: 4*4 transformation matrix in camera frame
    :param translation: camera translation in world frame
    :param axis_vec: camera axis vector in world frame
    """
    extrinsic = setup_extrinsic(translation, axis_vec)
    # Rw2c = extrinsic[:3, :3]
    # R_c = m[:3, :3]
    # R_w = Rw2c.T @ R_c

    # pos_camera_vector = m[:, 3]
    # pos_world_vector = np.linalg.inv(extrinsic) @ pos_camera_vector
    # T_w = pos_world_vector[:3]
    # T_w = T_w.reshape(-1, 1)
    # return np.vstack((np.hstack((R_w, T_w)), np.array([0, 0, 0, 1])))
    # return np.linalg.inv(extrinsic) @ m
    return extrinsic @ m


CAMERA_CONFIG = {
    "old": {
        "translation": np.array([[1.14164360], [0.15815239], [0.66422200]]),
        "axis_vec": np.array([-1.57165949, -1.63112887, 1.07928078]),
    },
    "new": {
        "translation": np.array([[1.11076422], [-0.07966290], [0.67947702]]),
        "axis_vec": np.array([-1.61997882, -1.56988553, 0.86362178]),
    },
}


def transform_bundletrack_output(
    pred_pose, output_pose_dir, odom_file_dir
):
    """
    https://github.com/wenbowen123/BundleTrack/issues/38
    """
    init_pose = np.loadtxt(
        output_pose_dir + "%04i.txt" % 1
    )  # initial cube pose in camera frame, bundletrack's internal coordinate system
    init_pose_new = np.loadtxt(
        odom_file_dir + "%04i.txt" % 0
    )  # initial cube pose represented in camera frame, matching tagslam
    pred_new = (pred_pose @ np.linalg.inv(init_pose)) @ init_pose_new
    
    # cam_translation = CAMERA_CONFIG['old']['translation']
    # cam_axis_vec = CAMERA_CONFIG['old']['axis_vec']
    # pred_new_world = camera_to_world(pred_new, cam_translation, cam_axis_vec)
    # return pred_new_world
    return pred_new


def setup_extrinsic(translation, axis_vec):
    # Setup camera extrinsic
    angle = np.linalg.norm(axis_vec)
    axis = axis_vec / angle
    rotation = axis_angle_to_rotation_matrix(
        axis, angle
    )  # directions of the world-axes in camera coordinates
    rotation_prime = rotation.T  # USE THIS
    translation_prime = -rotation_prime @ translation  # USE THIS
    extrinsic = np.vstack(
        (np.hstack((rotation_prime, translation_prime)), np.array([0, 0, 0, 1]))
    )
    return extrinsic


def pos_quat_to_trans_mat(pos_quat):
    quat = pos_quat[3:]
    rot = R.from_quat(quat).as_matrix()
    trans = pos_quat[:3].reshape(-1, 1)
    return np.vstack((np.hstack((rot, trans)), np.array([0, 0, 0, 1])))


def trans_mat_to_pos_quat(trans):
    quat = R.from_matrix(trans[:3, :3]).as_quat().reshape(-1, 1)
    pos = trans[:3, 3].reshape(-1, 1)
    return np.vstack((pos, quat))
