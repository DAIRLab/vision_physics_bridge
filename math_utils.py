from webbrowser import get
import numpy as np
import math
import tf.transformations as tr


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
    beta = -np.arcsin(R[2, 0])
    alpha = np.arctan2(R[2, 1] / np.cos(beta), R[2, 2] / np.cos(beta))
    gamma = np.arctan2(R[1, 0] / np.cos(beta), R[0, 0] / np.cos(beta))
    return np.array((alpha, beta, gamma))


def quaternion_to_rotation_matrix(Q):
    """
    Covert a quaternion into a full three-dimensional rotation matrix.

    Input
    :param Q: A 4 element array representing the quaternion (q0,q1,q2,q3)

    Output
    :return: A 3x3 element matrix representing the full 3D rotation matrix.
             This rotation matrix converts a point in the local reference
             frame to a point in the global reference frame.
    """
    # Extract the values from Q
    q0 = Q[0]
    q1 = Q[1]
    q2 = Q[2]
    q3 = Q[3]

    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)

    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)

    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1

    # 3x3 rotation matrix
    rot_matrix = np.array([[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]])
    return rot_matrix


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
    Rw2c = extrinsic[:3, :3]
    R_c = m[:3, :3]
    R_w = Rw2c.T @ R_c

    pos_camera_vector = m[:, 3]
    pos_world_vector = np.linalg.inv(extrinsic) @ pos_camera_vector
    T_w = pos_world_vector[:3]

    # T_c = m[:3, 3]
    # T_w = -Rw2c.T @ T_c
    T_w = T_w.reshape(-1, 1)
    return np.vstack((np.hstack((R_w, T_w)), np.array([0, 0, 0, 1])))


def transform_bundletrack_output_to_world(pred_pose, cam_translation, cam_axis_vec):
    """
    https://github.com/wenbowen123/BundleTrack/issues/38
    """
    OUTPUT_POSE_DIR = "/home/cnets-vision/mengti_ws/poses/"
    ODOM_FILE_PATH = "/home/cnets-vision/mengti_ws/BundleTrack/Data/YCBINEOAT/contact_nets/annotated_poses/"
    init_pose = np.loadtxt(
        OUTPUT_POSE_DIR + "%04i.txt" % 1
    )  # initial cube pose in camera frame, bundletrack's internal coordinate system
    init_pose_new = np.loadtxt(
        ODOM_FILE_PATH + "%04i.txt" % 0
    )  # initial cube pose represented in camera frame, matching tagslam
    pred_new = (pred_pose @ np.linalg.inv(init_pose)) @ init_pose_new
    pred_new_world = camera_to_world(pred_new, cam_translation, cam_axis_vec)
    return pred_new_world


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


def get_angular_velocity(curr_state, next_state, dt):
    R_diff = next_state @ np.linalg.inv(curr_state)
    R = np.vstack((np.hstack((R_diff, np.zeros((3, 1)))), np.array([0, 0, 0, 1])))
    Q = rotation_matrix_to_quaternion(R)
    # TODO: Double check the indices meaning.
    axis = np.linalg.norm(np.array(Q[1:]))
    angle = 2 * np.arctan2(axis, Q[0])
    ang_velocity = np.array(Q[1:]) * angle / dt
    return ang_velocity.reshape(1, -1)


def get_linear_velocity(curr_state, next_state, dt):
    return (next_state - curr_state) / dt
