import math
import numpy as np
import os.path as op
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
    return extrinsic @ m


def camera_to_world(m, translation, axis_vec):
    """
    Transform camera coordinates to world coordinates.
    :param m: 4*4 transformation matrix in camera frame
    :param translation: camera translation in world frame
    :param axis_vec: camera axis vector in world frame
    """
    extrinsic = setup_extrinsic(translation, axis_vec)
    return np.linalg.inv(extrinsic) @ m


def transform_bundletrack_output(
    pred_pose, output_pose_dir, odom_file_dir, translation, axis_vec, to_world=False
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
    init_pose = np.loadtxt(op.join(output_pose_dir, "0001.txt"))

    # This is camera_T_T0, the TagSLAM origin wrt camera at 0th timestamp.
    init_pose_new = np.loadtxt(op.join(odom_file_dir, "0000.txt"))

    # This converts camera_T_Bn to camera_T_Tn, switching from reporting pose of
    # BundleSDF origin to TagSLAM origin.
    pred_new = (pred_pose @ np.linalg.inv(init_pose)) @ init_pose_new

    if to_world:
        # This yields world_T_Tn.
        pred_new_world = camera_to_world(pred_new, translation, axis_vec)
        return pred_new_world
    
    return pred_new


def setup_extrinsic(translation, axis_vec):
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


def pos_quat_to_trans_mat(pos_quat):
    quat = pos_quat[3:]
    rot = R.from_quat(quat).as_matrix()
    trans = pos_quat[:3].reshape(-1, 1)
    return np.vstack((np.hstack((rot, trans)), np.array([0, 0, 0, 1])))


def trans_mat_to_pos_quat(trans):
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
    quat_wxyz: (N,4)
    """
    w = quat_wxyz[:, 0:1]
    xyz = quat_wxyz[:, 1:]
    return np.concatenate((xyz, w), axis=1)

def xyzw2wxyz(quat_xyzw):
    """
    quat_xyzw: (N,4)
    """
    xyz = quat_xyzw[:, 0:3]
    w = quat_xyzw[:, 3:4]
    return np.concatenate((w, xyz), axis=1)