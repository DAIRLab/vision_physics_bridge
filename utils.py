import numpy as np
import math

def filter(real_img, sim_img):
    """
    Filter the simulated image from the real depth image.
    """
    masked_img = np.subtract(real_img, sim_img)
    return masked_img

def import_data(img_file, position_file, velocity_file):
    """
    Load data generated from rosbag.
    """
    images = np.loadtxt(img_file)
    positions = np.loadtxt(position_file)
    velocities = np.loadtxt(velocity_file)
    return images, positions, velocities

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
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

def get_table_world_coordinates():
    """
    Get table's world coordinate with camera extrinsics. X is towards to camera. Y is towards to right of the camera. Z is upward. 
    Note: Assume robot is in the middle of the table.
    return: (a, b, c, d) that corresponds to the left bottom, right bottom, left top, right top corners of the table.
    """
    length = 182.8/100
    width = 91.3/100
    height = 73.2/100
    robot_to_length = 46.9
    robot_to_width = 33.7 
    robot_to_height = 1
    # camera_x = 1.14164360
    # camera_y = 0.15815239
    # camera_z = 0.66422200

    left_bottom = [length - robot_to_width, -width / 2, height-robot_to_height]
    right_bottom = [length - robot_to_width, width / 2, height-robot_to_height]
    left_top = [-robot_to_width, -width/2, height-robot_to_height]
    right_top = [robot_to_width, width/2, height-robot_to_height]
    return left_bottom, right_bottom, left_top, right_top

def world_to_point_cloud(X, Y, Z, K, R, depth_scale=1000):  
    """
    World coordinate = (X, Y, Z). In world coordinate, d is in -X.
    u = 1/Z * [fx, 0, cz] @ R0 * X
    v = 1/Z * [0, fy, cy] @ R1 * Y
    Transform depth image to point cloud 
    z = d / depth_scale = -X / depth_scale
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    """
    u = 1/Z * (K[0] @ R[:, 0]) * X
    v = 1/Z * (K[1] @ R[:, 1]) * Y
    z = -X / depth_scale
    # print(u)
    # print(K[0,0])
    x = (u - K[0][1]) * z / K[0][0]
    y = (v - K[1][2]) * z / K[1][1]
    z = -X / depth_scale
    return x, y, z

def perspective_projection(X, Y, Z, fx, fy):
    """
    Transform real world points to points on the image plane.
    """
    x = fx * X / Z
    y = fy * Y / Z
    return x, y
