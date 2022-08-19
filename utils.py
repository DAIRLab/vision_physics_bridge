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

def rotation_matrix_to_euler(R):
    beta = -np.arcsin(R[2,0])
    alpha = np.arctan2(R[2,1]/np.cos(beta),R[2,2]/np.cos(beta))
    gamma = np.arctan2(R[1,0]/np.cos(beta),R[0,0]/np.cos(beta))
    return np.array((alpha, beta, gamma))

def get_table_world_coordinates():
    """
    Get table's world coordinate with camera extrinsics. X is towards to camera. Y is towards to right of the camera(facing the robot). Z is upward. 
    Note: Assume robot is in the middle of the table.
    return: (a, b, c, d) that corresponds to the left bottom, right bottom, left top, right top corner of the table.
    """
    length = 182.8/100 #in meters
    width = 91.3/100
    height = 73.2/100
    camera_buffer = 1
    robot_to_length = 46.9/100
    robot_to_width = 33.7/100
    robot_to_height = 1/100
    camera_x = 1.14164360
    camera_y = 0.15815239
    camera_z = 0.66422200
    # left_bottom = [camera_x+camera_buffer, -width/2, height-robot_to_height]
    # right_bottom = [camera_x+camera_buffer, width/2, height-robot_to_height]
    # left_top = [-robot_to_width, -width/2, height-robot_to_height]
    # right_top = [-robot_to_width, width/2, height-robot_to_height]
    # For some reason the ros and open3d have inverted x, y orientation. 
    # Also, the origin is at (0,0,0) instead of the robot base.
    translation_y = -1
    left_bottom = [-width/2, camera_x+translation_y, height-robot_to_height]
    right_bottom = [width/2, camera_x+translation_y, height-robot_to_height]
    left_top = [-width/2, -robot_to_width+translation_y, height-robot_to_height]
    right_top = [width/2, -robot_to_width+translation_y, height-robot_to_height]
    return left_bottom, right_bottom, left_top, right_top

def world_to_image(point, K, R, T):
    """
    point: 3*1 array 
    K: 3*3 intrinsic matrix 
    R: 3*3 rotation matrix
    T: 3*1 translation matrix
    """
    # 3dpoint tmp = cameraintrinsic(3x3) * rotationvect(1x3) * xyz(3x1) + cameraintrinsic(3x3)*translation(3,1)
    # 2dpoint screen = tmp.x/tmp.z, tmp.y / tmp.z
    world_coord = K @ R @ point + K @ T
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

def find_x_at_max_y(pcd_array, y):
    for arr in pcd_array:
        if arr[1] == y:
            x = arr[0]
            break
    return x

def get_boundary(pcd):
    """
    [Open3D INFO] Picked point #197438 (0.31, 0.35, -0.62) to add in queue.
    [Open3D INFO] Picked point #195816 (0.44, 0.87, -1.4) to add in queue.
    [Open3D INFO] Picked point #65344 (-0.13, 1.5, -0.96) to add in queue.
    [Open3D INFO] Picked point #60460 (-0.24, 0.98, -0.26) to add in queue.
    """
    min_x = pcd.get_min_bound()[0]
    max_x = pcd.get_max_bound()[0]
    max_y = pcd.get_max_bound()[1]
    min_y = pcd.get_min_bound()[1]
    min_z = pcd.get_min_bound()[2]
    
    height = 0.732
    left_bottom = np.array((min_x, min_y, min_z))
    # left_bottom_ =np.array((find_x_at_max_y(np.asarray(pcd.points), max_y), max_y, height))
    left_bottom_ = np.array((max_x, max_y, height))
    # x towards the robot
    # [-1.78202008  0.35017328 -2.00775471] [0.92882352 3.76741797 0.732     ]
    return left_bottom, left_bottom_

