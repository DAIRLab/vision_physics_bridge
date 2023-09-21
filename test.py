import numpy as np
import os
import time
from file_utils import load_toss_time_from_yaml
import yaml

import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf

from tempfile import TemporaryDirectory

from PIL import Image
import io
from math_utils import trans_mat_to_pos_quat, transform_bundletrack_output
# video resolution
video_resolution=[640, 480]

#
## parameters:
#
toss_id=1
cam = 'cam0'
video_name = f'cube_hand_{toss_id}'
cam_poses_file = './assets/realsense_pose_cube_hand_60_2.yaml'
output_file = './' + video_name + '.mp4'
yaml_path = './assets/config.yaml'
toss_type = 'cube'

intrinsics = [380.2484436035156, 379.8265380859375,314.2138977050781, 240.59800720214844,] # fx, fy, cx, cy

with open(cam_poses_file, 'r') as stream:
    data_loaded = yaml.safe_load(stream)
print(data_loaded[cam]['pose']['position']['x'])

cam_pos_dict = data_loaded[cam]['pose']['position']
cam_position = [cam_pos_dict['x'], cam_pos_dict['y'], cam_pos_dict['z']]
cam_rot_dict = data_loaded[cam]['pose']['rotation']
cam_orientation = [cam_rot_dict['x'], cam_rot_dict['y'], cam_rot_dict['z']]

# For extrinsic matrix
cam_trans = np.array([cam_pos_dict['x'], cam_pos_dict['y'], cam_pos_dict['z']]).reshape(-1, 1)
cam_axis_vec = np.array([cam_rot_dict['x'], cam_rot_dict['y'], cam_rot_dict['z']])

# Compute T_WC, transform from world to camera
cam_angle = np.linalg.norm(cam_orientation)
cam_axis = cam_orientation/cam_angle
T_WC = tf.translation_matrix(cam_position) @ tf.quaternion_matrix(tf.quaternion_about_axis(cam_angle, cam_axis))
print(tf.quaternion_matrix(tf.quaternion_about_axis(cam_angle, cam_axis)))

# Given fov_x, fov_y, desire output to be h x w resolution
# set fov=fov_y, screen_h = h
# set screen_w = fov_x/fov_y * w

fx = intrinsics[0]
fy = intrinsics[1]

fov_x = 2*np.arctan(video_resolution[1]/(2*fy))*180/np.pi
fov_y = 2*np.arctan(video_resolution[0]/(2*fx))*180/np.pi


cam_fov = fov_x

print('Extracted field of view: ' + f'{cam_fov:f}')


resolution = video_resolution

vis = meshcat.Visualizer()
vis["real_1"].set_object(g.Box([0.1048, 0.1048, 0.1048]),
                       g.MeshLambertMaterial(
                             color=0x00ff00,
                             reflectivity=0.0,
                             transparent=0,
                             opacity=.4))

def get_bundletrack_results():
    """
    State vector is 4 quaternion + 3 xyz position + 3 angular velocity + 3 linear velocity.
    """
    frame_num = len([name for name in os.listdir(DATA_DIR)])
    print("%i frames in total!"%frame_num)
    poses = np.zeros((7, frame_num))
    for frame_id in range(1, frame_num+1):
        pose = np.loadtxt(DATA_DIR + "%04i.txt" % frame_id)#in camera frame
        pose = transform_bundletrack_output(
            pose,
            DATA_DIR,
            ODOM_FILE_PATH,
            cam_trans,
            cam_axis_vec
        )
        pose_quat = trans_mat_to_pos_quat(pose)[3:]
        pose_quat = pose_quat.reshape(1, -1)
        pose_pos = np.array(pose[:3, 3])
        poses[:4, frame_id-1] = pose_quat
        poses[4:7, frame_id-1] = pose_pos
    return poses

DATA_DIR = os.path.join(os.getcwd(), "ob_in_cams", "ob_in_cam_cube_hand/")
BUNDLESDF_DATA = os.path.join(os.getcwd(), "..", "BundleSDF", "data")
ODOM_FILE_PATH = os.path.join(BUNDLESDF_DATA, "cube_hand_toss_60_2", "annotated_poses/")
bundletrack_poses = get_bundletrack_results()
vis["cam"].set_transform(T_WC)
vis["cam_view"].set_transform(T_WC @ tf.translation_matrix([0,0,.05]))

# T_MC, look along z-axis but rotote by 180 degrees
T_MC = tf.translation_matrix([0, 0, -1]) @ tf.rotation_matrix(np.pi, (0,0,1))

T_MW = T_MC @ tf.inverse_matrix(T_WC)

cam = g.PerspectiveCamera(fov=cam_fov, zoom=1, aspect=640/480)
vis["/Cameras/default/rotated"].set_object(cam)

# vis["/Cameras/default/rotated/<object>"].set_property("zoom", 1)
vis["/Cameras/default/rotated/<object>"].set_property("position", [0,0,0])
vis["/Cameras/default"].set_transform(T_MC)

# view in meshcat save to images
# Turn off background, axes, and grid.
vis['/Background'].set_property("visible", False)
vis['/Grid'].set_property("visible", False)
vis['/Axes'].set_property("visible", False)
import cv2
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('overlay_video.mp4', fourcc, 20.0, (640, 480))
# with TemporaryDirectory(prefix="ros-process-") as tmpdir:
    # print(tmpdir)
for i, pose in enumerate(bundletrack_poses.T):
    print('Processing frame ' + f'{i:d}' + ' of ' + f'{bundletrack_poses.shape[1]:d}', end='\r')
    print(i)
    # im = Image.open(io.BytesIO(cam_data[i])).convert('RGB')
    rgb_file_name = os.path.join(BUNDLESDF_DATA, "cube_hand_toss_60_2", "rgb", f"{i+1:04}.png")
    im = cv2.imread(rgb_file_name)
    # im = Image.open(rgb_file_name)
    # im = Image.fromarray(cam_data[i]).convert('RGB')
    T_WA = tf.translation_matrix(pose[4:7]) @ tf.quaternion_matrix(pose[:4])
    vis["real_1"].set_transform(T_MW @ T_WA)
    mesh_im = vis.get_image()
    im.paste(mesh_im, (0,0), mask = mesh_im)
    # im.save(tmpdir + '/' + f'{i:07d}' + '.png', format="png")
    im.imwrite(f"./tmp_{i}.png", im)
    out.write(im)
out.release()
