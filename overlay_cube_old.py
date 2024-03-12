### Remove the dependency of Jupyter notebook. 

import numpy as np
import os
import time
from file_utils import load_toss_time_from_yaml

import rosbag
import yaml
from tqdm import tqdm

import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf

from tempfile import TemporaryDirectory

from PIL import Image
import io
import rospy 
from math_utils import transform_bundletrack_origin_to_tagslam_origin
# video resolution
video_resolution=[640, 480]

#
## parameters:
#
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('toss_id', type=int)
parser.add_argument('trial_id', type=int)
args = parser.parse_args()
toss_id = args.toss_id
trial_id = args.trial_id
cam = 'cam0'
video_name = f'cube_{toss_id}_{trial_id}'
odom_bag_file = '/home/cnets-vision/mengti_ws/robot_filter/rosbags/odom_10.bag'
cam_bag_file = '/home/cnets-vision/mengti_ws/robot_filter/rosbags/raw_10.bag'
cam_poses_file = './assets/realsense_pose_cube.yaml'
output_file = './videos/' + video_name + '.mp4'
if not os.path.exists(os.path.dirname(output_file)):
    os.makedirs(os.path.dirname(output_file))
start_time = None
end_time = None
yaml_path = './assets/config.yaml'
toss_type = 'cube'
start_time = load_toss_time_from_yaml(toss_type, toss_id, 'start_time')
end_time = load_toss_time_from_yaml(toss_type, toss_id, 'end_time')

cam_topic = '/camera/color/image_raw'

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
# vis = meshcat.Visualizer(zmq_url="tcp://127.0.0.1:6000") # can specify this after the first run but the frame will mismatch

##################
# Create the cubes. Set the opacity of the "real" to > 0 if you want to see it, for invisible
vis["real_1"].set_object(g.Box([0.1048, 0.1048, 0.1048]),
                       g.MeshLambertMaterial(
                             color=0x00ff00,
                             reflectivity=0.0,
                             transparent=0,
                             opacity=.4))

# vis["real_1"].set_object(g.Box([0.096, 0.061, 0.096]),
#                        g.MeshLambertMaterial(
#                              color=0x00ff00,
#                              reflectivity=0.0,
#                              transparent=0,
#                              opacity=.4))

vis["real_2"].set_object(g.Box([0.1048, 0.1048, 0.1048]),
                       g.MeshLambertMaterial(
                             color=0xff22dd,
                             reflectivity=0.0,
                             transparent=0,
                             opacity=.4))


########################
from math_utils import trans_mat_to_pos_quat
# import tf.transformations as tr
import transformations as tr

bag = rosbag.Bag(odom_bag_file)

# get summary info from rosbag as a dictionary
info = yaml.load(bag._get_yaml_info(), Loader=yaml.FullLoader)
TOPIC_STRING_1 = '/tagslam/odom/body_cube'

# extract metadata from cube and board topics
elbow_1_topic = [topic for topic in info['topics'] if topic['topic'] == TOPIC_STRING_1][0]

num_msg = elbow_1_topic['messages']

def extract_times(messages):
    t_ros = np.zeros(len(messages))
    for i, data in enumerate(list(messages)):
        (_, msg, _) = data
        tstamp = msg.header.stamp
        t_ros[i] = tstamp.secs + tstamp.nsecs * 1e-9

    return t_ros


def extract_poses(messages, start_time, end_time):
    # poses = np.zeros((7, len(messages)))
    poses = np.zeros((7,1))
    for i, data in enumerate(messages):
        (_, msg, _) = data
        if start_time <= msg.header.stamp < end_time:
            pose = msg.pose.pose
            pose_pos = np.asarray([pose.position.x, pose.position.y, pose.position.z])
            pose_quat = np.asarray([pose.orientation.w, pose.orientation.x, pose.orientation.y,
                                    pose.orientation.z])
            # print(np.hstack((pose_quat, pose_pos)).T.shape)
            poses = np.hstack((poses, np.hstack((pose_quat, pose_pos)).T.reshape((-1,1))))
            # poses[:4, i] = pose_quat
            # poses[4:7, i] = pose_pos
    return poses[:,1:]

def get_bundletrack_results():
    """
    State vector is 4 quaternion + 3 xyz position + 3 angular velocity + 3 linear velocity.
    """
    frame_num = len([name for name in os.listdir(DATA_DIR)])
    print("%i frames in total!"%frame_num)
    poses = np.zeros((7, frame_num))
    for frame_id in range(1, frame_num+1):
        pose = np.loadtxt(DATA_DIR + "%04i.txt" % frame_id)#in camera frame
        pose = transform_bundletrack_origin_to_tagslam_origin(
            pose,
            DATA_DIR,
            ODOM_FILE_PATH,
            cam_trans,
            cam_axis_vec,
            to_world=True
        )
        pose_quat = tr.quaternion_from_matrix(pose) #trans_mat_to_pos_quat(pose)[3:]
        pose_quat = pose_quat.reshape(1, -1)
        pose_pos = np.array(pose[:3, 3])
        poses[:4, frame_id-1] = pose_quat
        poses[4:7, frame_id-1] = pose_pos
    return poses

DATA_DIR = f"/home/cnets-vision/mengti_ws/BundleSDF/results/find_good_case/cube_{toss_id}_{trial_id}/ob_in_cam/"
ODOM_FILE_PATH = f"/home/cnets-vision/mengti_ws/BundleSDF/data/cube_{toss_id}/annotated_poses/"

bundletrack_poses = get_bundletrack_results()
# t_ros = extract_times(list(bag.read_messages(topics=[TOPIC_STRING_1])))
elbow_1 = extract_poses(list(bag.read_messages(topics=[TOPIC_STRING_1])), start_time, end_time)
# t_ros_2 = extract_times(list(bag.read_messages(topics=[TOPIC_STRING_2])))
# elbow_2 = extract_poses(list(bag.read_messages(topics=[TOPIC_STRING_2])))
print("elbow1: ", elbow_1.shape)
if (bundletrack_poses.shape[1]>elbow_1.shape[1]):
    bundletrack_poses = bundletrack_poses[:, :elbow_1.shape[1]]
print("bundletrack: ", bundletrack_poses.shape)
bag.close()

################
from cv_bridge import CvBridge

# Load video
raw_bag = rosbag.Bag(cam_bag_file)
cam_messages= list(raw_bag.read_messages(topics=[cam_topic]))

num_cam_msg = len(cam_messages)
# print(num_cam_msg)
# extract camera
t_cam = np.zeros(len(cam_messages))
cam_data = []
bridge = CvBridge()
for i, data in enumerate(cam_messages):
    (_, msg, _) = data
    tstamp = msg.header.stamp
    if start_time <= tstamp < end_time:
        t_cam[i] = tstamp.secs + tstamp.nsecs * 1e-9
        # cam_data.append(msg.data)
        cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        cam_data.append(cv_img)
print(len(cam_data))

base_url = "http://127.0.0.1"

meshcat_url = base_url + ":" + vis.url().split(":")[-1]

# ''' Create precisely-sized iframe with meshcat view; put this in its own Jupyter cell. '''
### For Jypyter notebook environment. 
# from IPython.display import HTML
# frame_html = """
# <div style="height: {height}px; width: {width}px; overflow-x: visible; overflow-y: visible; resize: none">
#     <iframe src="{url}" style="width: 100%; height: 100%; border: none"></iframe>
# </div>
# """.format(url=meshcat_url, width=resolution[0], height=resolution[1])
# HTML(frame_html)

### For python script. 
### Need x server to run. Either run locally or run remotely with x forward configured. 
from selenium import webdriver

options = webdriver.ChromeOptions()
options.add_argument('--headless')
driver = webdriver.Chrome(options=options)

driver.set_window_size(resolution[0], resolution[1])  # Set the desired window size
driver.get(meshcat_url)

########################
# Frames
# (W) World
# (M) Meshcat
# (C) Camera
# (A) link_1
# (B) link_2

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


###########################
# view in meshcat save to images
# Turn off background, axes, and grid.
vis['/Background'].set_property("visible", False)
vis['/Grid'].set_property("visible", False)
vis['/Axes'].set_property("visible", False)

with TemporaryDirectory(prefix="ros-process-") as tmpdir:
    print(tmpdir)
    for i, pose in enumerate(tqdm(bundletrack_poses.T)):
        # print('Processing frame ' + f'{i:d}' + ' of ' + f'{bundletrack_poses.shape[1]:d}', end='\r')
        # print(i)
        # im = Image.open(io.BytesIO(cam_data[i])).convert('RGB')
        im = Image.fromarray(cam_data[i]).convert('RGB')
        T_WA = tf.translation_matrix(pose[4:7]) @ tf.quaternion_matrix(pose[:4])
        vis["real_1"].set_transform(T_MW @ T_WA)

        # pose_2 = elbow_1[:,i]
        # T_WB = tf.translation_matrix(pose_2[4:7]) @ tf.quaternion_matrix(pose_2[:4])
        # vis["real_2"].set_transform(T_MW @ T_WB)

        mesh_im = vis.get_image()
        # print(mesh_im.size)
        # mesh_im.show()
        im.paste(mesh_im, (0,0), mask = mesh_im)
        # im.show()
        # break
        im.save(tmpdir + '/' + f'{i:07d}' + '.png', format="png")
    os.system('ffmpeg -y -r 150 -i ' + tmpdir + '/%07d.png -vcodec libx264 -preset slow -crf 18 ' + output_file)

vis.delete()
driver.quit() # if not exited properly, orphan chrome processes will remain active. 