import numpy as np
import os
import time

import rosbag
import yaml

import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf

from PIL import Image
import io
import rospy

import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pdb
from PIL import Image
import imageio
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder
from pydrake.math import RigidTransform
from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    AngleAxis,
    CameraInfo,
    ClippingRange,
    DepthRange,
    DepthRenderCamera,
    DiagramBuilder,
    MeshcatVisualizer,
    MeshcatVisualizerParams,
    Parser,
    RigidTransform,
    RenderCameraCore,
    Simulator,
    StartMeshcat,
)
from manipulation.scenarios import AddRgbdSensor
from manipulation.utils import FindResource
from pydrake.common import FindResourceOrThrow
from math_utils import (
    quaternion_to_rotation_matrix,
    rotation_matrix_to_quaternion,
    transform_bundletrack_output_to_world,
)

from cv_bridge import CvBridge
from urdf_filter import FrankaPlaybackSim
from pydrake.all import StartMeshcat
from file_utils import import_data

###################NEW DATASET################
# start time
start_time = rospy.rostime.Time(secs=1667330342, nsecs=959312)  # toss 1
# start_time = rospy.rostime.Time(secs=1667330357, nsecs=453744)  # toss 2
# start_time = rospy.rostime.Time(secs=1667330379, nsecs=657584)  # toss 3
# start_time = rospy.rostime.Time(secs=1667330401, nsecs=359995)  # toss 4
# start_time = rospy.rostime.Time(secs=1667330422, nsecs=301875)  # toss 5
# start_time = rospy.rostime.Time(secs=1667330442, nsecs=284764)  # toss 6
# start_time = rospy.rostime.Time(secs=1667330465, nsecs=216068)  # toss 7
# start_time = rospy.rostime.Time(secs=1667330490, nsecs=318756)  # toss 8

# end time
end_time = rospy.rostime.Time(secs=1667330357, nsecs=453744)  # toss 1
# end_time = rospy.rostime.Time(secs=1667330379, nsecs=657584)  # toss 2
# end_time = rospy.rostime.Time(secs=1667330401, nsecs=359995)  # toss 3
# end_time = rospy.rostime.Time(secs=1667330422, nsecs=301875)  # toss 4
# end_time = rospy.rostime.Time(secs=1667330442, nsecs=284764)  # toss 5
# end_time = rospy.rostime.Time(secs=1667330465, nsecs=216068)  # toss 6
# end_time = rospy.rostime.Time(secs=1667330490, nsecs=318756)  # toss 7
# end_time = rospy.rostime.Time(secs=1667330509, nsecs=187270)  # toss 8
odom_bag_file = "./odom_19.bag"
cam_bag_file = "./raw_19.bag"
cam_topic = "/camera/color/image_raw"
ROOT_DIR = "./dataset/new_split/1/"
POSITION_FILE_PATH = ROOT_DIR + "texts/joint_position.txt"
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
meshcat = StartMeshcat()
positions = import_data(POSITION_FILE_PATH)


def checkStarttime(time1, time2):
    return time1.secs < time2.secs or (
        time1.secs == time2.secs and time1.nsecs < time2.nsecs
    )


def checkEndtime(time1, time2):
    return time1.secs == time2.secs


bag = rosbag.Bag(odom_bag_file)
# get summary info from rosbag as a dictionary
info = yaml.load(bag._get_yaml_info(), Loader=yaml.FullLoader)
TOPIC_STRING_1 = "/tagslam/odom/body_box"

# extract metadata from cube and board topics
elbow_1_topic = [topic for topic in info["topics"] if topic["topic"] == TOPIC_STRING_1][
    0
]

num_msg = elbow_1_topic["messages"]


def extract_poses(messages, start_time, end_time):
    # poses = np.zeros((7, len(messages)))
    # poses = np.zeros((7, 1))
    poses = []
    global odom_timestamps
    odom_timestamps = []
    for i, data in enumerate(messages):
        (_, msg, _) = data
        if checkStarttime(msg.header.stamp, start_time):
            continue
        if checkEndtime(msg.header.stamp, end_time):
            break
        Q = np.zeros((4, 1))
        Q[0] = msg.pose.pose.orientation.x
        Q[1] = msg.pose.pose.orientation.y
        Q[2] = msg.pose.pose.orientation.z
        Q[3] = msg.pose.pose.orientation.w
        rotation_matrix = quaternion_to_rotation_matrix(Q)[:, :, 0]
        position = msg.pose.pose.position
        translation = np.array([[position.x], [position.y], [position.z]])
        result = np.vstack(
            (np.hstack((rotation_matrix, translation)), np.array([0, 0, 0, 1]))
        )
        poses.append(result)
        odom_timestamps.append(msg.header.stamp)
    return np.array(poses)


def get_bundletrack_results():
    """
    State vector is 4 quaternion + 3 xyz position + 3 angular velocity + 3 linear velocity.
    """
    frame_num = len([name for name in os.listdir(DATA_DIR)])
    print("%i frames in total!" % frame_num)
    poses = np.zeros((7, frame_num))
    for frame_id in range(1, frame_num + 1):
        pose = np.loadtxt(DATA_DIR + "%04i.txt" % frame_id)  # in camera frame
        # For camera extrinsics
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
        pose = transform_bundletrack_output_to_world(
            pose,
            CAMERA_CONFIG["new"]["translation"],
            CAMERA_CONFIG["new"]["axis_vec"],
            DATA_DIR,
            ODOM_FILE_PATH,
        )
        pose_quat = rotation_matrix_to_quaternion(pose)
        # r = R.from_matrix(pose[:3, :3])
        # pose_quat = quaternion_from_matrix(pose)
        # pose_quat_ = r.as_quat()
        # print(f'pose_quat: {pose_quat}, pose_quat_: {pose_quat_}')
        pose_quat = pose_quat.reshape(1, -1)
        pose_pos = np.array(pose[:3, 3])
        poses[:4, frame_id - 1] = pose_quat
        poses[4:7, frame_id - 1] = pose_pos
    return poses


DATA_DIR = "/home/cnets-vision/mengti_ws/results/poses_1/"
ODOM_FILE_PATH = "/home/cnets-vision/mengti_ws/BundleTrack/Data/YCBINEOAT/contact_nets_new_split/1/annotated_poses/"

bundletrack_poses = get_bundletrack_results()
odom_msg = extract_poses(
    list(bag.read_messages(topics=[TOPIC_STRING_1])), start_time, end_time
)
# print(odom_msg.shape)
# print("odom_msg: ", odom_msg.shape)
# print(odom_timestamps)
# if (bundletrack_poses.shape[1]>elbow_1.shape[1]):
#     bundletrack_poses = bundletrack_poses[:, :elbow_1.shape[1]]
# print("bundletrack: ", bundletrack_poses.shape)
bag.close()


def get_most_recent_odom_idx(tstamp, matched_tstamps):
    """
    Get the most recent odometry timestamp that is before tstamp.
    tstamp: timestamps of raw images
    """
    most_recent = None
    idx = None
    for i, odom_tstamp in enumerate(odom_timestamps):
        if odom_tstamp.secs < tstamp.secs or (
            odom_tstamp.secs == tstamp.secs and odom_tstamp.nsecs < tstamp.nsecs
        ):
            if (
                most_recent == None
                or most_recent.secs < odom_tstamp.secs
                or (
                    most_recent.secs == odom_tstamp.secs
                    and most_recent.nsecs < odom_tstamp.nsecs
                )
            ):
                if odom_tstamp not in matched_tstamps:
                    most_recent = odom_tstamp
                    idx = i
                    matched_tstamps.add(odom_tstamp)
    # print(idx)
    return idx


# Load video
raw_bag = rosbag.Bag(cam_bag_file)
cam_messages = list(raw_bag.read_messages(topics=[cam_topic]))
matched_tstamps = set()
num_cam_msg = len(cam_messages)

# extract camera
t_cam = np.zeros(len(cam_messages))
cam_data = []
bridge = CvBridge()
for i, data in enumerate(cam_messages):
    (_, msg, _) = data
    tstamp = msg.header.stamp
    if checkStarttime(tstamp, start_time):
        continue
    if checkEndtime(tstamp, end_time):
        break
    odom_idx = get_most_recent_odom_idx(tstamp, matched_tstamps)
    if odom_idx != None:
        odom_tstamp = odom_timestamps[odom_idx]
        # print('odom: ', odom_tstamp.secs, odom_tstamp.nsecs)
        # print('video: ', tstamp.secs, tstamp.nsecs)
    t_cam[i] = tstamp.secs + tstamp.nsecs * 1e-9
    # cam_data.append(msg.data)
    cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
    cam_data.append([cv_img, odom_idx, tstamp])  # record tstamps for debugging purposes
print(len(cam_data))
raw_bag.close()
frame_num = len([name for name in os.listdir(DATA_DIR)])

#### Drake Simulator
def setup_extrinsic(translation, axis_vec):
    angle = np.linalg.norm(axis_vec)
    axis = axis_vec / angle
    return translation, angle, axis


builder = DiagramBuilder()

# Add a cube as MultibodyPlant
plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
X_model = RigidTransform.Identity()
parser = Parser(plant)
model_file = FindResourceOrThrow(
    "drake/manipulation/models/franka_description/urdf/panda_arm_hand_wide_finger.urdf"
)
model = parser.AddModelFromFile(model_file)
plant.WeldFrames(
    plant.world_frame(),
    plant.GetFrameByName("panda_link0", model),
    X_model,
)

# Add a box for the camera in the environment.
camera_pos, angle, axis = setup_extrinsic(
    CAMERA_CONFIG["new"]["translation"], CAMERA_CONFIG["new"]["axis_vec"]
)
angle_axis = AngleAxis(angle=angle, axis=axis)
X_Camera = RigidTransform(angle_axis, camera_pos)
camera_instance = parser.AddModelFromFile(FindResource("models/camera_box.sdf"))
camera_frame = plant.GetFrameByName("base", camera_instance)
plant.WeldFrames(plant.world_frame(), camera_frame, X_Camera)

# Add a box of bundletrack pose in the environment.
# X_box = RigidTransform.Identity()
box_instance = parser.AddModelFromFile(
    FindResource(
        "/home/cnets-vision/mengti_ws/robot_filter/assets/contactnets_cube_new.urdf"
    )
)
box_frame = plant.GetFrameByName("body", box_instance)
# plant.WeldFrames(camera_frame, box_frame, X_box) #in camera frame
# plant.WeldFrames(plant.world_frame(), box_frame, X_box)  # in world frame

# Add a box of ground-truth pose in the environment.
# X_gt_box = RigidTransform.Identity()
gt_instance = parser.AddModelFromFile(
    FindResource(
        "/home/cnets-vision/mengti_ws/robot_filter/assets/contactnets_cube_gt.urdf"
    )
)
gt_frame = plant.GetFrameByName("body_gt", gt_instance)
# plant.WeldFrames(plant.world_frame(), gt_frame, X_gt_box)
plant.Finalize()

# Visualize in meshcat
params = MeshcatVisualizerParams()
visualizer = MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat, params)
X_PC = RigidTransform()
renderer = "my_renderer"
intrinsic = CameraInfo(
    640,
    480,
    380.2484436035156,
    379.8265380859375,
    314.2138977050781,
    240.59800720214844,
)
depth_camera = DepthRenderCamera(
    RenderCameraCore(
        renderer, intrinsic, ClippingRange(near=0.1, far=10.0), RigidTransform()
    ),
    DepthRange(0.1, 10.0),
)
camera = AddRgbdSensor(
    builder,
    scene_graph,
    X_PC=X_PC,
    depth_camera=depth_camera,
    parent_frame_id=plant.GetBodyFrameIdOrThrow(camera_frame.body().index()),
)
camera.set_name("rgbd_sensor")

# Export the camera outputs
builder.ExportOutput(camera.color_image_output_port(), "color_image")
builder.ExportOutput(camera.depth_image_32F_output_port(), "depth_image")

# Setup contexts
diagram = builder.Build()
diagram.set_name("depth_camera_demo_system")
context = diagram.CreateDefaultContext()
diagram.Publish(context)
plant_context = plant.GetMyMutableContextFromRoot(context)
# plant.SetPositions(plant_context, model, position)
plant.get_actuation_input_port().FixValue(plant_context, np.zeros(9))
simulator = Simulator(diagram, context)
simulator.Initialize()
# simulator.set_target_realtime_rate(0.5)

for i in range(1, frame_num):
    bundletrack_pose = np.loadtxt(DATA_DIR + "%04i.txt" % i)
    pose = transform_bundletrack_output_to_world(
        bundletrack_pose,
        CAMERA_CONFIG["new"]["translation"],
        CAMERA_CONFIG["new"]["axis_vec"],
        DATA_DIR,
        ODOM_FILE_PATH,
    )
    # gt_pose = np.loadtxt(GT_POSE_DIR + "%04i.txt" % frame_id)
    # bundletrack_pose = np.array(
    #     [[1, 0, 0, 0.2], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    # )
    # gt_pose = np.array([[1, 0, 0, 0.3], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    # bundletrack_pose = world_to_camera(gt_pose)
    # gt_pose = camera_to_world(bundletrack_pose)
    odom_idx = cam_data[i][1]
    if odom_idx != None:
        gt_pose = odom_msg[odom_idx]
        # pose_2 = np.array([0.5276159458071291, -0.5565072250934956, -0.46577238162553075, 0.44156223872028155, 0.7517670087101709, 0.2749021984583954, 0.018160298466467384]).T## For experiment purposes

        odom_ = odom_timestamps[odom_idx]
        tstamp = cam_data[i][2]
        print(
            f"bt frame: {i}, odom idx: {odom_idx}, odom timestamp: {odom_.secs}, {odom_.nsecs}, raw image timestamp:, {tstamp.secs}, {tstamp.nsecs}"
        )
        # print(pose, gt_pose)
        # system = FrankaPlaybackSim(
        #     meshcat,
        #     positions[i],
        #     i,
        #     pose,
        #     gt_pose,
        #     translation=CAMERA_CONFIG["new"]["translation"],
        #     axis_vec=CAMERA_CONFIG["new"]["axis_vec"],
        #     show=False,
        # )
        plant.SetPositions(plant_context, model, positions[i])
        body_cube = plant.GetBodyByName("body")
        body_gt = plant.GetBodyByName("body_gt")
        pose_ = RigidTransform(pose)
        gt_pose_ = RigidTransform(gt_pose)
        plant.SetFreeBodyPose(context=plant_context, body=body_cube, X_WB=pose_)
        plant.SetFreeBodyPose(context=plant_context, body=body_gt, X_WB=gt_pose_)
        # time.sleep(2.0)
        # simulator.AdvanceTo(simulator.get_context().get_time() + 1.0)
