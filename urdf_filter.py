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
from pydrake.common import FindResourceOrThrow
from pydrake.common.eigen_geometry import AngleAxis
from manipulation.scenarios import AddRgbdSensor
from manipulation.utils import FindResource
import numpy as np
from math_utils import world_to_camera
import rospy
from sensor_msgs.msg import JointState
from file_utils import import_data, filter
from tqdm import tqdm

"""
Generate robot masks for every frame in mask_data.
Note: I installed drake from source through CMake.
"""


class FrankaPlaybackSim:
    def __init__(
        self,
        meshcat,
        position,
        frame_id,
        object_pose,
        gt_pose,
        translation,
        axis_vec,
        urdf_file,
        show=False,
    ):
        self.meshcat = meshcat
        self.builder = DiagramBuilder()
        self.frame_id = frame_id
        self.object_pose = object_pose
        self.gt_pose = gt_pose

        # Add a cube as MultibodyPlant
        self.plant, self.scene_graph = AddMultibodyPlantSceneGraph(
            self.builder, time_step=0.0
        )
        self.X_model = RigidTransform.Identity()
        self.parser = Parser(self.plant)
        self.model_file = FindResourceOrThrow(
            "drake/manipulation/models/franka_description/urdf/panda_arm_hand_wide_finger.urdf"
        )
        self.model = self.parser.AddModelFromFile(self.model_file)

        self.plant.WeldFrames(
            self.plant.world_frame(),
            self.plant.GetFrameByName("panda_link0", self.model),
            self.X_model,
        )

        # Add a box for the camera in the environment.
        camera_pos, angle, axis = self.setup_extrinsic(translation, axis_vec)
        angle_axis = AngleAxis(angle=angle, axis=axis)
        self.X_Camera = RigidTransform(angle_axis, camera_pos)
        self.camera_instance = self.parser.AddModelFromFile(
            FindResource("models/camera_box.sdf")
        )
        self.camera_frame = self.plant.GetFrameByName("base", self.camera_instance)
        self.plant.WeldFrames(
            self.plant.world_frame(), self.camera_frame, self.X_Camera
        )

        # Add a box of bundletrack pose in the environment.
        self.X_box = RigidTransform(self.object_pose)
        self.box_instance = self.parser.AddModelFromFile(FindResource(urdf_file))
        self.box_frame = self.plant.GetFrameByName("body", self.box_instance)
        self.plant.WeldFrames(self.camera_frame, self.box_frame, self.X_box)

        # Add a box of ground-truth pose in the environment.
        self.X_gt_box = RigidTransform(self.gt_pose)
        self.gt_instance = self.parser.AddModelFromFile(
            FindResource(
                "/home/cnets-vision/mengti_ws/robot_filter/assets/contactnets_cube_gt.urdf"
            )
        )
        self.gt_frame = self.plant.GetFrameByName("body", self.gt_instance)
        self.plant.WeldFrames(self.plant.world_frame(), self.gt_frame, self.X_gt_box)
        self.plant.Finalize()

        # Visualize in meshcat
        self.params = MeshcatVisualizerParams()
        self.visualizer = MeshcatVisualizer.AddToBuilder(
            self.builder, self.scene_graph, self.meshcat, self.params
        )
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
        self.camera = AddRgbdSensor(
            self.builder,
            self.scene_graph,
            X_PC=X_PC,
            depth_camera=depth_camera,
            parent_frame_id=self.plant.GetBodyFrameIdOrThrow(
                self.camera_frame.body().index()
            ),
        )
        self.camera.set_name("rgbd_sensor")

        # Export the camera outputs
        self.builder.ExportOutput(self.camera.color_image_output_port(), "color_image")
        self.builder.ExportOutput(
            self.camera.depth_image_32F_output_port(), "depth_image"
        )

        # Setup contexts
        self.diagram = self.build()
        self.context = self.diagram.CreateDefaultContext()
        self.diagram.Publish(self.context)
        self.plant_context = self.plant.GetMyMutableContextFromRoot(self.context)
        self.plant.SetPositions(self.plant_context, self.model, position)
        self.plant.get_actuation_input_port().FixValue(self.plant_context, np.zeros(9))
        self.simulator = Simulator(self.diagram, self.context)
        self.simulator.Initialize()
        if show:
            while True:
                self.simulator.AdvanceTo(self.simulator.get_context().get_time() + 2.0)

        # ROS - temporarily don't need this since we are not processing in real-time
        # self.diagram = self.build()
        # self.simulator = Simulator(self.diagram)
        # self.simulator.set_target_realtime_rate(1.0)
        # self.simulator.set_publish_every_time_step(False)
        # self.context = self.simulator.get_mutable_context()
        # self.state = self.context.get_mutable_continuous_state_vector()
        # self.state.SetFromVector(np.zeros(9*2))
        # self.simulator.Initialize()

    def setup_extrinsic(self, translation, axis_vec):
        angle = np.linalg.norm(axis_vec)
        axis = axis_vec / angle
        return translation, angle, axis

    def build(self):
        diagram = self.builder.Build()
        diagram.set_name("depth_camera_demo_system")
        return diagram

    def visualize(self):
        def callback(msg):
            global q, v, f
            q = np.array(msg.position)
            v = np.array(msg.velocity)
            f = np.array(msg.effort)

        rospy.init_node("listener", anonymous=True)
        rospy.Subscriber("/joint_states", JointState, callback=callback)
        rate = rospy.Rate(4)
        rate.sleep()
        while not rospy.is_shutdown():
            self.state = self.context.get_mutable_continuous_state_vector()
            self.state.SetFromVector(np.append(q, v))
            self.plot_camera_images()
            self.simulator.Initialize()
            rate.sleep()

    def plot_camera_images(self, real_depth_dir, simulated_depth_dir):
        color_image = self.diagram.GetOutputPort("color_image").Eval(self.context)
        depth_image = self.diagram.GetOutputPort("depth_image").Eval(self.context)

        # Plot the two images.
        plt.subplot(121)
        # plt.imshow(color_image.data)
        # plt.title('Color image')
        real = np.loadtxt(real_depth_dir)
        _min = 0
        _max = 1
        # plt.imshow(real*0.001, vmin = _min, vmax = _max)
        plt.imshow(real, vmin=_min, vmax=_max)
        plt.plot(320, 250, "ro")
        plt.colorbar()
        plt.title("Real image")
        plt.subplot(122)
        plt.imshow(np.squeeze(depth_image.data), vmin=_min, vmax=_max)  # (480.640,1)
        plt.plot(320, 250, "ro")
        plt.colorbar()
        plt.title("Depth image")
        np.savetxt(simulated_depth_dir, depth_image.data[:, :, 0])
        # pdb.set_trace()
        # plt.show()


def dilate(frame_id, mask_image_dir, dilated_mask_dir):
    """
    Cut mask out of image with certain pixel margin since there is some small leftovers of the robot after applying urdf filter.
    """
    # Load image and mask
    # rgb_image_file = "./rgb_data/%04i.png" % frame_id
    # mask_image_file = "./mask_data/%04i.png" % frame_id
    # dilated_mask_file = "./dilated_mask_data/%04i.png" % frame_id
    # image = cv2.imread(rgb_image_file)
    dilated_mask_file = dilated_mask_dir % frame_id
    mask_image_file = mask_image_dir % frame_id
    mask = cv2.imread(mask_image_file)
    # Create structuring element, dilate and bitwise-and
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    print("Finished structured element.")
    dilate = cv2.dilate(mask, kernel, iterations=3)
    print("Finished dilate.")
    # result = cv2.bitwise_and(image, dilate)
    # print("Finished bitwise.")

    # cv2.imshow("dilate", dilate)
    # cv2.imshow('result', result)
    # cv2.waitKey()
    im = Image.fromarray(dilate)  ##TODO
    # im = Image.fromarray(mask)
    if im.mode != "L":
        im = im.convert("L")
    print("Saving mask frame ", frame_id)
    im.save(dilated_mask_file)
    # cv2.imwrite(dilated_mask_file, dilate)


def run_urdf_filter(
    meshcat,
    frame_id,
    positions,
    mask_image_dir,
    simulated_depth_dir,
    real_depth_dir,
    cam_translation,
    cam_axis_vec,
):
    mask_image_file = mask_image_dir % frame_id
    simulated_depth_file = simulated_depth_dir % frame_id
    real_depth_file = real_depth_dir % frame_id
    filtered_depth_file = "./filtered_data/depth_without_robot_frame%04i.png" % frame_id
    filtered_rgb_file = "./filtered_data/rgb_without_robot_frame%04i.png" % frame_id
    system = FrankaPlaybackSim(
        meshcat,
        positions[frame_id - 1],
        frame_id,
        # bundletrack_pose,
        # gt_pose,
        cam_translation,
        cam_axis_vec,
        show=False,
    )
    system.plot_camera_images(real_depth_file, simulated_depth_file)
    simulated_image = np.loadtxt(simulated_depth_file)
    real_image = np.loadtxt(real_depth_file)
    # Need to multiply real depth images by 1000 since simulated depth image uses mm as unit
    mask = filter(real_image * 1000, simulated_image)
    im = Image.fromarray(mask)
    if im.mode != "L":
        im = im.convert("L")
    print("Saving mask frame ", frame_id)
    im.save(mask_image_file)
    # pdb.set_trace()
    # plt.show()
    # generate_depth_img_without_robot(real_depth_file, mask_image_file, filtered_depth_file)
    # generate_rgb_image_without_robot(rgb_image_file, mask_image_file, filtered_rgb_file) #optional, seems the point cloud looks fine with the unfiltered rgb data


if __name__ == "__main__":
    meshcat = StartMeshcat()
    ROOT_DIR = "./dataset/new_split/1/"
    POSITION_FILE_PATH = ROOT_DIR + "texts/joint_position.txt"
    OUTPUT_POSE_DIR = "/home/cnets-vision/mengti_ws/results/poses_1/"
    GT_POSE_DIR = ROOT_DIR + "tagslam_poses/"
    # OUTPUT_POSE_DIR = "/home/cnets-vision/mengti_ws/poses/"
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
    positions = import_data(POSITION_FILE_PATH)
    for frame_id in range(1, 2):
        bundletrack_pose = np.loadtxt(OUTPUT_POSE_DIR + "%04i.txt" % frame_id)
        # gt_pose = np.loadtxt(GT_POSE_DIR + "%04i.txt" % frame_id)
        # bundletrack_pose = np.array(
        #     [[1, 0, 0, 0.2], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        # )
        gt_pose = np.array([[1, 0, 0, 0.3], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        # bundletrack_pose = world_to_camera(gt_pose)
        # gt_pose = camera_to_world(bundletrack_pose)
        system = FrankaPlaybackSim(
            meshcat,
            positions[frame_id],
            frame_id,
            bundletrack_pose,
            gt_pose,
            translation=CAMERA_CONFIG["new"]["translation"],
            axis_vec=CAMERA_CONFIG["new"]["axis_vec"],
            urdf_file="/home/cnets-vision/mengti_ws/robot_filter/assets/contactnets_cube_new.urdf",
            show=True,
        )
