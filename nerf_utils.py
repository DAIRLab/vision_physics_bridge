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
from math_utils import transform_bundletrack_output
import rospy
from sensor_msgs.msg import JointState
from file_utils import import_data, filter
from tqdm import tqdm
import os
import yaml

class ObjectPlaybackSim:
    def __init__(
        self,
        meshcat,
        pose,
        frame_id,
        translation,
        axis_vec,
        show=False,
    ):
        self.meshcat = meshcat
        self.builder = DiagramBuilder()
        self.frame_id = frame_id
        # Add a cube as MultibodyPlant
        self.plant, self.scene_graph = AddMultibodyPlantSceneGraph(
            self.builder, time_step=0.0
        )
        self.X_model = RigidTransform(pose)
        self.parser = Parser(self.plant)
        self.model_file = FindResourceOrThrow("drake/../../../../../../../../dair_pll_latest/assets/contactnets_cube_mesh.urdf")
        self.model = self.parser.AddModelFromFile(self.model_file)

        self.plant.WeldFrames(
            self.plant.world_frame(),
            self.plant.GetFrameByName("body", self.model),
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
        # self.diagram.Publish(self.context)
        self.plant_context = self.plant.GetMyMutableContextFromRoot(self.context)
        self.simulator = Simulator(self.diagram, self.context)
        self.simulator.Initialize()
        if show:
            while True:
                self.simulator.AdvanceTo(self.simulator.get_context().get_time() + 2.0)
        self.simulator.AdvanceTo(0.01)
        # self.plot_camera_images()
        self.save_depth_and_mask()

    def setup_extrinsic(self, translation, axis_vec):
        angle = np.linalg.norm(axis_vec)
        axis = axis_vec / angle
        return translation, angle, axis

    def build(self):
        diagram = self.builder.Build()
        diagram.set_name("depth_camera_demo_system")
        return diagram
    
    def save_depth_and_mask(self):
        depth_image = self.diagram.GetOutputPort("depth_image").Eval(self.context)
        depth_mm = depth_image.data[:, :, 0] * 1000
        depth_mm_clipped = np.clip(depth_mm, 0, 65535)
        depth_uint16 = depth_mm_clipped.astype(np.uint16)
        img = Image.fromarray(depth_uint16)
        img.save(os.path.join(NERF_DEPTH_DIR, "%04i.png" % self.frame_id))  
        # img = imageio.imread(os.path.join(NERF_DEPTH_DIR, "%04i.png" % self.frame_id))
        # print(f'Saved depth with format {img.dtype}')
        mask = np.isfinite(np.squeeze(depth_image.data)).astype(np.uint8) * 255
        imageio.imwrite(os.path.join(NERF_MASK_DIR, "%04i.png" % self.frame_id), mask)
        # img = imageio.imread(os.path.join(NERF_MASK_DIR, "%04i.png" % self.frame_id))
        # print(f'Saved mask with format {img.dtype}')

    def plot_camera_images(self):
        color_image = self.diagram.GetOutputPort("color_image").Eval(self.context)
        depth_image = self.diagram.GetOutputPort("depth_image").Eval(self.context)
        # Plot the two images.
        plt.figure(figsize=(10,5))
        plt.subplot(121)
        plt.imshow(color_image.data)
        plt.title('Color image')
        _min = 0
        _max = 1
        plt.subplot(122)
        plt.imshow(np.squeeze(depth_image.data), vmin=_min, vmax=_max)  # (480.640,1)
        plt.plot(400, 320, "ro")
        plt.title("Depth image")
        plt.tight_layout()
        plt.savefig('./cube_drake_visualizer.png')
        # plt.show()
    
if __name__ == "__main__":
    toss_id = 1
    meshcat = StartMeshcat()
    CAMERA_EXTRINSICS_FILE = "./assets/realsense_pose_cube_old.yaml"
    OUTPUT_POSE_DIR = f"/home/cnets-vision/mengti_ws/BundleSDF/results/old_toss_{toss_id}/ob_in_cam/"
    ODOM_FILE_PATH = f"/home/cnets-vision/mengti_ws/BundleSDF/data/old_toss_{toss_id}/annotated_poses/"
    NERF_DEPTH_DIR = "./dataset/nerf/depth"
    NERF_MASK_DIR = "./dataset/nerf/masks"
    cam = 'cam0' # realsense camera name
    with open(CAMERA_EXTRINSICS_FILE, 'r') as stream:
        data_loaded = yaml.safe_load(stream)
    cam_pos_dict = data_loaded[cam]['pose']['position']
    cam_trans = np.array([cam_pos_dict['x'], cam_pos_dict['y'], cam_pos_dict['z']]).reshape(-1, 1)
    cam_rot_dict = data_loaded[cam]['pose']['rotation']
    cam_axis_vec = np.array([cam_rot_dict['x'], cam_rot_dict['y'], cam_rot_dict['z']])
    frame_num = len([name for name in os.listdir(OUTPUT_POSE_DIR)])
    print(f"Total frame is {frame_num}")
    for frame_id in range(1, frame_num+1):
        print(f"Processing frame {frame_id}")
        bundletrack_pose = np.loadtxt(OUTPUT_POSE_DIR + "%04i.txt" % frame_id)
        pose = transform_bundletrack_output(
            bundletrack_pose,
            OUTPUT_POSE_DIR,
            ODOM_FILE_PATH,
            cam_trans, 
            cam_axis_vec,
            to_world=True
        ) # cam frame
        system = ObjectPlaybackSim(
            meshcat,
            pose,
            frame_id,
            translation=cam_trans,
            axis_vec=cam_axis_vec,
        )
