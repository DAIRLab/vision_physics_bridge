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
import argparse

class FrankaSim:
    def __init__(
        self,
        meshcat,
        position,
        frame_id,
        translation,
        axis_vec,
        show=False,
    ):
        self.meshcat = meshcat
        self.builder = DiagramBuilder()
        self.frame_id = frame_id
        self.plant, self.scene_graph = AddMultibodyPlantSceneGraph(
            self.builder, time_step=0.0
        )
        self.parser = Parser(self.plant)

        # Add robot
        self.X_robo = RigidTransform.Identity()
        self.robo_model_file = FindResourceOrThrow(
            "drake/manipulation/models/franka_description/urdf/panda_arm_hand_wide_finger.urdf"
        )
        self.robo_model = self.parser.AddModelFromFile(self.robo_model_file)
        self.plant.WeldFrames(
            self.plant.world_frame(),
            self.plant.GetFrameByName("panda_link0", self.robo_model),
            self.X_robo,
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
        self.plant.SetPositions(self.plant_context, self.robo_model, position)
        self.plant.get_actuation_input_port().FixValue(self.plant_context, np.zeros(9))
        self.simulator = Simulator(self.diagram, self.context)
        self.simulator.Initialize()
        if show:
            while True:
                self.simulator.AdvanceTo(self.simulator.get_context().get_time() + 2.0)
        self.simulator.AdvanceTo(0.01)

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
        depth_mm[np.isinf(depth_mm)] = 10000.0 # a random large value
        depth_mm_clipped = np.clip(depth_mm, 0, 65535)
        depth_uint16 = depth_mm_clipped.astype(np.uint16)
        img = Image.fromarray(depth_uint16)
        # img.save(os.path.join(NERF_DEPTH_DIR, "%04i_franka.png" % self.frame_id))
        mask = np.isfinite(np.squeeze(depth_image.data)).astype(np.uint8) * 255
        return img, mask

class ObjectPlaybackSim:
    def __init__(
        self,
        meshcat,
        position,
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
        self.model_file = FindResourceOrThrow("drake/../../../../../dair_pll_latest/assets/contactnets_cube_mesh.urdf")
        self.model = self.parser.AddModelFromFile(self.model_file)

        self.plant.WeldFrames(
            self.plant.world_frame(),
            self.plant.GetFrameByName("body", self.model),
            self.X_model,
        )

        # Add robot
        self.X_robo = RigidTransform.Identity()
        self.robo_model_file = FindResourceOrThrow(
            "drake/manipulation/models/franka_description/urdf/panda_arm_hand_wide_finger.urdf"
        )
        self.robo_model = self.parser.AddModelFromFile(self.robo_model_file)
        self.plant.WeldFrames(
            self.plant.world_frame(),
            self.plant.GetFrameByName("panda_link0", self.robo_model),
            self.X_robo,
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
        self.plant.SetPositions(self.plant_context, self.robo_model, position)
        self.plant.get_actuation_input_port().FixValue(self.plant_context, np.zeros(9))
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
        depth_mm[np.isinf(depth_mm)] = 10000.0 # a random large value
        depth_mm_clipped = np.clip(depth_mm, 0, 65535)
        depth_uint16 = depth_mm_clipped.astype(np.uint16)
        img = Image.fromarray(depth_uint16)
        franka = FrankaSim(
            meshcat, positions[frame_id-1], frame_id, translation=cam_trans, axis_vec=cam_axis_vec
        )
        # import sys
        # np.set_printoptions(threshold=sys.maxsize)
        # print(np.array(img))
        # img.save(os.path.join(NERF_DEPTH_DIR, "%04i_total.png" % self.frame_id))
        # img = imageio.imread(os.path.join(NERF_DEPTH_DIR, "%04i.png" % self.frame_id))
        # print(f'Saved depth with format {img.dtype}')
        mask = np.isfinite(np.squeeze(depth_image.data)).astype(np.uint8) * 255
        # imageio.imwrite(os.path.join(NERF_MASK_DIR, "%04i_total.png" % self.frame_id), mask)
        # img = imageio.imread(os.path.join(NERF_MASK_DIR, "%04i.png" % self.frame_id))
        # print(f'Saved mask with format {img.dtype}')
        
        franka_depth, franka_mask = franka.save_depth_and_mask()
        kernel_size = 5
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        dilated_franka_mask = cv2.dilate(franka_mask, kernel, iterations=1)
        visible_cube_mask = mask & (~dilated_franka_mask)
        visible_cube_depth_arr = np.array(img) * (visible_cube_mask.astype(np.array(img).dtype))
        visible_cube_depth = Image.fromarray(visible_cube_depth_arr)
        imageio.imwrite(os.path.join(NERF_MASK_DIR, "%04i.png" % self.frame_id), visible_cube_mask)
        visible_cube_depth.save(os.path.join(NERF_DEPTH_DIR, "%04i.png" % self.frame_id))


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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--toss_id",
        type=int,
        required=True,
    )
    args = parser.parse_args()
    toss_id = args.toss_id
    meshcat = StartMeshcat()
    CAMERA_EXTRINSICS_FILE = "./assets/realsense_pose_cube_old.yaml"
    OUTPUT_POSE_DIR = f"/home/cnets-vision/mengti_ws/BundleSDF/results/old_toss_{toss_id}/ob_in_cam/"
    ODOM_FILE_PATH = f"/home/cnets-vision/mengti_ws/BundleSDF/data/old_toss_{toss_id}/annotated_poses/"
    NERF_DEPTH_DIR = f"./dataset/nerf_old_toss_{toss_id}/depth"
    NERF_MASK_DIR = f"./dataset/nerf_old_toss_{toss_id}/masks"
    POSITION_FILE_PATH = f"./dataset/old_toss_{toss_id}/texts/joint_position.txt"
    cam = 'cam0' # realsense camera name
    with open(CAMERA_EXTRINSICS_FILE, 'r') as stream:
        data_loaded = yaml.safe_load(stream)
    cam_pos_dict = data_loaded[cam]['pose']['position']
    cam_trans = np.array([cam_pos_dict['x'], cam_pos_dict['y'], cam_pos_dict['z']]).reshape(-1, 1)
    cam_rot_dict = data_loaded[cam]['pose']['rotation']
    cam_axis_vec = np.array([cam_rot_dict['x'], cam_rot_dict['y'], cam_rot_dict['z']])
    frame_num = len([name for name in os.listdir(OUTPUT_POSE_DIR)])
    print(f"Total frame is {frame_num}")
    positions = import_data(POSITION_FILE_PATH)
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
            positions[frame_id-1],
            pose,
            frame_id,
            translation=cam_trans,
            axis_vec=cam_axis_vec,
        )

import numpy as np
from stl import mesh
from scipy.spatial import ConvexHull
# pip install scipy numpy-stl

def obj_to_stl(input_obj_file, output_stl_file):
    # Read the obj file
    vertices = []
    faces = []
    with open(input_obj_file, 'r') as file:
        for line in file:
            if line.startswith('v '):
                vertices.append(list(map(float, line.split()[1:])))
            elif line.startswith('f'):
                # Assuming that the .obj mesh is triangulated.
                faces.append(list(map(int, line.split()[1:])))

    vertices = np.array(vertices)
    faces = np.array(faces) - 1  # OBJ files use 1-indexing

    mesh_data = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, face in enumerate(faces):
        for j in range(3):
            mesh_data.vectors[i][j] = vertices[face[j], :]

    mesh_data.save(output_stl_file)

def stl_to_obj(input_stl_file, output_obj_file):
    mesh_data = mesh.Mesh.from_file(input_stl_file)

    with open(output_obj_file, 'w') as file:
        for v in mesh_data.vectors.reshape((-1, 3)):
            file.write(f"v {' '.join(map(str, v))}\n")
        for i in range(0, len(mesh_data.vectors) * 3, 3):
            file.write(f"f {i+1} {i+2} {i+3}\n")

def create_convex_hull(input_obj_file, output_obj_file):
    # Convert obj to stl for easier handling
    obj_to_stl(input_obj_file, 'temp.stl')

    # Load STL and compute convex hull
    input_mesh = mesh.Mesh.from_file('temp.stl')
    points = input_mesh.vectors.reshape((-1, 3))
    hull = ConvexHull(points)

    # Create the convex hull mesh
    convex_mesh = mesh.Mesh(np.zeros(hull.simplices.shape[0], dtype=mesh.Mesh.dtype))
    for i, simplex in enumerate(hull.simplices):
        convex_mesh.vectors[i] = points[simplex]

    # Save convex hull as stl
    convex_mesh.save('temp_hull.stl')

    # Convert the convex hull stl to obj
    stl_to_obj('temp_hull.stl', output_obj_file)
