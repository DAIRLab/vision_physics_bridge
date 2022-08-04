import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pdb
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import MultibodyPlant, AddMultibodyPlantSceneGraph
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.rendering import MultibodyPositionToGeometryPose
from pydrake.systems.primitives import TrajectorySource
from pydrake.math import RigidTransform
from pydrake.geometry import SceneGraph, DrakeVisualizer
from pydrake.trajectories import PiecewisePolynomial
from pydrake.all import (AbstractValue, AddMultibodyPlantSceneGraph, AngleAxis,
                         BaseField, ConstantValueSource, CsdpSolver,
                         DepthImageToPointCloud, DiagramBuilder,
                         DifferentialInverseKinematicsIntegrator,
                         DifferentialInverseKinematicsParameters, EventStatus, LeafSystem,
                         MakePhongIllustrationProperties, MathematicalProgram,
                         MeshcatPointCloudVisualizer, MeshcatVisualizer,
                         MeshcatVisualizerParams, Parser, PiecewisePolynomial,
                         PiecewisePose, PointCloud, RigidTransform,
                         RollPitchYaw, RotationMatrix, Simulator, StartMeshcat,
                         ge)
from pydrake.common import FindResourceOrThrow
from pydrake.common.eigen_geometry import AngleAxis
from pydrake.multibody.meshcat import JointSliders
from manipulation.meshcat_cpp_utils import (AddMeshcatTriad,
                                            draw_open3d_point_cloud)
from manipulation.mustard_depth_camera_example import MustardPointCloud
from manipulation.open3d_utils import create_open3d_point_cloud
from manipulation.scenarios import (AddMultibodyTriad, AddRgbdSensor,
                                    MakeManipulationStation)
from manipulation.utils import AddPackagePaths, FindResource
import numpy as np
import rospy
from sensor_msgs.msg import JointState

"""
Note: I installed drake from source through CMake.
"""

class FrankaPlaybackSim:
    def __init__(self, position, velocity):
        self.meshcat = StartMeshcat()
        self.builder = DiagramBuilder()
        
        # Add a cube as MultibodyPlant
        self.plant, self.scene_graph = AddMultibodyPlantSceneGraph(self.builder, time_step=0.0)
        
        # self.X_model = RigidTransform(RollPitchYaw(-np.pi/2., 0, -np.pi/2.), [0, 0, 0.09515])
        self.X_model = RigidTransform.Identity()
        
        self.parser = Parser(self.plant)
        self.model_file = FindResourceOrThrow("drake/manipulation/models/franka_description/urdf/panda_arm_hand.urdf")
        self.model = self.parser.AddModelFromFile(self.model_file)
        
        self.plant.WeldFrames(self.plant.world_frame(),
                            self.plant.GetFrameByName("panda_link0", self.model),
                            self.X_model)

        # Add a box for the camera in the environment.
        # World to camera frame transformation
        axis_vec = [-1.57165949, -1.63112887, 1.07928078]
        angle = np.linalg.norm(axis_vec)
        axis = axis_vec / angle
        angle_axis = AngleAxis(angle=angle,axis=axis)
        self.X_Camera = RigidTransform(angle_axis, [1.14164360, 0.15815239, 0.66422200])

        self.camera_instance = self.parser.AddModelFromFile(FindResource("models/camera_box.sdf"))
        self.camera_frame = self.plant.GetFrameByName("base", self.camera_instance)    
        self.plant.WeldFrames(self.plant.world_frame(), self.camera_frame, self.X_Camera)
        AddMultibodyTriad(self.camera_frame, self.scene_graph, length=.1, radius=0.005)
        self.plant.Finalize()

        # Visualize in meshcat
        self.params = MeshcatVisualizerParams()
        self.visualizer = MeshcatVisualizer.AddToBuilder(
            self.builder, self.scene_graph, self.meshcat, self.params
        )
        X_PC = RigidTransform()
        self.camera = AddRgbdSensor(self.builder, self.scene_graph, X_PC=X_PC,
                           parent_frame_id=self.plant.GetBodyFrameIdOrThrow(
                               self.camera_frame.body().index()))
        self.camera.set_name("rgbd_sensor")

        # Export the camera outputs
        self.builder.ExportOutput(self.camera.color_image_output_port(), "color_image")
        self.builder.ExportOutput(self.camera.depth_image_32F_output_port(), "depth_image")
        

        # Setup contexts
        self.diagram = self.build()
        self.context = self.diagram.CreateDefaultContext()
        self.diagram.Publish(self.context)
        self.plant_context = self.plant.GetMyMutableContextFromRoot(self.context)
        self.plant.SetPositions(self.plant_context, position)
        self.plant.get_actuation_input_port().FixValue(self.plant_context, np.zeros(9))
        self.simulator = Simulator(self.diagram, self.context)
        self.simulator.Initialize()

        # TODO: why are the fingers so wide?
        
        # ROS - temporarily don't need this since we are not processing in real-time
        # self.diagram = self.build()
        # self.simulator = Simulator(self.diagram)
        # self.simulator.set_target_realtime_rate(1.0)
        # self.simulator.set_publish_every_time_step(False)
        # self.context = self.simulator.get_mutable_context()
        # self.state = self.context.get_mutable_continuous_state_vector()
        # self.state.SetFromVector(np.zeros(9*2))
        # self.simulator.Initialize()

    def build(self):
        # Setup JointSlider to control joint configuration
        # default_interactive_timeout = 1.0 if "TEST_SRCDIR" in os.environ else None
        # sliders = self.builder.AddSystem(JointSliders(meshcat, self.plant))
        diagram = self.builder.Build()
        # sliders.Run(diagram, default_interactive_timeout)
        diagram.set_name("depth_camera_demo_system")
        return diagram

    
    def visualize(self):
        def callback(msg):
            global q, v, f
            q = np.array(msg.position)
            v = np.array(msg.velocity)
            f = np.array(msg.effort)

        rospy.init_node('listener', anonymous=True)
        
        rospy.Subscriber('/joint_states', JointState, callback=callback)
        rate = rospy.Rate(4)
        rate.sleep()
        while not rospy.is_shutdown():
            
            self.state = self.context.get_mutable_continuous_state_vector()
            self.state.SetFromVector(np.append(q,v))
            self.plot_camera_images()
            self.simulator.Initialize()
            # self.builder.ExportOutput(self.camera.color_image_output_port(), "color_image")
            # self.builder.ExportOutput(self.camera.depth_image_32F_output_port(), "depth_image")
            # self.diagram = self.build()
            # self.context = self.diagram.CreateDefaultContext()
            rate.sleep()
            # break


    def plot_camera_images(self):
        # system = self.build()
        # Evaluate the camera output ports to get the images.
        # context = system.CreateDefaultContext()
        # self.diagram.Publish(self.context)
        color_image = self.diagram.GetOutputPort("color_image").Eval(self.context)
        depth_image = self.diagram.GetOutputPort("depth_image").Eval(self.context)

        # Plot the two images.
        plt.subplot(121)
        plt.imshow(color_image.data)
        plt.title('Color image')
        plt.subplot(122)
        plt.imshow(np.squeeze(depth_image.data))
        plt.title('Depth image')
        pdb.set_trace()
        #mpld3.display()
        # plt.show()
        cv2.imwrite('./data/simulated.png', depth_image.data)
    
'''
Filter the simulated image from the real depth image
'''
# def filter(real_img, sim_img):
#     masked_img = np.subtract(real_img, sim_img)
#     return masked_img

def import_data():
    img_file = "./data/images.txt"
    position_file = "./data/joint_position.txt"
    velocity_file = "./data/joint_velocity.txt"
    images = np.loadtxt(img_file)
    positions = np.loadtxt(position_file)
    velocities = np.loadtxt(velocity_file)
    return images, positions, velocities

if __name__ == "__main__":
    images, positions, velocities = import_data()
    system = FrankaPlaybackSim(positions[0], velocities[0])
    system.plot_camera_images()
    simulated_img = cv2.imread('./data/simulated.png')
    real_img = cv2.imread('./data/frame000000.png')
    mask = cv2.subtract(real_img, simulated_img)
    pdb.set_trace()
    plt.figure()
    plt.imshow(mask)
    plt.show()
    # cv2.imshow('mask', mask)