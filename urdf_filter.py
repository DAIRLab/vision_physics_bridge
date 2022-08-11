import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pdb
# import imageio
from PIL import Image
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder
from pydrake.math import RigidTransform
from pydrake.all import (AddMultibodyPlantSceneGraph, AngleAxis,
                         CameraInfo, ClippingRange, DepthRange,
                         DepthRenderCamera, DiagramBuilder,
                         MeshcatVisualizer, MeshcatVisualizerParams, Parser,
                         PiecewisePose, PointCloud, RigidTransform,
                         RollPitchYaw, RenderCameraCore, RotationMatrix, Simulator, StartMeshcat)
from pydrake.common import FindResourceOrThrow
from pydrake.common.eigen_geometry import AngleAxis
from pydrake.multibody.meshcat import JointSliders
from manipulation.scenarios import (AddMultibodyTriad, AddRgbdSensor)
from manipulation.utils import FindResource
import numpy as np
import rospy
from sensor_msgs.msg import JointState
from utils import import_data, filter

"""
Note: I installed drake from source through CMake.
"""

class FrankaPlaybackSim:
    def __init__(self, position, velocity):
        self.meshcat = StartMeshcat()
        self.builder = DiagramBuilder()
        
        # Add a cube as MultibodyPlant
        self.plant, self.scene_graph = AddMultibodyPlantSceneGraph(self.builder, time_step=0.0)
        self.X_model = RigidTransform.Identity()
        self.parser = Parser(self.plant)
        self.model_file = FindResourceOrThrow("drake/manipulation/models/franka_description/urdf/panda_arm_hand.urdf")
        self.model = self.parser.AddModelFromFile(self.model_file)
        
        self.plant.WeldFrames(self.plant.world_frame(),
                            self.plant.GetFrameByName("panda_link0", self.model),
                            self.X_model)

        # Add a box for the camera in the environment.
        camera_pos = [1.14164360, 0.15815239, 0.66422200]
        axis_vec = [-1.57165949, -1.63112887, 1.07928078]
        angle = np.linalg.norm(axis_vec)
        axis = axis_vec / angle
        angle_axis = AngleAxis(angle=angle,axis=axis)
        self.X_Camera = RigidTransform(angle_axis, camera_pos)

        self.camera_instance = self.parser.AddModelFromFile(FindResource("models/camera_box.sdf"))
        self.camera_frame = self.plant.GetFrameByName("base", self.camera_instance)    
        self.plant.WeldFrames(self.plant.world_frame(), self.camera_frame, self.X_Camera)
        # Not sure what this line does
        # AddMultibodyTriad(self.camera_frame, self.scene_graph, length=.1, radius=0.005)
        self.plant.Finalize()

        # Visualize in meshcat
        self.params = MeshcatVisualizerParams()
        self.visualizer = MeshcatVisualizer.AddToBuilder(
            self.builder, self.scene_graph, self.meshcat, self.params
        )
        X_PC = RigidTransform()
        renderer = "my_renderer"
        intrinsic = CameraInfo(640, 480, 380.2484436035156, 379.8265380859375, 314.2138977050781, 240.59800720214844)
        depth_camera = DepthRenderCamera(
            RenderCameraCore(
                renderer, intrinsic,
                ClippingRange(near=0.1, far=10.0), RigidTransform()),
            DepthRange(0.1, 10.0))
        self.camera = AddRgbdSensor(self.builder, self.scene_graph, X_PC=X_PC, depth_camera=depth_camera,
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
            rate.sleep()
            # break


    def plot_camera_images(self, frame_id):
        color_image = self.diagram.GetOutputPort("color_image").Eval(self.context)
        depth_image = self.diagram.GetOutputPort("depth_image").Eval(self.context)

        # Plot the two images.
        plt.subplot(121)
        
        # plt.imshow(color_image.data)
        # plt.title('Color image')

        # real = plt.imread('./aligned_data/frame00000{}.png'.format(frame_id))

        loaded_arr = np.loadtxt("./aligned_data/images.txt")
        load_original_arr = loaded_arr.reshape(
        loaded_arr.shape[0], loaded_arr.shape[1] // 640, 640)
        real = load_original_arr[0]
        np.savetxt('./aligned_data/real_depth.txt', real)
        _min = 0
        _max = 1
        plt.imshow(real, vmin = _min, vmax = _max)
        # plt.imshow(real)
        # red dot on the end-effector
        # plt.plot(300, 100, 'ro')
        # red dot on the panda_link0
        plt.plot(320,250,'ro')
        plt.colorbar()
        plt.title('Real image')
        plt.subplot(122)
        plt.imshow(np.squeeze(depth_image.data), vmin = _min, vmax = _max) #(480.640,1)
        # plt.imshow(depth_image.data[:,:,0])
        plt.plot(320, 250, 'ro')
        plt.colorbar()
        plt.title('Depth image')
        np.savetxt('./aligned_data/simulated_depth.txt', depth_image.data[:,:,0])
        
        # print(real[315:320, 250:255])
        # print(depth_image.data[315:320, 250:255,0])
        # pdb.set_trace()
        # plt.show()

    def mask_out_everything_except_cube(self):
        return 

if __name__ == "__main__":
    img_file = "./aligned_data/images.txt"
    position_file = "./aligned_data/joint_position.txt"
    velocity_file = "./aligned_data/joint_velocity.txt"
    images, positions, velocities = import_data(img_file, position_file, velocity_file)
    frame_id = 1
    system = FrankaPlaybackSim(positions[frame_id], velocities[frame_id])
    system.plot_camera_images(frame_id)
    simulated_image = np.loadtxt('./aligned_data/simulated_depth.txt')
    # real_image = plt.imread('./aligned_data/frame00000{}.png'.format(frame_id))
    real_image = np.loadtxt('./aligned_data/real_depth.txt')
    # Need to multiply real depth images by 1000 since simulated depth image uses mm as unit
    mask = filter(real_image*1000, simulated_image)
    im = Image.fromarray(mask)
    if im.mode != 'RGB':
        im = im.convert('RGB')
    im.save("./aligned_data/mask.png")

    # pdb.set_trace()
    # plt.figure()
    # plt.imshow(mask)
    plt.show()