"""This script generates dynamics predictions for robot experiments based on
recorded impedance controller commands stored in a rosbag.

    - raw_302.bag (oatly, 75 seconds)
"""

import os.path as op
import numpy as np
import matplotlib.pyplot as plt
import torch
import yaml

from pydrake.common.eigen_geometry import AngleAxis, Quaternion
from pydrake.geometry import HalfSpace, MeshcatVisualizer, StartMeshcat
from pydrake.math import RigidTransform
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import AddMultibodyPlant, CoulombFriction, \
    MultibodyPlant, MultibodyPlantConfig
from pydrake.multibody.tree import JacobianWrtVariable
from pydrake.systems.analysis import Simulator
from pydrake.systems.drawing import plot_graphviz, plot_system_graphviz
from pydrake.systems.framework import DiagramBuilder, LeafSystem
from pydrake.systems.primitives import TrajectorySource
from pydrake.trajectories import PiecewisePolynomial, PiecewisePose, \
    PiecewiseQuaternionSlerp, StackedTrajectory
from pydrake.visualization import AddFrameTriadIllustration

import file_utils
import rosbag_processor


SIM_TIME_STEP = 5e-4
CONTROLLER_TIME_STEP = 1e-3
VIZ_TIME_STEP = 5e-3

WORLD_TO_FRANKA_PLL_OFFSET = 0.0087

OBJECT = 'robotocc_oatly'
TEST_TOSS = 6
VISION_ASSET = f'{OBJECT}_{TEST_TOSS}'
BSDF_ITERATION = 1
PLL_ID = 'pll_id_t02e300b20_occoatly_6'


EXPORT_TEST_DATA = False
DEBUG = False


if EXPORT_TEST_DATA:
    start = file_utils.load_toss_time_from_yaml(
        OBJECT, TEST_TOSS, 'start_time', as_ros_time=True)
    end = file_utils.load_toss_time_from_yaml(
        OBJECT, TEST_TOSS, 'end_time', as_ros_time=True)

    rosbag_processor.extract_franka_states(
        start_time=start, end_time=end,
        bag_file='/home/bibit/vision/bundlenets/cnets-data-generation/rosbags/raw_302.bag',
        franka_states_output_dir='robot_dynamics'
    )
    rosbag_processor.extract_pose_commands(
        start_time=start, end_time=end,
        bag_file='/home/bibit/vision/bundlenets/cnets-data-generation/rosbags/raw_302.bag',
        ee_pose_command_output_dir='robot_dynamics'
    )
    breakpoint()


def visualize_drake_systems(plant=None, diagram=None):
    plt.figure()
    if diagram is not None:
        plot_system_graphviz(diagram)
    elif plant is not None:
        plot_graphviz(plant.GetTopologyGraphvizString())
    else:
        raise ValueError('Need to provide either a plant or a diagram.')
    plt.plot(1)
    plt.show(block=False)


### Inverse dynamics controller ###
class InverseDynamicsController(LeafSystem):
    def __init__(self, plant: MultibodyPlant, controller_params: dict):
        LeafSystem.__init__(self)
        self.plant = plant

        # Store necessary controller parameters and hard-coded settings.
        self._set_param_settings(controller_params)
        self._set_default_settings()

        # Declare inputs and outputs.
        self.ee_pose_command_input_index = self.DeclareVectorInputPort(
            'ee_pose_command', 7).get_index()
        self.robot_state_input_index = self.DeclareVectorInputPort(
            'robot_state', plant.num_positions() + plant.num_velocities()
        ).get_index()
        self.joint_torque_output_index = self.DeclareVectorOutputPort(
            'joint_torques', plant.num_actuators(), self.CalcControl
        ).get_index()
        
    def get_ee_pose_command_input_port(self):
        return self.get_input_port(self.ee_pose_command_input_index)
        
    def get_robot_state_input_port(self):
        return self.get_input_port(self.robot_state_input_index)
        
    def get_joint_torque_output_port(self):
        return self.get_output_port(self.joint_torque_output_index)

    def _set_param_settings(self, controller_params):
        self.q_d_nullspace = np.array(controller_params[
            'hybrid_impedance_wrench_controller']['q_d_nullspace'])
        self.nullspace_stiffness = np.array(controller_params[
            'hybrid_impedance_wrench_controller']['nullspace_stiffness'])

        self.cartesian_trans_acc_norm_limit = controller_params[
            'hybrid_impedance_wrench_controller'][
            'cartesian_trans_acc_norm_limit']
        self.cartesian_rot_acc_norm_limit = controller_params[
            'hybrid_impedance_wrench_controller'][
            'cartesian_rot_acc_norm_limit']

    def _set_default_settings(self):
        self.delta_tau_max = 1000.0
        self.filter_param = 0.005

        self.position_d = np.zeros(3)
        self.position_d_target = np.zeros(3)
        self.quaternion_d = Quaternion()
        self.quaternion_d_target = Quaternion()

    def StartFrom(self, joint_angles: np.ndarray, joint_velocities: np.ndarray,
                  joint_torques: np.ndarray, cartesian_stiffness: np.ndarray,
                  cartesian_damping: np.ndarray):
        """Resets the controller to the initial state and sets the controller to
        the initial joint angles."""
        self.cartesian_stiffness = cartesian_stiffness
        self.cartesian_damping = cartesian_damping

        self._set_controller_plant_state_and_context(
            joint_angles, joint_velocities)

        # Store some initial set of joint torques.
        # TODO better practice to use Drake state instead of class variable, or
        # to pipe the tau_d output to a DiscreteTimeDelay that feeds into this
        # system's input ports.
        self.tau_J_d = joint_torques

        # Set the equilibrium point to the initial condition.
        ee_pose = self.plant.EvalBodyPoseInWorld(
            self.plant_context, self.plant.GetBodyByName('end_effector_tip'))
        self.position_d = ee_pose.translation()
        self.position_d_target = ee_pose.translation()
        self.orientation_d = ee_pose.rotation().ToQuaternion()
        self.orientation_d_target = ee_pose.rotation().ToQuaternion()

    def _set_controller_plant_state_and_context(
            self, joint_angles: np.ndarray, joint_velocities: np.ndarray):
        if not hasattr(self, 'plant_context'):
            self.plant_context = self.plant.CreateDefaultContext()
        self.plant.SetPositions(self.plant_context, joint_angles)
        self.plant.SetVelocities(self.plant_context, joint_velocities)

    def _update_desired_targets(self, context):
        """Based on the new commanded pose, update the desired targets.  The
        pose is in the form [qw qx qy qz x y z]."""
        pose_command = self.EvalVectorInput(
            context, self.ee_pose_command_input_index).get_value().copy()
        self.position_d_target = pose_command[4:7]

        # Ensure quaternion representation is consistent.
        last_orientation_d_target = self.orientation_d_target.wxyz()
        new_orientation_d_target = pose_command[:4]
        assert np.abs(np.linalg.norm(new_orientation_d_target) - 1) < 1e-3, \
            f'Expected normalized quaternion but got ' + \
            f'{np.linalg.norm(new_orientation_d_target)} from ' + \
            f'{new_orientation_d_target} -- is the pose command in the ' + \
            f'right order?  {pose_command=}'
        if np.dot(last_orientation_d_target, new_orientation_d_target) < 0:
            new_orientation_d_target *= -1
        self.orientation_d_target = Quaternion(new_orientation_d_target)

    def _saturate_torque_rate(self, tau_d_calculated):
        tau_d_saturated = np.zeros(7)
        for i in range(7):
            difference = tau_d_calculated[i] - self.tau_J_d[i]
            tau_d_saturated[i] = self.tau_J_d[i] + np.clip(
                difference, -self.delta_tau_max, self.delta_tau_max)
        return tau_d_saturated

    def _saturate_cartesian_acceleration(self, cart_acc_des):
        cart_acc_saturated = cart_acc_des
        if np.abs(cart_acc_des[2] > self.cartesian_trans_acc_norm_limit):
            cart_acc_saturated[2] = np.sign(cart_acc_des[2]) * \
                self.cartesian_trans_acc_norm_limit
        return cart_acc_saturated

    def CalcControl(self, context, output):
        """Compute the control law for the robot, emulating the controller used
        for the Franka teleop setup."""
        # First get the current commanded pose and robot state.
        self._update_desired_targets(context)

        x = self.EvalVectorInput(
            context, self.robot_state_input_index).get_value()
        q = x[:self.plant.num_positions()]
        v = x[self.plant.num_positions():]

        # Compute Drake's quantities like mass matrix, jacobian, coriolis,
        # gravity, etc.  Slower but more accurate option if needed:
        # CalcMassMatrixViaInverseDynamics
        self._set_controller_plant_state_and_context(q, v)
        mass_matrix = self.plant.CalcMassMatrix(self.plant_context)
        M_inv = np.linalg.inv(mass_matrix)
        coriolis = self.plant.CalcBiasTerm(self.plant_context)
        gravity = self.plant.CalcGravityGeneralizedForces(self.plant_context)
        Jv_v_AB = self.plant.CalcJacobianTranslationalVelocity(
            self.plant_context,
            with_respect_to=JacobianWrtVariable.kV,
            frame_B=self.plant.GetFrameByName('end_effector_tip'),
            p_BoBi_B=np.zeros(3),
            frame_A=self.plant.world_frame(),
            frame_E=self.plant.world_frame()
        )  # (3, 7)
        Jv_w_AB = self.plant.CalcJacobianAngularVelocity(
            self.plant_context,
            with_respect_to=JacobianWrtVariable.kV,
            frame_B=self.plant.GetFrameByName('end_effector_tip'),
            frame_A=self.plant.world_frame(),
            frame_E=self.plant.world_frame()
        )  # (3, 7)
        jacobian = np.concatenate((Jv_v_AB, Jv_w_AB), axis=0)  # (6, 7)

        # Get the current end effector pose.
        ee_pose = self.plant.EvalBodyPoseInWorld(
            self.plant_context, self.plant.GetBodyByName('end_effector_tip'))
        position = ee_pose.translation()
        orientation = ee_pose.rotation().ToQuaternion()

        # Compute errors.
        position_error = position - self.position_d
        quaternion_error = self.orientation_d.multiply(orientation.inverse())
        angle_axis_error = AngleAxis(quaternion=quaternion_error)
        orientation_error = -angle_axis_error.axis() * angle_axis_error.angle()
        error = np.concatenate((position_error, orientation_error))
        assert error.shape == (6,)

        # Compute control -- no feedforward term needed because desired wrench
        # is always zero.
        # tau_wrench = jacobian.T @ np.concatenate((force_d, torque_d))

        # Compute control -- second task.
        cart_acc_des = -self.cartesian_stiffness * error - \
            self.cartesian_damping * (jacobian @ v)
        cart_acc_des = self._saturate_cartesian_acceleration(cart_acc_des)
        delassus_inv = np.linalg.inv(jacobian @ M_inv @ jacobian.T)
        tau_task = jacobian.T @ delassus_inv @ cart_acc_des

        # Compute control -- third nullspace.
        tau_q_pd = self.nullspace_stiffness * (self.q_d_nullspace - q) - \
            (2.0 * np.sqrt(self.nullspace_stiffness) * v)
        jacobian_dyncost_pinv_transpose = delassus_inv @ jacobian @ M_inv
        tau_nullspace = (
            np.eye(7) - jacobian.T @ jacobian_dyncost_pinv_transpose) @ tau_q_pd
        
        # Desired torque.
        tau_d = tau_task + tau_nullspace + coriolis - gravity  # + tau_wrench
        tau_d = self._saturate_torque_rate(tau_d)
        # tau_d = coriolis - gravity  # sanity check can make robot not move

        # Set the output to the desired torques.
        assert tau_d.ndim == 1
        assert tau_d.shape[0] == self.plant.num_actuators() == 7
        output.SetFromVector(tau_d)

        # Do updates for the next iteration.
        self._end_of_calc_updates(tau_d)

    def _end_of_calc_updates(self, tau_d):
        self.tau_J_d = tau_d

        self.position_d = self.filter_param * \
            self.position_d_target + (1 - self.filter_param) * \
            self.position_d
        
        # Special case to handle the quaternion.
        wxyz_d = self.orientation_d.wxyz()
        wxyz_d_target = self.orientation_d_target.wxyz()
        wxyz_d = self.filter_param * \
            wxyz_d_target + (1 - self.filter_param) * wxyz_d
        self.orientation_d = Quaternion(wxyz_d/np.linalg.norm(wxyz_d))


### Load data ###
command_ts = np.loadtxt('robot_dynamics/pose_command_times.txt') # (N,)
des_pos = np.loadtxt('robot_dynamics/des_pos.txt') # (N, 3)
des_quat_wxyz = np.loadtxt('robot_dynamics/des_quat_wxyz.txt') # (N, 4)

des_quats = []
for quat_wxyz in des_quat_wxyz:
    des_quats.append(Quaternion(quat_wxyz))

franka_joint_ts = np.loadtxt('robot_dynamics/joint_times.txt')  # (M,)
joint_angles = np.loadtxt('robot_dynamics/joint_angles.txt')  # (M, 7)
joint_velocities = np.loadtxt('robot_dynamics/joint_velocities.txt')  # (M, 7)
joint_torques = np.loadtxt('robot_dynamics/tau_J_ds.txt')  # (M, 7)

cartesian_stiffness = np.loadtxt('robot_dynamics/cartesian_stiffness.txt') #(6,)
cartesian_damping = np.loadtxt('robot_dynamics/cartesian_damping.txt')  # (6,)

# Zero out the trajectories.  TODO may need to keep track of this.
init_t = min(command_ts[0], franka_joint_ts[0])
command_ts -= init_t
franka_joint_ts -= init_t


### Load object-related data ###
# # Get the table height.  Use the average if using multiple tosses.
# table_heights = np.array([
#     file_utils.load_table_z_height(OBJECT, toss) for toss in
#     range(TEST_TOSS, TEST_TOSS+1)
# ])
# z_table = np.mean(table_heights)

# Offset the commanded end effector poses from the table height.  The Franka
# will also get offset by the same amount in the simulations.
# des_pos[:, 2] -= z_table
des_pos[:, 2] += WORLD_TO_FRANKA_PLL_OFFSET

# Load the object poses.
toss_dir = file_utils.contactnets_input_dir_bundlesdf(
    VISION_ASSET, BSDF_ITERATION, 'bundlesdf_id_00', full=False,
    create=False)
toss_data_dict = torch.load(
    op.join(toss_dir, f'{TEST_TOSS}.pt'), weights_only=False)
object_states = np.array(toss_data_dict['object_state'])
object_poses = object_states[:, :7]  # quat_wxyz, pos
cnets_data_gen_dir = file_utils.cnets_data_gen_dataset_dir(
    VISION_ASSET, check_exists=True)
object_pose_ts = np.loadtxt(op.join(
    cnets_data_gen_dir, 'bundlesdf_timestamps.txt')) - init_t


### Drake trajectory ###
position_trajectory = PiecewisePolynomial.CubicWithContinuousSecondDerivatives(
    breaks=command_ts,
    samples=des_pos.T,
    sample_dot_at_start=np.zeros(3),
    sample_dot_at_end=np.zeros(3)
)
orientation_trajectory = PiecewiseQuaternionSlerp(
    breaks=command_ts,
    quaternions=des_quats
)
# Sadly the more fool-proof PiecewisePose outputs 4x4 homogeneous transform
# matrices, but TrajectorySource needs a column vector.  Use StackedTrajectory
# instead, and use caution when interpreting the output ordering.
commanded_quat_pos_traj = StackedTrajectory()
commanded_quat_pos_traj.Append(orientation_trajectory)
commanded_quat_pos_traj.Append(position_trajectory)

# Build the object trajectory.
recorded_object_quat_traj = PiecewiseQuaternionSlerp(
    breaks=object_pose_ts,
    quaternions=[Quaternion(q) for q in object_poses[:, :4]]
)
recorded_object_pos_traj = \
    PiecewisePolynomial.CubicWithContinuousSecondDerivatives(
        breaks=object_pose_ts,
        samples=object_poses[:, 4:].T,
        sample_dot_at_start=np.zeros(3),
        sample_dot_at_end=np.zeros(3)
    )
recorded_object_quat_pos_traj = StackedTrajectory()
recorded_object_quat_pos_traj.Append(recorded_object_quat_traj)
recorded_object_quat_pos_traj.Append(recorded_object_pos_traj)


# TODO would it be better to combine angles and velocities into one trajectory,
# since technically the derivatives affect each other?  Maybe torques too?
gt_joint_angle_traj = PiecewisePolynomial.CubicWithContinuousSecondDerivatives(
    breaks=franka_joint_ts,
    samples=joint_angles.T,
    sample_dot_at_start=np.zeros(7),
    sample_dot_at_end=np.zeros(7)
)
gt_joint_vel_traj = PiecewisePolynomial.CubicWithContinuousSecondDerivatives(
    breaks=franka_joint_ts,
    samples=joint_velocities.T,
    sample_dot_at_start=np.zeros(7),
    sample_dot_at_end=np.zeros(7)
)
gt_joint_torque_traj = PiecewisePolynomial.CubicWithContinuousSecondDerivatives(
    breaks=franka_joint_ts,
    samples=joint_torques.T,
    sample_dot_at_start=np.zeros(7),
    sample_dot_at_end=np.zeros(7)
)


### Simulation Drake diagram ###
builder = DiagramBuilder()

# Add the robot to a simulation.
sim_mbp_config = MultibodyPlantConfig(time_step=SIM_TIME_STEP)
sim_plant, scene_graph = AddMultibodyPlant(sim_mbp_config, builder)
sim_parser = Parser(sim_plant)
sim_robot = sim_parser.AddModels(
    file_utils.franka_filepath(with_collision_geometry=True))[0]
sim_plant.WeldFrames(sim_plant.world_frame(),
                     sim_plant.GetFrameByName('panda_link0'),
                     RigidTransform(p=np.array([
                        0, 0, WORLD_TO_FRANKA_PLL_OFFSET])))

# Add the table (as a half space) to the simulation, located at z=0.
# TODO:  meshcat doesn't display halfspace geometries, so may want to use a box
# instead.
friction = CoulombFriction(1.0, 1.0)
sim_plant.RegisterCollisionGeometry(
    sim_plant.world_body(), RigidTransform(), HalfSpace(),
    'table', friction)
sim_plant.RegisterVisualGeometry(
    sim_plant.world_body(), RigidTransform(), HalfSpace(),
    'table', np.array([0.5, 0.5, 0.5, 1]))

# Add the object to the simulation.
sim_object = sim_parser.AddModels(op.join(
    file_utils.get_pll_urdf_output_dir(VISION_ASSET, BSDF_ITERATION, PLL_ID),
    'with_bundlesdf_mesh.urdf'))[0]

sim_plant.Finalize()
sim_plant.set_name('sim_plant')
sim_plant.SetDefaultPositions(np.vstack((
    gt_joint_angle_traj.value(commanded_quat_pos_traj.start_time()),
    recorded_object_quat_pos_traj.value(commanded_quat_pos_traj.start_time())
)))
if DEBUG:  visualize_drake_systems(plant=sim_plant)

# Add a trajectory source for the desired end effector pose.
traj_source = builder.AddSystem(TrajectorySource(commanded_quat_pos_traj))

# Add the controller, which needs a separate plant for control with just the
# robot.
control_plant = MultibodyPlant(time_step=CONTROLLER_TIME_STEP)
control_parser = Parser(control_plant)
control_robot = control_parser.AddModels(
    file_utils.franka_filepath(with_collision_geometry=True))[0]
control_plant.WeldFrames(control_plant.world_frame(),
                         control_plant.GetFrameByName('panda_link0'),
                         RigidTransform(p=np.array([
                            0, 0, WORLD_TO_FRANKA_PLL_OFFSET])))
control_plant.Finalize()
control_plant.set_name('control_plant')
control_plant.SetDefaultPositions(
    gt_joint_angle_traj.value(commanded_quat_pos_traj.start_time()))
control_params_file = 'robot_dynamics/franka_hw_controllers.yaml'
with open(control_params_file, 'r') as f:
    control_params = yaml.safe_load(f)
inv_dyn_controller = builder.AddSystem(
    InverseDynamicsController(control_plant, control_params))
inv_dyn_controller.set_name('inv_dyn_controller')

# Wire the diagram.
builder.Connect(traj_source.get_output_port(),
                inv_dyn_controller.get_ee_pose_command_input_port())
builder.Connect(sim_plant.get_state_output_port(sim_robot),
                inv_dyn_controller.get_robot_state_input_port())
builder.Connect(inv_dyn_controller.get_joint_torque_output_port(),
                sim_plant.get_actuation_input_port())

diagram = builder.Build()
if DEBUG:  visualize_drake_systems(diagram=diagram); breakpoint()
simulator = Simulator(diagram)
simulator.Initialize()
simulator.set_target_realtime_rate(1)


### Visualization Drake diagram ###
viz_builder = DiagramBuilder()

# Add the robot and a floating end effector to the plant.
viz_mbp_config = MultibodyPlantConfig(time_step=VIZ_TIME_STEP)
viz_plant, viz_scene_graph = AddMultibodyPlant(viz_mbp_config, viz_builder)
viz_parser = Parser(viz_plant)
viz_robot = viz_parser.AddModels(
    file_utils.franka_filepath(with_collision_geometry=True))
viz_plant.WeldFrames(viz_plant.world_frame(),
                     viz_plant.GetFrameByName('panda_link0'),
                     RigidTransform(p=np.array([
                        0, 0, WORLD_TO_FRANKA_PLL_OFFSET])))
viz_commanded_ee = viz_parser.AddModels(file_utils.ee_urdf_filepath())
viz_gt_object = viz_parser.AddModels(file_utils.get_urdf_with_bundlesdf_mesh(
    f'vision_{OBJECT}', VISION_ASSET, BSDF_ITERATION, PLL_ID))[0]
# TODO:  by default, both of these URDFs have the same model name, which means
# they can't both be added to the plant unless one of the URDFs is edited.  It's
# also useful to change the colors of each object to distinguish them and to
# allow seeing their overlap via transparency.
viz_object = viz_parser.AddModels(op.join(
    file_utils.get_pll_urdf_output_dir(VISION_ASSET, BSDF_ITERATION, PLL_ID),
    'with_bundlesdf_mesh.urdf'))[0]
viz_plant.Finalize()
viz_plant.set_name('viz_plant')
viz_plant.SetDefaultPositions(np.vstack((
    gt_joint_angle_traj.value(commanded_quat_pos_traj.start_time()),
    commanded_quat_pos_traj.value(commanded_quat_pos_traj.start_time()),
    recorded_object_quat_pos_traj.value(commanded_quat_pos_traj.start_time()),
    recorded_object_quat_pos_traj.value(commanded_quat_pos_traj.start_time())
)))
AddFrameTriadIllustration(
    scene_graph=viz_scene_graph,
    body=viz_plant.GetBodyByName('floating_end_effector_tip'),
    length=0.05, radius=0.008, opacity=0.5
)
AddFrameTriadIllustration(
    scene_graph=viz_scene_graph,
    body=viz_plant.GetBodyByName('end_effector_tip'),
    length=0.1, radius=0.005, opacity=0.8
)

# Add a meshcat visualizer.
meshcat = StartMeshcat()
MeshcatVisualizer.AddToBuilder(viz_builder, viz_scene_graph, meshcat)

# Build the diagram.
viz_diagram = viz_builder.Build()
if DEBUG:  visualize_drake_systems(diagram=viz_diagram)
viz_simulator = Simulator(viz_diagram)
viz_context = viz_simulator.get_context()
viz_diagram.ForcedPublish(viz_context)

breakpoint()

### Simulate an experiment ###
# Prepare to run a simulation.
sim_plant_context = sim_plant.CreateDefaultContext()
sim_plant.SetPositions(sim_plant_context, np.vstack((
    gt_joint_angle_traj.value(0),
    recorded_object_quat_pos_traj.value(0))))
inv_dyn_controller.StartFrom(
    joint_angles=gt_joint_angle_traj.value(0),
    joint_velocities=gt_joint_vel_traj.value(0),
    joint_torques=gt_joint_torque_traj.value(0),
    cartesian_stiffness=cartesian_stiffness,
    cartesian_damping=cartesian_damping
)

t_final = min(10.0, commanded_quat_pos_traj.end_time())
for t in np.arange(0, t_final, VIZ_TIME_STEP):
    # Run the simulation.
    simulator.AdvanceTo(t)

    # Get the current joint angles from the simulator.
    sim_context = simulator.get_context()
    sim_plant_context = sim_plant.GetMyContextFromRoot(sim_context)
    sim_positions = sim_plant.GetPositions(sim_plant_context)

    franka_joint_angles = sim_positions[:7]
    object_quat_pos = sim_positions[7:]

    # Update the visualization.
    viz_states = np.vstack((
        franka_joint_angles.reshape(-1, 1),
        commanded_quat_pos_traj.value(t),
        recorded_object_quat_pos_traj.value(t),
        object_quat_pos.reshape(-1, 1)
    ))
    viz_plant_context = viz_plant.GetMyMutableContextFromRoot(viz_context)
    viz_plant.SetPositions(viz_plant_context, viz_states)
    viz_context = viz_simulator.get_context()
    viz_diagram.ForcedPublish(viz_context)


breakpoint()
