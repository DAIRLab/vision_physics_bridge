"""This script generates dynamics predictions for robot experiments based on
recorded impedance controller commands stored in a rosbag.

    - raw_302.bag (oatly, 75 seconds)

/diagnostics
/hybrid_impedance_wrench_controller/pose_viz
/msg
/panda/franka_control/error_recovery/status
/panda/franka_state_controller_custom/F_ext
/panda/franka_state_controller_custom/franka_states
/panda/franka_state_controller_custom/joint_states
/panda/franka_state_controller_custom/joint_states_desired
/panda/home_pose
/panda/hybrid_impedance_wrench_controller/control_diagnostic
/panda/hybrid_impedance_wrench_controller/pose_wrench_desired
/panda/hybrid_impedance_wrench_controllerdynamic_reconfigure_torque_limits_node/
    parameter_descriptions
/panda/hybrid_impedance_wrench_controllerdynamic_reconfigure_torque_limits_node/
    parameter_updates
/panda/joint_states
/panda/spacenav/joy
/panda/spacenav/offset
/panda/spacenav/rot_offset
/panda/spacenav/twist
/panda/virtual_wall_viz
/rosout
/tf

Steps in the inverse dynamics impedance controller:
 - subscribe to pose_wrench_desired
 - set up publisher at control_diagnostic
 - configure nullspace stiffness
 - get a bunch of things from Franka HW, e.g. joint/state interfaces
 - create node for dynamic reconfigure torque limits
 - set a bunch of desired values and limits

Update:  desired pose -> desired joint torques
    - task-space PD
    - nullspace stiffness
    - coriolis
    - feedforward wrench
    - in sim, should add gravity compensation too (Franka does this
      automatically in hardware)

Other notes:
    - franka_states contains mass matrix, coriolis, joint states, O_T_EE, etc.,
      but might want to get these from Drake instead.
    - task-space error is saturated, only for z translation
    - torque rate is also saturated
    - tau_filter_coeff shouldn't do anything
    - can check control_diagnostic for ROS params, or check code for static
        settings at fish_franka/config/franka_hw_controllers.yaml
    - libfranka uses some parameters in fish_franka/config/
      franka_control_node.yaml
        - check the bag to see if any joint configuration/effort/velocity/
          acceleration limits are hit, then might be hard to simulate.
"""

import numpy as np
import matplotlib.pyplot as plt
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
from pydrake.trajectories import PiecewisePolynomial, \
    PiecewiseQuaternionSlerp, StackedTrajectory

import file_utils
import rosbag_processor


SIM_TIME_STEP = 1e-4
CONTROLLER_TIME_STEP = 1e-3


EXPORT_TEST_DATA = False


if EXPORT_TEST_DATA:
    start = file_utils.load_toss_time_from_yaml(
        'robotocc_oatly', 1, 'start_time', as_ros_time=True)
    end = file_utils.load_toss_time_from_yaml(
        'robotocc_oatly', 1, 'end_time', as_ros_time=True)

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


### Inverse dynamics controller ###
'''TODO:
    - confirm all the recorded wrench_d.force and wrench_d.torque are zeros.
      if not, will need to reset them in _update_desired_targets.
    - confirm all the cartesian_stiffness and cartesian_damping are the same
      throughout the whole bags.  if not, will need to reset them in
      _update_desired_targets.
    - see if I need to implement what's in torquelimitsParamCallback.
    - confirm tau_J_d are joint efforts.

Notes:
    - removed exponential filtering on tau_d (so removed tau_filter_coeff and
      tau_d_last).
'''
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
        self.nullspace_stiffness_target = np.array(controller_params[
            'hybrid_impedance_wrench_controller']['nullspace_stiffness'])

        self.pinv_lambda = controller_params[
            'hybrid_impedance_wrench_controller']['pinv_lambda']

        self.cartesian_trans_acc_norm_limit = controller_params[
            'hybrid_impedance_wrench_controller'][
            'cartesian_trans_acc_norm_limit']
        self.cartesian_trans_acc_norm_limit_target = controller_params[
            'hybrid_impedance_wrench_controller'][
            'cartesian_trans_acc_norm_limit']
        self.cartesian_rot_acc_norm_limit = controller_params[
            'hybrid_impedance_wrench_controller'][
            'cartesian_rot_acc_norm_limit']
        self.cartesian_rot_acc_norm_limit_target = controller_params[
            'hybrid_impedance_wrench_controller'][
            'cartesian_rot_acc_norm_limit']

    def _set_default_settings(self):
        self.delta_tau_max = 1000.0
        self.filter_params = 0.005

        self.position_d = np.zeros(3)
        self.position_d_target = np.zeros(3)
        self.quaternion_d = Quaternion()
        self.quaternion_d_target = Quaternion()

        self.force_d = np.zeros(3)
        self.force_d_target = np.zeros(3)
        self.torque_d = np.zeros(3)
        self.torque_d_target = np.zeros(3)

        self.cartesian_stiffness = np.zeros(6)
        self.cartesian_stiffness_target = np.zeros(6)
        self.cartesian_damping = np.zeros(6)
        self.cartesian_damping_target = np.zeros(6)

        self.torque_upper_limits = np.array([
            87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0])
        self.torque_upper_limits_target = np.array([
            87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0])
        self.torque_lower_limits = np.array([
            87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0])
        self.torque_lower_limits_target = np.array([
            87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0])
        
        self.cart_acc_upper_limits = np.array([
            20.0, 20.0, 20.0, 40.0, 40.0, 40.0])
        self.cart_acc_upper_limits_target = np.array([
            20.0, 20.0, 20.0, 40.0, 40.0, 40.0])
        self.cart_acc_lower_limits = np.array([
            20.0, 20.0, 20.0, 40.0, 40.0, 40.0])
        self.cart_acc_lower_limits_target = np.array([
            20.0, 20.0, 20.0, 40.0, 40.0, 40.0])

    def StartFrom(self, joint_angles: np.ndarray, joint_velocities: np.ndarray,
                  joint_torques: np.ndarray):
        """Resets the controller to the initial state and sets the controller to
        the initial joint angles."""
        self.cartesian_stiffness_target[:3] = 100.0
        self.cartesian_stiffness_target[3:] = 10.0
        self.cartesian_dampling_target = 2.0 * np.sqrt(
            self.cartesian_stiffness_target)

        self._set_controller_plant_state(joint_angles, joint_velocities)

        # Store some initial set of joint torques.
        self.tau_J_d = joint_torques

        # Set the equilibrium point to the initial condition.
        ee_pose = self.plant.EvalBodyPoseInWorld(
            self.plant_context, self.plant.GetBodyByName('end_effector_tip'))
        self.position_d = ee_pose.translation()
        self.position_d_target = ee_pose.translation()
        self.orientation_d = ee_pose.rotation().ToQuaternion()
        self.orientation_d_target = ee_pose.rotation().ToQuaternion()

    def _set_controller_plant_state(self, joint_angles, joint_velocities):
        if not hasattr(self, 'plant_context'):
            self.plant_context = self.plant.CreateDefaultContext()
        self.plant.SetPositions(self.plant_context, joint_angles)
        self.plant.SetVelocities(self.plant_context, joint_velocities)

    def _update_desired_targets(self, context):
        """Based on the new commanded pose, update the desired targets."""
        pose_command = self.EvalVectorInput(
            context, self.ee_pose_command_input_index).get_value().copy()
        self.position_d_target = pose_command[:3]

        # Ensure quaternion representation is consistent.
        last_orientation_d_target = self.orientation_d_target.wxyz()
        new_orientation_d_target = pose_command[3:]
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
        """TODO:
         - get inertia, jacobian, coriolis, gravity from Drake
        """
        # First get the current commanded pose and robot state.
        self._update_desired_targets(context)

        # tau_J_d = np.zeros(7) #self.EvalVectorInput(context, 1).get_value()
        # tau_J_d = self.EvalVectorInput(
        #     context, self.robot_actuation_input_index).get_value()
        x = self.EvalVectorInput(
            context, self.robot_state_input_index).get_value()
        q = x[:self.plant.num_positions()]
        v = x[self.plant.num_positions():]

        # Compute Drake's quantities like mass matrix, jacobian, coriolis,
        # gravity, etc.
        # TODO: can use CalcMassMatrix instead if speed is needed.
        self._set_controller_plant_state(q, v)
        mass_matrix = self.plant.CalcMassMatrixViaInverseDynamics(self.plant_context)
        M_inv = np.linalg.inv(mass_matrix)
        coriolis = self.plant.CalcBiasTerm(self.plant_context)
        gravity = self.plant.CalcGravityGeneralizedForces(self.plant_context)

        # TODO:  CalcJacobianSpatialVelocity I don't quite understand how it's
        # defined / different from CalcJacobianTranslationalVelocity.
        Jv_w_AB = self.plant.CalcJacobianAngularVelocity(
            self.plant_context,
            with_respect_to=JacobianWrtVariable.kV,
            frame_B=self.plant.GetFrameByName('end_effector_tip'),
            frame_A=self.plant.world_frame(),
            frame_E=self.plant.world_frame()
        )  # (3, 7)
        Jv_v_AB = self.plant.CalcJacobianTranslationalVelocity(
            self.plant_context,
            with_respect_to=JacobianWrtVariable.kV,
            frame_B=self.plant.GetFrameByName('end_effector_tip'),
            p_BoBi_B=np.zeros(3),
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
        quaternion_error = self.orientation_d.inverse().multiply(orientation)
        angle_axis_error = AngleAxis(quaternion=quaternion_error)
        orientation_error = -angle_axis_error.axis() * angle_axis_error.angle()
        error = np.concatenate((position_error, orientation_error))
        assert error.shape == (6,)

        # Compute control -- first feedforward.
        tau_wrench = jacobian.T @ np.concatenate((self.force_d, self.torque_d))

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
        # tau_d = tau_task + tau_nullspace + tau_wrench + coriolis - gravity
        tau_d = coriolis - gravity  # sanity check can make robot not move
        # tau_d = self._saturate_torque_rate(tau_d)

        # Set the output to the desired torques.
        assert tau_d.ndim == 1
        assert tau_d.shape[0] == self.plant.num_actuators() == 7
        output.SetFromVector(tau_d)

        # Do updates for the next iteration.
        self._end_of_calc_updates(tau_d)

    def _end_of_calc_updates(self, tau_d):
        self.tau_J_d = tau_d

        self.cartesian_stiffness = self.filter_params * \
            self.cartesian_stiffness_target + (1 - self.filter_params) * \
            self.cartesian_stiffness
        self.cartesian_damping = self.filter_params * \
            self.cartesian_damping_target + (1 - self.filter_params) * \
            self.cartesian_damping
        self.nullspace_stiffness = self.filter_params * \
            self.nullspace_stiffness_target + (1 - self.filter_params) * \
            self.nullspace_stiffness

        self.position_d = self.filter_params * \
            self.position_d_target + (1 - self.filter_params) * \
            self.position_d
        self.force_d = self.filter_params * \
            self.force_d_target + (1 - self.filter_params) * \
            self.force_d
        self.torque_d = self.filter_params * \
            self.torque_d_target + (1 - self.filter_params) * \
            self.torque_d

        self.torque_upper_limits = self.filter_params * \
            self.torque_upper_limits_target + (1 - self.filter_params) * \
            self.torque_upper_limits
        self.torque_lower_limits = self.filter_params * \
            self.torque_lower_limits_target + (1 - self.filter_params) * \
            self.torque_lower_limits
        
        self.cart_acc_upper_limits = self.filter_params * \
            self.cart_acc_upper_limits_target + (1 - self.filter_params) * \
            self.cart_acc_upper_limits
        self.cart_acc_lower_limits = self.filter_params * \
            self.cart_acc_lower_limits_target + (1 - self.filter_params) * \
            self.cart_acc_lower_limits
        
        self.cartesian_trans_acc_norm_limit = self.filter_params * \
            self.cartesian_trans_acc_norm_limit_target + (1 - self.filter_params) * \
            self.cartesian_trans_acc_norm_limit
        self.cartesian_rot_acc_norm_limit = self.filter_params * \
            self.cartesian_rot_acc_norm_limit_target + (1 - self.filter_params) * \
            self.cartesian_rot_acc_norm_limit
        
        # Special case to handle the quaternion.
        wxyz_d = self.orientation_d.wxyz()
        wxyz_d_target = self.orientation_d_target.wxyz()
        wxyz_d = self.filter_params * \
            wxyz_d_target + (1 - self.filter_params) * wxyz_d
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

# Zero out the trajectories.  TODO may need to keep track of this.
init_t = min(command_ts[0], franka_joint_ts[0])
command_ts -= init_t
franka_joint_ts -= init_t


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
commanded_pos_quat_traj = StackedTrajectory()
commanded_pos_quat_traj.Append(position_trajectory)
commanded_pos_quat_traj.Append(orientation_trajectory)

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


### Drake diagram ###
builder = DiagramBuilder()

# Add the robot to a simulation.
sim_mbp_config = MultibodyPlantConfig(time_step=SIM_TIME_STEP)
sim_plant, scene_graph = AddMultibodyPlant(sim_mbp_config, builder)
sim_parser = Parser(sim_plant)
sim_robot = sim_parser.AddModels(
    file_utils.franka_filepath(with_collision_geometry=True))
sim_plant.WeldFrames(sim_plant.world_frame(),
                     sim_plant.GetFrameByName('panda_link0'),
                     RigidTransform())

# Add the table (as a half space) to the simulation.
# TODO:  meshcat doesn't display halfspace geometries, so may want to use a box
# instead.
friction = CoulombFriction(1.0, 1.0)
sim_plant.RegisterCollisionGeometry(
    sim_plant.world_body(), RigidTransform(), HalfSpace(),
    'table', friction)
sim_plant.RegisterVisualGeometry(
    sim_plant.world_body(), RigidTransform(), HalfSpace(),
    'table', np.array([0.5, 0.5, 0.5, 1]))

sim_plant.Finalize()
sim_plant.set_name('sim_plant')

# Add the robot to the controller plant.
control_plant = MultibodyPlant(time_step=CONTROLLER_TIME_STEP)
control_parser = Parser(control_plant)
control_robot = control_parser.AddModels(
    file_utils.franka_filepath(with_collision_geometry=True))
control_plant.WeldFrames(control_plant.world_frame(),
                         control_plant.GetFrameByName('panda_link0'),
                         RigidTransform())
control_plant.Finalize()
control_plant.set_name('control_plant')

# Make the robot start at the recorded Franka configuration at the first time
# step of the commanded EE poses.  TODO better to do at start of simulation?
sim_plant.SetDefaultPositions(
    gt_joint_angle_traj.value(commanded_pos_quat_traj.start_time()))
control_plant.SetDefaultPositions(
    gt_joint_angle_traj.value(commanded_pos_quat_traj.start_time()))

# Add a meshcat visualizer.
meshcat = StartMeshcat()
MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)

# Visualize the plant.
plt.figure()
plot_graphviz(sim_plant.GetTopologyGraphvizString())
plt.plot(1)
plt.show(block=False)


# Add a trajectory source for the desired end effector pose.
traj_source = builder.AddSystem(TrajectorySource(commanded_pos_quat_traj))

# Add the controller.
control_params_file = 'robot_dynamics/franka_hw_controllers.yaml'
with open(control_params_file, 'r') as f:
    control_params = yaml.safe_load(f)
inv_dyn_controller = builder.AddSystem(
    InverseDynamicsController(control_plant, control_params))
inv_dyn_controller.set_name('inv_dyn_controller')

# Wire the diagram.
builder.Connect(traj_source.get_output_port(),
                inv_dyn_controller.get_ee_pose_command_input_port())
builder.Connect(sim_plant.get_state_output_port(),
                inv_dyn_controller.get_robot_state_input_port())
builder.Connect(inv_dyn_controller.get_joint_torque_output_port(),
                sim_plant.get_actuation_input_port())

diagram = builder.Build()
simulator = Simulator(diagram)
simulator.Initialize()
simulator.set_target_realtime_rate(1)

plt.figure()
plot_system_graphviz(diagram)
plt.plot(1)
plt.show(block=False)

breakpoint()


# Prepare to run a simulation.
sim_context = sim_plant.CreateDefaultContext()
sim_plant.SetPositions(sim_context, gt_joint_angle_traj.value(0))
inv_dyn_controller.StartFrom(
    joint_angles=gt_joint_angle_traj.value(0),
    joint_velocities=gt_joint_vel_traj.value(0),
    joint_torques=gt_joint_torque_traj.value(0)
)

simulator.AdvanceTo(1.0)

breakpoint()
