"""This file performs output conversions from BundleSDF trajectories to input
formats required by PLL."""

import click
import os
import os.path as op
import numpy as np
import torch
from tensordict.tensordict import TensorDictBase, TensorDict
import matplotlib
matplotlib.use('TkAgg') # if not used, might see the following error:
# QObject::moveToThread: Current thread (0x55e00d8db800) is not the object's thread (0x55e00db6f700).
# Cannot move to target thread (0x55e00d8db800)

# qt.qpa.plugin: Could not load the Qt platform plugin "xcb" in "/opt/conda/envs/py38/lib/python3.8/site-packages/cv2/qt/plugins" even though it was found.
# This application failed to start because no Qt platform plugin could be initialized. Reinstalling the application may fix this problem.
import matplotlib.pyplot as plt
import pdb
import trimesh

import file_utils
import math_utils

from overlay_videos import OverlayVideoGenerator
from vis_utils import SDFSliceViewer
from process_tagslam_trajectories import TagSLAMTrajectoryConverter


USED_ROBOT_JOINT_NAMES = [
    'panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4',
    'panda_joint5', 'panda_joint6', 'panda_joint7'
]


def build_pll_robot_object_tensordict(
        object_state: np.ndarray, robot_state: np.ndarray,
        robot_effort: np.ndarray) -> TensorDictBase:
    """Create a PLL input asset TensorDict object of shape (N,) containing keys:
        - 'object_state' (N, 13)
        - 'robot_state' (N, 14)
        - 'robot_effort' (N, 7)
    """
    assert object_state.ndim == robot_state.ndim == robot_effort.ndim == 2, \
        f'{object_state.shape=} != {robot_state.ndim=} != {robot_effort.ndim=}'
    assert object_state.shape[0] == robot_state.shape[0] == \
        robot_effort.shape[0], f'{object_state.shape[0]=} != ' + \
        f'{robot_state.shape[0]=} != {robot_effort.shape[0]=}'
    assert object_state.shape[1] == 13, f'{object_state.shape=}, expected ' + \
        '(N, 13)'
    assert robot_state.shape[1] == 14, f'{robot_state.shape=}, expected (N, 14)'
    assert robot_effort.shape[1] == 7, f'{robot_effort.shape=}, expected (N, 7)'

    n_horizon = object_state.shape[0]

    return TensorDict(
        {'object_state': torch.tensor(object_state),
         'robot_state': torch.tensor(robot_state),
         'robot_effort': torch.tensor(robot_effort)}, [n_horizon])


class TrajectoryConverterBundleSDFToPLL(TagSLAMTrajectoryConverter):
    """An extension of TagSLAMTrajectoryConverter that can process pose data
    from BundleSDF.  Can detect if there are corresponding TagSLAM poses, in
    which case these will be loaded.  Can convert between BundleSDF and TagSLAM
    origins (if both) and between camera and world frames.

    Since the ContactNets toss's start and end are defined based on indexing
    into the BundleSDF trajectories, the corresponding TagSLAM trajectories
    are selected to be the same number of frames most closely time-
    synchronized with the BundleSDF messages.

    Because of this, the poses may not actually be identical at the beginning
    of the toss, since BundleSDF and TagSLAM are forced to be identical at the
    beginning of the _BundleSDF trajectory_, not at the _ContactNets
    trajectory_.  The BundleSDF trajectories are usually ~10 seconds, start with
    the object unmoving on the table, and include the toss wind-up and
    execution.  The ContactNets trajectories are usually < 1 second and include
    only the object immediately after it has been released at the beginning of
    the toss.
    """
    def __init__(self, tracking_bundlesdf_id: str, nerf_bundlesdf_id: str,
                 start_toss: int, end_toss: int, object: str,
                 cycle_iteration: int, bsdf_only: bool, cam_trans: np.ndarray,
                 cam_rot_axis_angle: np.ndarray, frame_rate: int,
                 z_table: float, relative_start_frames: list,
                 relative_end_frames: list, start_ros_times: list,
                 plot: bool = False, bsdf_offset_frames: int = 1) -> None:
        """Prepare for processing pose data from TagSLAM and BundleSDF.

        Args:
            start_toss:  First toss index.
            end_toss:  Last toss index.
            relative_start_frames:  BundleSDF frame indices at which a PLL toss
                begins, represented as the index after the start time of each
                toss.
            relative_end_frames:  BundleSDF frame indices at which a PLL toss
                ends, represented as the index after the start time of each
                toss.
            start_ros_times:  ROS times at which each toss begins.
            toss_type:  The tossed object name.
            cycle_iteration:  The cycle iteration number of the BundleSDF/
                ContactNets cycle.
            bsdf_only:  Whether to only use BundleSDF data and not TagSLAM data.
            cam_trans (3,):  Position of the camera in world coordinates.
            cam_rot_axis_angle (3,):  Orientation of the camera is world
                coordinates in Rodrigues form.
            frame_rate:  The expected frame rate of the data.  This isn't used
                for processing but can be used to manually inspect that the
                data's timestamps result in a similar frame rate as expected.
            z_table:  The height of the table in world frame.  PLL expects the
                table plane to be at z=0, so this will get subtracted out.
            plot:  Whether to show the overlay plot of BundleSDF and TagSLAM
                trajectories.
        """
        self.start_toss = start_toss
        self.end_toss = end_toss
        self.object = object
        self.cycle_iteration = cycle_iteration
        self.bsdf_offset_frames = bsdf_offset_frames

        if (object in file_utils.TAGLESS_OBJECTS or \
            object.startswith('robot')) and not bsdf_only:
            bsdf_only = True
            print(f'Overriding to set {bsdf_only=} for tagless {object=}.')
        self.bsdf_only = bsdf_only

        self.tracking_bundlesdf_id = tracking_bundlesdf_id
        self.nerf_bundlesdf_id = nerf_bundlesdf_id

        self.plot = plot
        self.cam_trans = cam_trans
        self.cam_rot_axis_angle = cam_rot_axis_angle
        self.frame_rate = frame_rate
        self.z_table = z_table

        self.dataset = f'{self.object}_{self.start_toss}'
        self.dataset += f'-{self.end_toss}' if \
            self.start_toss != self.end_toss else ''

        self.bsdf_vision_asset = self.dataset

        # Robot interaction experiments will be converted slightly differently.
        self.has_robot_interactions = object.startswith('robot')

        self._set_up_directories()

        # Load the full pose trajectories from TagSLAM and BundleSDF.
        self._load_poses()

        # Compute the absolute start and end frames for each toss.
        self._get_absolute_frames(
            start_ros_times, relative_start_frames, relative_end_frames,
            self.bundlesdf_full_times
        )

    def _get_absolute_frames(self, start_ros_times, relative_start_frames,
                             relative_end_frames, full_times) -> None:
        """Compute the absolute start and end frames for each toss."""
        self.start_frames = math_utils.convert_relative_frames_to_absolute(
            relative_start_frames, full_times, start_ros_times)
        self.end_frames = math_utils.convert_relative_frames_to_absolute(
            relative_end_frames, full_times, start_ros_times)
        
        if self.bsdf_offset_frames > 1:
            assert self.bsdf_offset_frames < self.start_frames[0], \
              f'{self.bsdf_offset_frames=} is too large for ' \
              f'{self.start_frames[0]=}.'
            self.start_frames = self.start_frames - self.bsdf_offset_frames + 1
            self.end_frames = self.end_frames - self.bsdf_offset_frames + 1

    def _load_poses(self) -> None:
        """Load the timestamped poses reported from TagSLAM, BundleSDF, and the
        Franka, saving the results in attributes:
            - self.tagslam_full_times (only if not self.bsdf_only)
            - self.tagslam_full_poses (only if not self.bsdf_only)
            - self.bundlesdf_full_times
            - self.bundlesdf_b_full_poses (BundleSDF body origin)
            - self.bundlesdf_t_full_poses (TagSLAM body origin) (only if not
                self.bsdf_only)
            - self.keyframe_full_times
            - self.keyframe_b_full_poses (BundleSDF body origin)
            - self.keyframe_t_full_poses (TagSLAM body origin) (only if not
                self.bsdf_only)
            - self.robot_joint_angles (only if self.has_robot_interactions)
            - self.robot_joint_velocities (only if self.has_robot_interactions)
            - self.robot_joint_efforts (only if self.has_robot_interactions)

        The TagSLAM information and BundleSDF time information are generated by
        rosbag_processor.py's extract_synchronized_images_and_tagslam_poses.
        The TagSLAM time and poses are in the dataset directory's file
        synced_tagslam.txt.  The Franka joint states are in the dataset
        directory's files synced_joint_angles.txt, synced_joint_velocities.txt,
        and synced_joint_efforts.txt, which share the timestamps of the
        BundleSDF images/poses.  The BundleSDF information comes from the
        BundleSDF output directory's XXXX.txt files, which are generated by
        running BundleSDF and are timestamped according to the BundleSDF times,
        which come from the bundlesdf_timestamps.txt file.
        """
        super()._load_poses()

        # Handle robot states.
        if self.has_robot_interactions:
            self._load_robot_joint_states()

            print(f'Robot joint states:' + \
                  f'\n\t{self.robot_joint_angles.shape=}' + \
                  f'\n\t{self.robot_joint_velocities.shape=}' + \
                  f'\n\t{self.robot_joint_efforts.shape=}\n')

    def _load_robot_joint_states(self) -> None:
        """Load the robot joint angles, velocities, and efforts from the
        synchronized robot files."""
        robot_joint_angles = np.loadtxt(
            op.join(self.cnets_data_gen_dir, 'synced_joint_angles.txt'))
        robot_joint_velocities = np.loadtxt(
            op.join(self.cnets_data_gen_dir, 'synced_joint_velocities.txt'))
        robot_joint_efforts = np.loadtxt(
            op.join(self.cnets_data_gen_dir, 'synced_joint_efforts.txt'))

        if self.bsdf_offset_frames > 1:
            raise NotImplementedError(
                'Robot joint states are not offset by the BundleSDF offset ' + \
                'frames yet.')

        n_time_steps = robot_joint_angles.shape[0]
        n_time_steps_expected = self.bundlesdf_full_times.shape[0]
        assert n_time_steps == n_time_steps_expected, \
            f'{n_time_steps=} != {n_time_steps_expected=}'
        assert robot_joint_velocities.shape == robot_joint_angles.shape == \
            robot_joint_efforts.shape, f'{robot_joint_velocities.shape=} !=' + \
            f' {robot_joint_angles.shape=} != {robot_joint_efforts.shape=}'

        # Ensure the joint angles are in the same order as the robot joint
        # names used in the robot URDF.  This will also filter out any reported
        # joints not used in the robot URDF, e.g. the gripper joints.
        robot_joint_names = np.loadtxt(
            op.join(self.cnets_data_gen_dir, 'joint_names.txt'), dtype=str)
        indices = [np.where(robot_joint_names == name)[0][0] for name in \
                   USED_ROBOT_JOINT_NAMES]
        self.robot_joint_angles = robot_joint_angles[:, indices]
        self.robot_joint_velocities = robot_joint_velocities[:, indices]
        self.robot_joint_efforts = robot_joint_efforts[:, indices]

    def do_process(self) -> None:
        """Generate contactnets-format trajectories for BundleSDF outputs, also
        for TagSLAM outputs if not self.bsdf_only, and also for Franka outputs
        if self.has_robot_interactions.  This requires estimating velocities
        from differences in pose, converting everything to PLL format, and
        trimming the trajectories to view the autonomous dynamics during the
        toss only.  The PLL format is in:
	        [ quaternion  position  angular_velocity  linear_velocity ]
        where:
            - quaternion: 		[qw, qx, qy, qz]
            - position:  		[x, y, z] in meters
            - angular_velocity:	[wx, wy, wz] in rad/second in body frame
            - linear_velocity: 	[vx, vy, vz] in meters/second

        This method stores the full state trajectories in attributes:
            - self.bundlesdf_b_full_processed_states

        Then this method additionally stores the trimmed trajectories:
            - self.bundlesdf_b_toss_processed_states: List[np.ndarray(N, 13)]
            - self.bundlesdf_toss_times: List[np.ndarray(N,)]

        Additionally these, only if not self.bsdf_only:
            - self.tagslam_t_full_processed_states
            - self.tagslam_b_full_processed_states
            - self.bundlesdf_t_full_processed_states
            - self.tagslam_t_toss_processed_states: List[np.ndarray(N, 13)]
            - self.tagslam_b_toss_processed_states: List[np.ndarray(N, 13)]
            - self.bundlesdf_t_toss_processed_states: List[np.ndarray(N, 13)]
            - self.tagslam_toss_times: List[np.ndarray(N,)]

        And finally these, only if self.has_robot_interactions:
            - self.robot_full_processed_states: np.ndarray(N, 14) with order
                [q1, q2, q3, q4, q5, q6, q7, dq1, dq2, dq3, dq4, dq5, dq6, dq7].
            - self.robot_full_processed_efforts: np.ndarray(N, 7)
            - self.robot_toss_processed_states: List[np.ndarray(N, 14)]
            - self.robot_toss_processed_efforts: List[np.ndarray(N, 7)]
        """
        # Process BundleSDF data.
        q_bsdf_b, p_bsdf_b, w_bsdf_b, v_bsdf_b = self._process_poses(
            self.bundlesdf_b_full_poses, self.bundlesdf_full_times)
        q_key_b, p_key_b, w_key_b, v_key_b = self._process_poses(
            self.keyframe_b_full_poses, self.keyframe_full_times,
            filter_rot=False, filter_pos=False, filter_lin_vel=False,
            filter_ang_vel=False)

        self.bundlesdf_b_full_processed_states = np.concatenate(
            (q_bsdf_b, p_bsdf_b, w_bsdf_b, v_bsdf_b), axis=1)
        self.keyframe_b_full_processed_states = np.concatenate(
            (q_key_b, p_key_b, w_key_b, v_key_b), axis=1)

        # Process TagSLAM data.
        if not self.bsdf_only:
            q_ts, p_ts, w_ts, v_ts = self._process_poses(
                self.tagslam_t_full_poses, self.tagslam_full_times)
            q_ts_b, p_ts_b, w_ts_b, v_ts_b = self._process_poses(
                self.tagslam_b_full_poses, self.tagslam_full_times)
            q_bsdf_t, p_bsdf_t, w_bsdf_t, v_bsdf_t = self._process_poses(
                self.bundlesdf_t_full_poses, self.bundlesdf_full_times)
            q_key_t, p_key_t, w_key_t, v_key_t = self._process_poses(
                self.keyframe_t_full_poses, self.keyframe_full_times,
                filter_rot=False, filter_pos=False, filter_lin_vel=False,
                filter_ang_vel=False)

            self.tagslam_t_full_processed_states = np.concatenate(
                (q_ts, p_ts, w_ts, v_ts), axis=1)
            self.tagslam_b_full_processed_states = np.concatenate(
                (q_ts_b, p_ts_b, w_ts_b, v_ts_b), axis=1)
            self.bundlesdf_t_full_processed_states = np.concatenate(
                (q_bsdf_t, p_bsdf_t, w_bsdf_t, v_bsdf_t), axis=1)

            self.keyframe_t_full_processed_states = np.concatenate(
                (q_key_t, p_key_t, w_key_t, v_key_t), axis=1)

        # Process robot data.  There's no real processing to do here since the
        # Franka does its own filtering/processing before reporting these
        # values.
        if self.has_robot_interactions:
            self.robot_full_processed_states = np.concatenate(
                (self.robot_joint_angles, self.robot_joint_velocities), axis=1)
            self.robot_full_processed_efforts = self.robot_joint_efforts

        # Store trimmed trajectories for toss only.
        self._trim_processed_trajectories()

        # Print information about the trimmed trajectories.
        for i in range(len(self.start_frames)):
            toss_i = i + self.start_toss
            print(f'\n=================== TOSS {toss_i} ===================')
            if not self.bsdf_only:
                tagslam_toss_dts = np.mean(
                    self.tagslam_toss_times[i][1:] - \
                    self.tagslam_toss_times[i][:-1])
                print(f'TagSLAM toss {toss_i} trajectory information:' + \
                    f'\n\t{self.tagslam_t_toss_processed_states[i].shape=}' + \
                    f'\n\t{self.tagslam_toss_times[i][0]=}' + \
                    f'\n\tAverage frame rate (toss): {1/tagslam_toss_dts}\n')

            bsdf_toss_dts = np.mean(self.bundlesdf_toss_times[i][1:] - \
                                    self.bundlesdf_toss_times[i][:-1])
            print(f'BundleSDF toss {toss_i} trajectory information:' + \
                f'\n\t{self.bundlesdf_b_toss_processed_states[i].shape=}' + \
                f'\n\t{self.bundlesdf_toss_times[i][0]=}' + \
                f'\n\tAverage frame rate (toss): {1/bsdf_toss_dts}\n')

            print(f'BundleSDF keyframe toss {toss_i} trajectory ' + \
                    f'information:' + \
                f'\n\t{self.keyframe_b_toss_processed_states[i].shape=}')
            if self.keyframe_toss_times[i].shape[0] > 0:
                print(f'\t{self.keyframe_toss_times[i][0]=}\n')
            else:
                print(f'\tNo keyframes for toss {toss_i}.\n')

    def _trim_processed_trajectories(self) -> None:
        """After trajectories are already processed, store trimmed versions of
        them corresponding to autonomous dynamics throughout a toss.  This
        stores the following attributes:
            - self.bundlesdf_b_toss_processed_states: List[np.ndarray(N, 13)]
            - self.bundlesdf_toss_times: List[np.ndarray(N,)]
            - self.keyframe_b_toss_processed_states: List[np.ndarray(M, 13)]
            - self.keyframe_toss_times: List[np.ndarray(M,)]

        And additionally these, only if not self.bsdf_only:
            - self.tagslam_t_toss_processed_states: List[np.ndarray(N, 13)]
            - self.tagslam_b_toss_processed_states: List[np.ndarray(N, 13)]
            - self.bundlesdf_t_toss_processed_states: List[np.ndarray(N, 13)]
            - self.keyframe_t_toss_processed_states: List[np.ndarray(M, 13)]
            - self.tagslam_toss_times: List[np.ndarray(N,)]

        And finally these, only if self.has_robot_interactions:
            - self.robot_toss_processed_states: List[np.ndarray(N, 14)]
            - self.robot_toss_processed_efforts: List[np.ndarray(N, 7)]
        """
        # Process BundleSDF data.
        self.bundlesdf_b_toss_processed_states = []
        self.bundlesdf_toss_times = []
        self.keyframe_b_toss_processed_states = []
        self.keyframe_toss_times = []

        # Be preparted to detect if the tosses have full keyframe data.
        self.has_full_keyframe_tosses = True

        for i in range(len(self.start_frames)):
            # Need to do one less than provided start and end frames because the
            # annotated start_frame and end_frame in config.yaml were annotated
            # based on the 1-indexed images, but the below needs to use 0-
            # indexing.
            b_start = self.start_frames[i] - 1
            b_end = self.end_frames[i] - 1
            self.bundlesdf_b_toss_processed_states.append(
                self.bundlesdf_b_full_processed_states[b_start:b_end])
            self.bundlesdf_toss_times.append(
                self.bundlesdf_full_times[b_start:b_end])

            # Keep any keyframe data in the same time range as the toss.
            key_start = np.argmin(
                (self.keyframe_full_times - self.bundlesdf_toss_times[i][0])**2)
            key_end = np.argmin(
                (self.keyframe_full_times - self.bundlesdf_toss_times[i][-1]
                 )**2)
            self.keyframe_b_toss_processed_states.append(
                self.keyframe_b_full_processed_states[key_start:key_end])
            self.keyframe_toss_times.append(
                self.keyframe_full_times[key_start:key_end])

            # Need all tosses to have continuous keyframe data to be able to
            # export them as trajectories.
            if self.keyframe_toss_times[i].shape != \
                self.bundlesdf_toss_times[i].shape:
                print(f'Keyframe data for toss {i+self.start_toss} is not ' + \
                      'continuous.  Will not export keyframe trajectories.')
                self.has_full_keyframe_tosses = False

        # Trim any TagSLAM-related data.
        if not self.bsdf_only:
            self.bundlesdf_t_toss_processed_states = []
            self.tagslam_t_toss_processed_states = []
            self.tagslam_b_toss_processed_states = []
            self.tagslam_toss_times = []
            self.keyframe_t_toss_processed_states = []

            for i in range(len(self.start_frames)):
                # Need to do one less than provided start and end frames because
                # the annotated start_frame and end_frame in config.yaml were
                # annotated based on the 1-indexed images, but the below needs
                # to use 0-indexing.
                b_start = self.start_frames[i] - 1
                b_end = self.end_frames[i] - 1
                self.bundlesdf_t_toss_processed_states.append(
                    self.bundlesdf_t_full_processed_states[b_start:b_end])

                # Find the start and end frames that are most synchronized in
                # time with those pre-selected for BundleSDF trajectories.
                t_start = np.argmin(
                    (self.tagslam_full_times - self.bundlesdf_toss_times[i][0]
                     )**2)
                t_end = t_start + (b_end - b_start)
                self.tagslam_t_toss_processed_states.append(
                    self.tagslam_t_full_processed_states[t_start:t_end])
                self.tagslam_b_toss_processed_states.append(
                    self.tagslam_b_full_processed_states[t_start:t_end])
                self.tagslam_toss_times.append(
                    self.tagslam_full_times[t_start:t_end])

                # Keep any keyframe data in the same time range as the toss.
                key_start = np.argmin(
                    (self.keyframe_full_times - self.bundlesdf_toss_times[i][0]
                     )**2)
                key_end = np.argmin(
                    (self.keyframe_full_times - self.bundlesdf_toss_times[i][-1]
                    )**2)
                self.keyframe_t_toss_processed_states.append(
                    self.keyframe_t_full_processed_states[key_start:key_end])

        # Trim any robot-related data.
        if self.has_robot_interactions:
            self.robot_toss_processed_states = []
            self.robot_toss_processed_efforts = []

            for i in range(len(self.start_frames)):
                # Need to do one less than provided start and end frames because
                # the annotated start_frame and end_frame in config.yaml were
                # annotated based on the 1-indexed images, but the below needs
                # to use 0-indexing.
                b_start = self.start_frames[i] - 1
                b_end = self.end_frames[i] - 1
                self.robot_toss_processed_states.append(
                    self.robot_full_processed_states[b_start:b_end])
                self.robot_toss_processed_efforts.append(
                    self.robot_full_processed_efforts[b_start:b_end])

    def plot_trajectory(self, full_trajectory: bool = True) -> None:
        """Visualize the BundleSDF and TagSLAM trajectories overlayed on a set
        of plots."""
        def do_plot(q_ts_t, p_ts_t, w_ts_t, v_ts_t, t_ts_t,
                    q_ts_b, p_ts_b, w_ts_b, v_ts_b, t_ts_b,
                    q_bsdf_t, p_bsdf_t, w_bsdf_t, v_bsdf_t,
                    q_bsdf_b, p_bsdf_b, w_bsdf_b, v_bsdf_b, t_bsdf,
                    q_key_t, p_key_t, _w_key_t, _v_key_t,
                    q_key_b, p_key_b, _w_key_b, _v_key_b, t_key,
                    toss_num, full):
            b_only = self.bsdf_only

            first_t = min(t_bsdf) if b_only else min(min(t_ts_t), min(t_bsdf))
            if not b_only:
                t_ts_t -= first_t
            t_bsdf -= first_t
            t_key -= first_t

            def make_columns_in_row_share_y(ax, row, up_to=4):
                for i in range(up_to):
                    ax[row, i].sharey(ax[row, 0])

            fig, ax = plt.subplots(4, 4, figsize=(15, 15), sharex='all',
                                sharey='none')

            make_columns_in_row_share_y(ax, 0, up_to=3)
            make_columns_in_row_share_y(ax, 1)
            ax[1, 0].set_ylim([-1, 1])
            make_columns_in_row_share_y(ax, 2, up_to=3)
            make_columns_in_row_share_y(ax, 3, up_to=3)

            if not b_only:
                ax[0, 0].plot(t_ts_t, p_ts_t[:, 0], label='TagSLAM, T Origin',
                              color='orange')
                ax[0, 0].plot(t_ts_b, p_ts_b[:, 0], label='TagSLAM, B Origin',
                              linestyle='--', color='orange')
                ax[0, 0].plot(t_bsdf, p_bsdf_t[:, 0],
                              label='BundleSDF, T Origin', color='blue')
            ax[0, 0].plot(t_bsdf, p_bsdf_b[:, 0], label='BundleSDF, B Origin',
                          linestyle='--', color='blue')
            if not b_only:
                ax[0, 0].scatter(t_key, p_key_t[:, 0], c='orange', s=20,
                                label='Keyframes, T Origin')
            ax[0, 0].scatter(t_key, p_key_b[:, 0], c='green', s=20,
                             label='Keyframes, B Origin')
            ax[0, 0].set_title('X Position')
            if not b_only:
                ax[0, 1].plot(t_ts_t, p_ts_t[:, 1], label='TagSLAM, T Origin',
                              color='orange')
                ax[0, 1].plot(t_ts_b, p_ts_b[:, 1], label='TagSLAM, B Origin',
                              linestyle='--', color='orange')
                ax[0, 1].plot(t_bsdf, p_bsdf_t[:, 1],
                              label='BundleSDF, T Origin', color='blue')
            ax[0, 1].plot(t_bsdf, p_bsdf_b[:, 1], label='BundleSDF, B Origin',
                          linestyle='--', color='blue')
            if not b_only:
                ax[0, 1].scatter(t_key, p_key_t[:, 1], c='orange', s=20,
                                 label='Keyframes, T Origin')
            ax[0, 1].scatter(t_key, p_key_b[:, 1], c='green', s=20,
                             label='Keyframes, B Origin')
            ax[0, 1].set_title('Y Position')
            if not b_only:
                ax[0, 2].plot(t_ts_t, p_ts_t[:, 2], label='TagSLAM, T Origin',
                              color='orange')
                ax[0, 2].plot(t_ts_b, p_ts_b[:, 2], label='TagSLAM, B Origin',
                              linestyle='--', color='orange')
                ax[0, 2].plot(t_bsdf, p_bsdf_t[:, 2],
                              label='BundleSDF, T Origin', color='blue')
            ax[0, 2].plot(t_bsdf, p_bsdf_b[:, 2], label='BundleSDF, B Origin',
                          linestyle='--', color='blue')
            if not b_only:
                ax[0, 2].scatter(t_key, p_key_t[:, 2], c='orange', s=20,
                                 label='Keyframes, T Origin')
            ax[0, 2].scatter(t_key, p_key_b[:, 2], c='green', s=20,
                             label='Keyframes, B Origin')
            ax[0, 2].set_title('Z Position')

            if not b_only:
                ax[1, 0].plot(t_ts_t, q_ts_t[:, 0], label='TagSLAM, T Origin',
                              color='orange')
                ax[1, 0].plot(t_ts_b, q_ts_b[:, 0], label='TagSLAM, B Origin',
                              linestyle='--', color='orange')
                ax[1, 0].plot(t_bsdf, q_bsdf_t[:, 0],
                              label='BundleSDF, T Origin', color='blue')
            ax[1, 0].plot(t_bsdf, q_bsdf_b[:, 0], label='BundleSDF, B Origin',
                          linestyle='--', color='blue')
            if len(t_key) > 0:
                if not b_only:
                    ax[1, 0].scatter(t_key, q_key_t[:, 0], c='orange', s=20,
                                    label='Keyframes, T Origin')
                ax[1, 0].scatter(t_key, q_key_b[:, 0], c='green', s=20,
                                label='Keyframes, B Origin')
            ax[1, 0].set_title('W Quaternion')
            if not b_only:
                ax[1, 1].plot(t_ts_t, q_ts_t[:, 1], label='TagSLAM, T Origin',
                              color='orange')
                ax[1, 1].plot(t_ts_b, q_ts_b[:, 1], label='TagSLAM, B Origin',
                              linestyle='--', color='orange')
                ax[1, 1].plot(t_bsdf, q_bsdf_t[:, 1],
                              label='BundleSDF, T Origin', color='blue')
            ax[1, 1].plot(t_bsdf, q_bsdf_b[:, 1], label='BundleSDF, B Origin',
                          linestyle='--', color='blue')
            if len(t_key) > 0:
                if not b_only:
                    ax[1, 1].scatter(t_key, q_key_t[:, 1], c='orange', s=20,
                                    label='Keyframes, T Origin')
                ax[1, 1].scatter(t_key, q_key_b[:, 1], c='green', s=20,
                                label='Keyframes, B Origin')
            ax[1, 1].set_title('X Quaternion')
            if not b_only:
                ax[1, 2].plot(t_ts_t, q_ts_t[:, 2], label='TagSLAM, T Origin',
                              color='orange')
                ax[1, 2].plot(t_ts_b, q_ts_b[:, 2], label='TagSLAM, B Origin',
                              linestyle='--', color='orange')
                ax[1, 2].plot(t_bsdf, q_bsdf_t[:, 2],
                              label='BundleSDF, T Origin', color='blue')
            ax[1, 2].plot(t_bsdf, q_bsdf_b[:, 2], label='BundleSDF, B Origin',
                          linestyle='--', color='blue')
            if len(t_key) > 0:
                if not b_only:
                    ax[1, 2].scatter(t_key, q_key_t[:, 2], c='orange', s=20,
                                    label='Keyframes, T Origin')
                ax[1, 2].scatter(t_key, q_key_b[:, 2], c='green', s=20,
                                label='Keyframes, B Origin')
            ax[1, 2].set_title('Y Quaternion')
            if not b_only:
                ax[1, 3].plot(t_ts_t, q_ts_t[:, 3], label='TagSLAM, T Origin',
                              color='orange')
                ax[1, 3].plot(t_ts_b, q_ts_b[:, 3], label='TagSLAM, B Origin',
                              linestyle='--', color='orange')
                ax[1, 3].plot(t_bsdf, q_bsdf_t[:, 3],
                              label='BundleSDF, T Origin', color='blue')
            ax[1, 3].plot(t_bsdf, q_bsdf_b[:, 3], label='BundleSDF, B Origin',
                          linestyle='--', color='blue')
            if len(t_key) > 0:
                if not b_only:
                    ax[1, 3].scatter(t_key, q_key_t[:, 3], c='orange', s=20,
                                    label='Keyframes, T Origin')
                ax[1, 3].scatter(t_key, q_key_b[:, 3], c='green', s=20,
                                label='Keyframes, B Origin')
            ax[1, 3].set_title('Z Quaternion')

            if not b_only:
                ax[2, 0].plot(t_ts_t, v_ts_t[:, 0], label='TagSLAM, T Origin',
                              color='orange')
                ax[2, 0].plot(t_ts_b, v_ts_b[:, 0], label='TagSLAM, B Origin',
                              linestyle='--', color='orange')
                ax[2, 0].plot(t_bsdf, v_bsdf_t[:, 0],
                              label='BundleSDF, T Origin', color='blue')
            ax[2, 0].plot(t_bsdf, v_bsdf_b[:, 0], label='BundleSDF, B Origin',
                          linestyle='--', color='blue')
            ax[2, 0].set_title('X Velocity')
            if not b_only:
                ax[2, 1].plot(t_ts_t, v_ts_t[:, 1], label='TagSLAM, T Origin',
                              color='orange')
                ax[2, 1].plot(t_ts_b, v_ts_b[:, 1], label='TagSLAM, B Origin',
                              linestyle='--', color='orange')
                ax[2, 1].plot(t_bsdf, v_bsdf_t[:, 1],
                              label='BundleSDF, T Origin', color='blue')
            ax[2, 1].plot(t_bsdf, v_bsdf_b[:, 1], label='BundleSDF, B Origin',
                          linestyle='--', color='blue')
            ax[2, 1].set_title('Y Velocity')
            if not b_only:
                ax[2, 2].plot(t_ts_t, v_ts_t[:, 2], label='TagSLAM, T Origin',
                              color='orange')
                ax[2, 2].plot(t_ts_b, v_ts_b[:, 2], label='TagSLAM, B Origin',
                              linestyle='--', color='orange')
                ax[2, 2].plot(t_bsdf, v_bsdf_t[:, 2],
                              label='BundleSDF, T Origin', color='blue')
            ax[2, 2].plot(t_bsdf, v_bsdf_b[:, 2], label='BundleSDF, B Origin',
                          linestyle='--', color='blue')
            ax[2, 2].set_title('Z Velocity')

            if not b_only:
                ax[3, 0].plot(t_ts_t, w_ts_t[:, 0], label='TagSLAM, T Origin',
                              color='orange')
                ax[3, 0].plot(t_ts_b, w_ts_b[:, 0], label='TagSLAM, B Origin',
                              linestyle='--', color='orange')
                ax[3, 0].plot(t_bsdf, w_bsdf_t[:, 0],
                              label='BundleSDF, T Origin', color='blue')
            ax[3, 0].plot(t_bsdf, w_bsdf_b[:, 0], label='BundleSDF, B Origin',
                          linestyle='--', color='blue')
            ax[3, 0].set_title('X Angular Velocity')
            if not b_only:
                ax[3, 1].plot(t_ts_t, w_ts_t[:, 1], label='TagSLAM, T Origin',
                              color='orange')
                ax[3, 1].plot(t_ts_b, w_ts_b[:, 1], label='TagSLAM, B Origin',
                              linestyle='--', color='orange')
                ax[3, 1].plot(t_bsdf, w_bsdf_t[:, 1],
                              label='BundleSDF, T Origin', color='blue')
            ax[3, 1].plot(t_bsdf, w_bsdf_b[:, 1], label='BundleSDF, B Origin',
                          linestyle='--', color='blue')
            ax[3, 1].set_title('Y Angular Velocity')
            if not b_only:
                ax[3, 2].plot(t_ts_t, w_ts_t[:, 2], label='TagSLAM, T Origin',
                              color='orange')
                ax[3, 2].plot(t_ts_b, w_ts_b[:, 2], label='TagSLAM, B Origin',
                              linestyle='--', color='orange')
                ax[3, 2].plot(t_bsdf, w_bsdf_t[:, 2],
                              label='BundleSDF, T Origin', color='blue')
            ax[3, 2].plot(t_bsdf, w_bsdf_b[:, 2], label='BundleSDF, B Origin',
                          linestyle='--', color='blue')
            ax[3, 2].set_title('Z Angular Velocity')

            # Include an orientation error plot in the empty upper right.
            if not b_only:
                q_errors = math_utils.quaternion_errors(q_ts_t, q_bsdf_t)
                q_errors *= 180 / np.pi
                ax[0, 3].plot(t_ts_t, q_errors, color='r',
                            label='TagSLAM-to-BundleSDF T')

                # Include the orientation error for the keyframes.
                if len(t_key) > 0:
                    tagslam_time_idx = [np.argmin((t_ts_t-t)**2) for t in t_key]
                    q_key_errors = math_utils.quaternion_errors(
                        q_ts_t[tagslam_time_idx], q_key_t)
                    q_key_errors *= 180 / np.pi
                    ax[0, 3].scatter(t_key, q_key_errors, color='r', s=20,
                                    label='TagSLAM-to-Keyframes T')
                ax[0, 3].set_title('Orientation Error')
                ax[0, 3].legend()

            ax[0, 0].set_ylabel('Position [m]')
            ax[0, 3].set_ylabel('Angular error [deg]')
            ax[1, 0].set_ylabel('Orientation')
            ax[2, 0].set_ylabel('Linear Velocity [m/s]')
            ax[3, 0].set_ylabel('Angular Velocity [rad/s]')
            ax[3, 0].set_xlabel('Time from first pose [s]')
            ax[3, 1].set_xlabel('Time from first pose [s]')
            ax[3, 2].set_xlabel('Time from first pose [s]')
            ax[3, 3].set_xlabel('Time from first pose [s]')

            ax[0, 1].tick_params(labelleft=False)
            ax[0, 2].tick_params(labelleft=False)
            ax[1, 1].tick_params(labelleft=False)
            ax[1, 2].tick_params(labelleft=False)
            ax[1, 3].tick_params(labelleft=False)
            ax[2, 1].tick_params(labelleft=False)
            ax[2, 2].tick_params(labelleft=False)
            ax[2, 3].tick_params(labelleft=False)
            ax[3, 1].tick_params(labelleft=False)
            ax[3, 2].tick_params(labelleft=False)
            ax[3, 3].tick_params(labelleft=False)

            handles, labels = ax[1,2].get_legend_handles_labels()
            fig.legend(handles, labels, loc='lower center')

            fig.suptitle(title)

            # Save the figure in the PLL assets directory where the trajectory
            # data goes.
            pll_bsdf_asset_dir = file_utils.contactnets_input_dir_bundlesdf(
                dataset=self.dataset, iteration=self.cycle_iteration,
                bundlesdf_id=self.tracking_bundlesdf_id, full=full_trajectory)
            plot_name = '' if full else f'toss_{toss_num}_'
            plot_name += f'{self.tracking_bundlesdf_id}.png'
            plt.savefig(op.join(pll_bsdf_asset_dir, plot_name))
            print(f'Saved plot to {op.join(pll_bsdf_asset_dir, plot_name)}.\n')

            # Also save the figure in the results inspection directory.
            dir, prefix = file_utils.inspection_trajectory_plots_dir_and_prefix(
                dataset=self.dataset,
                tracking_bundlesdf_id=self.tracking_bundlesdf_id,
                nerf_bundlesdf_id=self.nerf_bundlesdf_id,
                cycle_iteration=self.cycle_iteration
            )
            plot_name = f'{prefix}.png' if full else \
                f'{prefix}_toss_{toss_num}.png'
            plt.savefig(op.join(dir, plot_name))
            print(f'Saved plot to {op.join(dir, plot_name)}.\n')

            if self.plot:
                plt.show()
            plt.close()

        if full_trajectory:
            if not self.bsdf_only:
                q_ts_t = self.tagslam_t_full_processed_states[:, 0:4]
                p_ts_t = self.tagslam_t_full_processed_states[:, 4:7]
                w_ts_t = self.tagslam_t_full_processed_states[:, 7:10]
                v_ts_t = self.tagslam_t_full_processed_states[:, 10:13]
                t_ts_t = self.tagslam_full_times
                q_ts_b = self.tagslam_b_full_processed_states[:, 0:4]
                p_ts_b = self.tagslam_b_full_processed_states[:, 4:7]
                w_ts_b = self.tagslam_b_full_processed_states[:, 7:10]
                v_ts_b = self.tagslam_b_full_processed_states[:, 10:13]
                t_ts_b = self.tagslam_full_times
            else:
                q_ts_t = None
                p_ts_t = None
                w_ts_t = None
                v_ts_t = None
                t_ts_t = None
                q_ts_b = None
                p_ts_b = None
                w_ts_b = None
                v_ts_b = None
                t_ts_b = None

            q_bsdf_b = self.bundlesdf_b_full_processed_states[:, 0:4]
            p_bsdf_b = self.bundlesdf_b_full_processed_states[:, 4:7]
            w_bsdf_b = self.bundlesdf_b_full_processed_states[:, 7:10]
            v_bsdf_b = self.bundlesdf_b_full_processed_states[:, 10:13]
            if not self.bsdf_only:
                q_bsdf_t = self.bundlesdf_t_full_processed_states[:, 0:4]
                p_bsdf_t = self.bundlesdf_t_full_processed_states[:, 4:7]
                w_bsdf_t = self.bundlesdf_t_full_processed_states[:, 7:10]
                v_bsdf_t = self.bundlesdf_t_full_processed_states[:, 10:13]
            else:
                q_bsdf_t = None
                p_bsdf_t = None
                w_bsdf_t = None
                v_bsdf_t = None
            t_bsdf = self.bundlesdf_full_times

            q_key_b = self.keyframe_b_full_processed_states[:, 0:4]
            p_key_b = self.keyframe_b_full_processed_states[:, 4:7]
            w_key_b = self.keyframe_b_full_processed_states[:, 7:10]
            v_key_b = self.keyframe_b_full_processed_states[:, 10:13]
            if not self.bsdf_only:
                q_key_t = self.keyframe_t_full_processed_states[:, 0:4]
                p_key_t = self.keyframe_t_full_processed_states[:, 4:7]
                w_key_t = self.keyframe_t_full_processed_states[:, 7:10]
                v_key_t = self.keyframe_t_full_processed_states[:, 10:13]
            else:
                q_key_t = None
                p_key_t = None
                w_key_t = None
                v_key_t = None
            t_key = self.keyframe_full_times

            title='Full Trajectory Results'
            do_plot(q_ts_t, p_ts_t, w_ts_t, v_ts_t, t_ts_t,
                    q_ts_b, p_ts_b, w_ts_b, v_ts_b, t_ts_b,
                    q_bsdf_t, p_bsdf_t, w_bsdf_t, v_bsdf_t,
                    q_bsdf_b, p_bsdf_b, w_bsdf_b, v_bsdf_b, t_bsdf,
                    q_key_t, p_key_t, w_key_t, v_key_t,
                    q_key_b, p_key_b, w_key_b, v_key_b, t_key,
                    toss_num=None, full=full_trajectory)

        else:
            for i in range(len(self.bundlesdf_toss_times)):
                if not self.bsdf_only:
                    q_ts_t = self.tagslam_t_toss_processed_states[i][:, 0:4]
                    p_ts_t = self.tagslam_t_toss_processed_states[i][:, 4:7]
                    w_ts_t = self.tagslam_t_toss_processed_states[i][:, 7:10]
                    v_ts_t = self.tagslam_t_toss_processed_states[i][:, 10:13]
                    t_ts_t = self.tagslam_toss_times[i]
                    q_ts_b = self.tagslam_b_toss_processed_states[i][:, 0:4]
                    p_ts_b = self.tagslam_b_toss_processed_states[i][:, 4:7]
                    w_ts_b = self.tagslam_b_toss_processed_states[i][:, 7:10]
                    v_ts_b = self.tagslam_b_toss_processed_states[i][:, 10:13]
                    t_ts_b = self.tagslam_toss_times[i]
                else:
                    q_ts_t = None
                    p_ts_t = None
                    w_ts_t = None
                    v_ts_t = None
                    t_ts_t = None
                    q_ts_b = None
                    p_ts_b = None
                    w_ts_b = None
                    v_ts_b = None
                    t_ts_b = None

                q_bsdf_b = self.bundlesdf_b_toss_processed_states[i][:, 0:4]
                p_bsdf_b = self.bundlesdf_b_toss_processed_states[i][:, 4:7]
                w_bsdf_b = self.bundlesdf_b_toss_processed_states[i][:, 7:10]
                v_bsdf_b = self.bundlesdf_b_toss_processed_states[i][:, 10:13]
                if not self.bsdf_only:
                    q_bsdf_t = self.bundlesdf_t_toss_processed_states[i][:, 0:4]
                    p_bsdf_t = self.bundlesdf_t_toss_processed_states[i][:, 4:7]
                    w_bsdf_t = self.bundlesdf_t_toss_processed_states[i][:, 7:10]
                    v_bsdf_t = self.bundlesdf_t_toss_processed_states[i][:, 10:13]
                else:
                    q_bsdf_t = None
                    p_bsdf_t = None
                    w_bsdf_t = None
                    v_bsdf_t = None
                t_bsdf = self.bundlesdf_toss_times[i]

                q_key_b = self.keyframe_b_toss_processed_states[i][:, 0:4]
                p_key_b = self.keyframe_b_toss_processed_states[i][:, 4:7]
                w_key_b = self.keyframe_b_toss_processed_states[i][:, 7:10]
                v_key_b = self.keyframe_b_toss_processed_states[i][:, 10:13]
                if not self.bsdf_only:
                    q_key_t = self.keyframe_t_toss_processed_states[i][:, 0:4]
                    p_key_t = self.keyframe_t_toss_processed_states[i][:, 4:7]
                    w_key_t = self.keyframe_t_toss_processed_states[i][:, 7:10]
                    v_key_t = self.keyframe_t_toss_processed_states[i][:, 10:13]
                else:
                    q_key_t = None
                    p_key_t = None
                    w_key_t = None
                    v_key_t = None
                t_key = self.keyframe_toss_times[i]

                title=f'Toss {i + self.start_toss} Trajectory Results'
                do_plot(q_ts_t, p_ts_t, w_ts_t, v_ts_t, t_ts_t,
                        q_ts_b, p_ts_b, w_ts_b, v_ts_b, t_ts_b,
                        q_bsdf_t, p_bsdf_t, w_bsdf_t, v_bsdf_t,
                        q_bsdf_b, p_bsdf_b, w_bsdf_b, v_bsdf_b, t_bsdf,
                        q_key_t, p_key_t, w_key_t, v_key_t,
                        q_key_b, p_key_b, w_key_b, v_key_b, t_key,
                        toss_num=i+self.start_toss, full=full_trajectory)

    def save_data(self, save_tagslam: bool = False,
                  save_bundlesdf: bool = False) -> None:
        """Stores data as .pt files in the ContactNets input directory.  For the
        BundleSDF trajectories, stores those of the BundleSDF origin.  The
        TagSLAM origin trajectories are just for comparison with TagSLAM
        tracking results.

        If this is a robot experiment, each .pt file is a TensorDict of shape
        (N,) containing keys "object_state" (N, 13) and "robot_state" (N, 14).
        For non-robot experiments, each .pt file is a Tensor of shape (N, 13).
        """
        if self.bsdf_only:  save_tagslam = False

        toss_filenames = [f'{toss_i}.pt' for toss_i in range(
            self.start_toss, self.end_toss+1)]

        print('Saving files summary:')

        # Handle robot experiments, which require saving TensorDicts as .pt
        # files.
        if self.has_robot_interactions:
            # Save full trajectory.
            full_bundlesdf_dir = file_utils.contactnets_input_dir_bundlesdf(
                dataset=self.dataset, iteration=self.cycle_iteration,
                bundlesdf_id=self.tracking_bundlesdf_id, full=True)
            traj_filename = f'{self.tracking_bundlesdf_id}.pt'

            traj_dict = build_pll_robot_object_tensordict(
                object_state=self.bundlesdf_b_full_processed_states,
                robot_state=self.robot_full_processed_states,
                robot_effort=self.robot_full_processed_efforts)

            torch.save(traj_dict, op.join(full_bundlesdf_dir, traj_filename))
            print(f'\t{op.join(full_bundlesdf_dir, traj_filename)}')

            # Do toss trajectories.
            toss_bundlesdf_dir = file_utils.contactnets_input_dir_bundlesdf(
                dataset=self.dataset, iteration=self.cycle_iteration,
                bundlesdf_id=self.tracking_bundlesdf_id, full=False)
            for i in range(len(toss_filenames)):
                traj_dict = build_pll_robot_object_tensordict(
                    object_state=self.bundlesdf_b_toss_processed_states[i],
                    robot_state=self.robot_toss_processed_states[i],
                    robot_effort=self.robot_toss_processed_efforts[i])

                torch.save(
                    traj_dict, op.join(toss_bundlesdf_dir, toss_filenames[i]))
                print(f'\t{op.join(toss_bundlesdf_dir, toss_filenames[i])}')

            # Do keyframe toss trajectories, if they are continuous.
            if self.has_full_keyframe_tosses:
                key_bundlesdf_dir = file_utils.contactnets_input_dir_keyframes(
                    dataset=self.dataset, iteration=self.cycle_iteration,
                    bundlesdf_id=self.tracking_bundlesdf_id, bsdf=True
                )
                for i in range(len(toss_filenames)):
                    traj_dict = build_pll_robot_object_tensordict(
                        object_state=self.keyframe_b_toss_processed_states[i],
                        robot_state=self.robot_toss_processed_states[i])
                    torch.save(traj_dict,
                               op.join(key_bundlesdf_dir, toss_filenames[i]))
                    print(f'\t{op.join(key_bundlesdf_dir, toss_filenames[i])}')

        # If not a robot experiment, save BundleSDF if desired.
        elif save_bundlesdf:
            # Save full trajectory.
            full_bundlesdf_dir = file_utils.contactnets_input_dir_bundlesdf(
                dataset=self.dataset, iteration=self.cycle_iteration,
                bundlesdf_id=self.tracking_bundlesdf_id, full=True)
            traj_filename = f'{self.tracking_bundlesdf_id}.pt'
            torch.save(
                torch.tensor(self.bundlesdf_b_full_processed_states),
                op.join(full_bundlesdf_dir, traj_filename))
            print(f'\t{op.join(full_bundlesdf_dir, traj_filename)}')

            # Do toss trajectories.
            toss_bundlesdf_dir = file_utils.contactnets_input_dir_bundlesdf(
                dataset=self.dataset, iteration=self.cycle_iteration,
                bundlesdf_id=self.tracking_bundlesdf_id, full=False)
            for i in range(len(toss_filenames)):
                torch.save(
                    torch.tensor(self.bundlesdf_b_toss_processed_states[i]),
                    op.join(toss_bundlesdf_dir, toss_filenames[i]))
                print(f'\t{op.join(toss_bundlesdf_dir, toss_filenames[i])}')

            # Do keyframe toss trajectories, if they are continuous.
            if self.has_full_keyframe_tosses:
                key_bundlesdf_dir = file_utils.contactnets_input_dir_keyframes(
                    dataset=self.dataset, iteration=self.cycle_iteration,
                    bundlesdf_id=self.tracking_bundlesdf_id, bsdf=True
                )
                for i in range(len(toss_filenames)):
                    torch.save(
                        torch.tensor(self.keyframe_b_toss_processed_states[i]),
                        op.join(key_bundlesdf_dir, toss_filenames[i]))
                    print(f'\t{op.join(key_bundlesdf_dir, toss_filenames[i])}')

        # Regardless of if this was a robot experiment or not, save TagSLAM-
        # related information if desired.
        if save_tagslam:
            # Save full trajectory.
            full_tagslam_dir = file_utils.contactnets_input_dir_tagslam(
                dataset=self.dataset, full=True)
            torch.save(
                # torch.tensor(self.tagslam_t_full_processed_states),
                torch.tensor(self.tagslam_b_full_processed_states),
                op.join(full_tagslam_dir, 'tagslam.pt'))
            print(f"\t{op.join(full_tagslam_dir, 'tagslam.pt')}")

            # Do toss trajectories.
            toss_tagslam_dir = file_utils.contactnets_input_dir_tagslam(
                dataset=self.dataset, full=False)
            for i in range(len(toss_filenames)):
                torch.save(
                    # torch.tensor(self.tagslam_t_toss_processed_states[i]),
                    torch.tensor(self.tagslam_b_toss_processed_states[i]),
                    op.join(toss_tagslam_dir, toss_filenames[i]))
                print(f'\t{op.join(toss_tagslam_dir, toss_filenames[i])}')

            # Do keyframe toss trajectories, if they are continuous.
            if self.has_full_keyframe_tosses:
                key_tagslam_dir = file_utils.contactnets_input_dir_keyframes(
                    dataset=self.dataset, iteration=self.cycle_iteration,
                    bundlesdf_id=self.tracking_bundlesdf_id, bsdf=False
                )
                for i in range(len(toss_filenames)):
                    torch.save(
                        # torch.tensor(self.keyframe_t_toss_processed_states[i]),
                        torch.tensor(self.keyframe_b_toss_processed_states[i]),
                        op.join(key_tagslam_dir, toss_filenames[i]))
                    print(f'\t{op.join(key_tagslam_dir, toss_filenames[i])}')

        print(f'\n')


class GeometryConverterBundleSDFToPLL:
    """Class for processing shape data from BundleSDF.

    TODO:
        - There could be two BundleSDF IDs, one for the tracking experiment and
            one for the associated geometry reconstruction experiment, which
            would have results within the tracking experiment.  For now, this
            code only grabs the NeRF results from the tracking experiment's
            associated NeRF run.
    """
    def __init__(self, tracking_bundlesdf_id: str, nerf_bundlesdf_id: str,
                 start_toss: int, end_toss: int, object: str,
                 cycle_iteration: int, plot: bool):
        # Get the BundleSDF results directory where we can find the meshes.
        vision_asset = f'{object}_{start_toss}'
        vision_asset += f'-{end_toss}' if start_toss != end_toss else ''

        # Set up directories.
        self._set_up_directories(
            vision_asset, cycle_iteration, tracking_bundlesdf_id,
            nerf_bundlesdf_id)

        # Load the BundleSDF results' mesh and compute its convex hull.  This
        # mesh is already represented in world units about the BundleSDF
        # tracking origin.
        self.mesh_bsdf = trimesh.load(
            op.join(self.nerf_results_dir, 'textured_mesh.obj'), force='mesh')
        self.mesh_bsdf_hull = self.mesh_bsdf.convex_hull
        self.plot = plot

        self.vision_asset = vision_asset
        self.tracking_bundlesdf_id = tracking_bundlesdf_id
        self.nerf_bundlesdf_id = nerf_bundlesdf_id
        self.cycle_iteration = cycle_iteration

    def _set_up_directories(
            self, vision_asset: str, cycle_iteration: int,
            tracking_bundlesdf_id: str, nerf_bundlesdf_id: str) -> None:
        """Given the vision asset, cycle iteration, and BundleSDF ID, loads the
        following attributes:
            - self.nerf_results_dir
            - self.geometry_for_pll_dir
        """
        self.nerf_results_dir = file_utils.bundlesdf_nerf_results_dir(
            dataset=vision_asset, cycle_iteration=cycle_iteration,
            tracking_bundlesdf_id=tracking_bundlesdf_id,
            nerf_bundlesdf_id=nerf_bundlesdf_id
        )
        self.geometry_for_pll_dir = file_utils.contactnets_input_geometry_dir(
            vision_asset, cycle_iteration, nerf_bundlesdf_id)

    def plot_mesh_and_hull_points(self):
        mesh = self.mesh_bsdf
        hull = self.mesh_bsdf_hull

        verts1 = mesh.vertices
        verts2 = hull.vertices
        vert_norms2 = hull.vertex_normals

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(verts1[:, 0], verts1[:, 1], verts1[:, 2], s=0.1,
                   label='Mesh vertices')
        ax.scatter(verts2[:, 0], verts2[:, 1], verts2[:, 2], s=10, color='r',
                   label='Convex hull vertices')
        prefix = [''] + ['_']*(len(verts2)-1)
        for i in range(len(verts2)):
            ax.quiver(*verts2[i], *vert_norms2[i]/50, color='r',
                      label=prefix[i]+'Vertex normals', zorder=1.5)
        plt.legend()
        plt.show()

    def plot_support_directions_and_points(self, pts, dirs):
        mesh = self.mesh_bsdf
        hull = self.mesh_bsdf_hull

        mesh_verts = mesh.vertices
        hull_verts = hull.vertices

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(mesh_verts[:, 0], mesh_verts[:, 1], mesh_verts[:, 2], s=0.1,
                   label='Mesh vertices')
        ax.scatter(hull_verts[:, 0], hull_verts[:, 1], hull_verts[:, 2], s=10,
                   color='r', label='Convex hull vertices')
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=12,
                   color='orange', label='Support points')

        prefix = [''] + ['_']*(len(pts)-1)
        for i in range(len(pts)):
            ax.quiver(*pts[i], *dirs[i]/25, color='orange',
                      label=prefix[i]+'Support directions', zorder=1.5)
        plt.legend()
        plt.show()

    def query_support_directions_to_get_points(self):
        # Use the convex hull.
        hull = self.mesh_bsdf_hull
        hull_points = hull.vertices

        # Get some evenly-spaced query directions from PLL.
        support_directions = math_utils.get_deep_support_query_directions()

        # Compute the support point for every query direction, selecting out of
        # the convex hull vertices.
        support_points = torch.zeros_like(support_directions)
        support_scalars = torch.zeros((support_directions.shape[0],))

        for i in range(support_directions.shape[0]):
            dir = support_directions[i].reshape(1, 3)
            dots = torch.sum(dir * hull_points, dim=1)
            support_points[i, :] = torch.Tensor(hull_points[torch.argmax(dots)])
            support_scalars[i] = torch.max(dots)

        return support_points, support_scalars, support_directions

    def _copy_meshes(self):
        """Copy the mesh_cleaned.obj file over to the cnets-data-generation/
        consolidated_results/meshes/ directory for manual inspection.

        For later subsequent PLL runs, create a mesh.obj convex hull file out of
        the BundleSDF world dimensions file.  This involves copying over the
        convex hull in .obj format for PLL to show geometry comparisons.  This
        .obj file requires vertex, vertex normals, and face definitions where
        the faces' associated vertices are annotated with their associated
        vertex normals explicitly (otherwise error on Mengti's lab computer).
        """
        # Copy mesh_cleaned.obj from the BundleSDF NeRF results to the
        # inspection folder.
        original_mesh_path = op.join(self.nerf_results_dir, 'mesh_cleaned.obj')
        new_mesh_path = file_utils.inspection_mesh_filepath(
            dataset=self.vision_asset,
            tracking_bundlesdf_id=self.tracking_bundlesdf_id,
            nerf_bundlesdf_id=self.nerf_bundlesdf_id,
            cycle_iteration=self.cycle_iteration
        )
        os.system(f'cp {original_mesh_path} {new_mesh_path}')
        print(f'Copied mesh from BundleSDF to {new_mesh_path}.\n')

        # Write the custom convex hull mesh file for PLL later.
        new_filepath = op.join(self.geometry_for_pll_dir, 'mesh.obj')
        with open(new_filepath, 'w') as f:
            f.write(f'# Vertices\n')
            for vertex in self.mesh_bsdf_hull.vertices:
                f.write(f'v {vertex[0]} {vertex[1]} {vertex[2]}\n')

            f.write(f'\n# Vertex normals\n')
            for normal in self.mesh_bsdf_hull.vertex_normals:
                f.write(f'vn {normal[0]} {normal[1]} {normal[2]}\n')

            f.write(f'\n# Faces:  vertex index // vertex normal index\n')
            for face in self.mesh_bsdf_hull.faces:
                # +1 because obj files use 1-indexing but trimesh uses 0.
                # This is of format v_i//vn_i, which for us are always the same.
                f.write(f'f {face[0]+1}//{face[0]+1} ' + \
                        f'{face[1]+1}//{face[1]+1} {face[2]+1}//{face[2]+1}\n')
        print(f'Wrote convex hull from BundleSDF as mesh at {new_filepath}.')

    def process_and_save(self):
        """Process the data."""
        # Query different directions and get the support points.
        support_points, support_scalars, support_directions = \
            self.query_support_directions_to_get_points()
        if self.plot:
            self.plot_support_directions_and_points(
                support_points, support_directions)

        # Write these as tensors to PLL's input geometry folder.
        torch.save(
            support_points,
            op.join(self.geometry_for_pll_dir, 'support_points.pt'))
        torch.save(
            support_scalars,
            op.join(self.geometry_for_pll_dir, 'support_scalars.pt'))
        torch.save(
            support_directions,
            op.join(self.geometry_for_pll_dir, 'support_directions.pt'))

        print(f'Saved {support_points.shape=} and {support_directions.shape=}.')

        # Write the mesh files to inspection directories.
        self._copy_meshes()




#######################################################################
@click.command()
@click.option('--vision-asset',
              type=str,
              default=None,
              help="directory of the asset folder e.g. cube_2; encodes " + \
                "system and tosses.")
@click.option('--bundlesdf-id',
              type=str,
              default=None,
              help="what BundleSDF run ID associated with pose outputs to use.")
@click.option('--nerf-bundlesdf-id',
              type=str,
              default=None,
              help="what BundleSDF run ID associated with NeRF outputs to use.")
@click.option('--cycle-iteration',
              type=int,
              default=1,
              help="BundleSDF iteration number (can't choose 0 since that " + \
                "means use TagSLAM poses).")
@click.option('--bsdf-only',
              is_flag=True,
              help="whether to generate just BundleSDF-related data.")
@click.option('--make-videos/--skip-videos',
              type=bool,
              default=True,
              help="whether to make overlay and slice videos.")
@click.option('--remote/--local',
              default=False,
              help="whether to run on a remote server.")
@click.option('--show/--noshow',
              default=True,
              help="whether to show the plots.")
@click.option('--offset-frames',
              type=int,
              default=1,
              help="how many frames to offset the BundleSDF poses by.")

def main_command(vision_asset: str, bundlesdf_id: str, nerf_bundlesdf_id: str,
                 cycle_iteration: int, bsdf_only: bool, make_videos: bool,
                 remote: bool, show: bool, offset_frames: int):
    # First decode the system and start/end tosses from the provided asset
    # directory.
    assert cycle_iteration > 0, f'Invalid cycle iteration: {cycle_iteration}.'
    assert '_' in vision_asset, f'Invalid asset directory: {vision_asset}.'
    object = '_'.join(vision_asset.split('_')[:-1])

    start_toss = int(vision_asset.split('_')[-1].split('-')[0])
    end_toss = start_toss if '-' not in vision_asset else \
        int(vision_asset.split('-')[1])
    assert start_toss <= end_toss, f'Invalid toss range: {start_toss} ' + \
            f'-{end_toss} inferred from {vision_asset=}.'

    # Decode the BundleSDF run ID.
    if bundlesdf_id[:13] != 'bundlesdf_id_':
        bundlesdf_id = f'bundlesdf_id_{bundlesdf_id}'
    if nerf_bundlesdf_id is None:
        nerf_bundlesdf_id = bundlesdf_id
    elif nerf_bundlesdf_id[:13] != 'bundlesdf_id_':
        nerf_bundlesdf_id = f'bundlesdf_id_{nerf_bundlesdf_id}'

    # Automatically detect if BundleSDF-only is necessary based on if the object
    # is a tagless one.
    object = '_'.join(vision_asset.split('_')[:-1])
    if object in file_utils.TAGLESS_OBJECTS or object.startswith('robot'):
        bsdf_only = True
        print(f'Automatically setting {bsdf_only=} for tagless {object=}.')

    print(f'Processing toss {vision_asset} from BundleSDF tracking run ID ' + \
            f'{bundlesdf_id} and NeRF run ID {nerf_bundlesdf_id}.\n')

    # Get the camera extrinsics.
    cam_trans, cam_rot_axis_angle = file_utils.load_camera_extrinsics(object)

    # Get the table height.  Use the average if using multiple tosses.
    table_heights = np.array([
        file_utils.load_table_z_height(object, toss) for toss in
        range(start_toss, end_toss+1)
    ])
    z_table = np.mean(table_heights)

    # Start times are for the start and end of a BundleSDF trajectory, which
    # starts with the object unmoving on the table, includes the toss wind-up
    # and execution, and ends with the object unmoving on the table again.
    start_ros_times = np.array([file_utils.load_toss_time_from_yaml(
        object, toss_i, 'start_time', as_ros_time=True) for toss_i in range(
            start_toss, end_toss+1)])

    # Start/end frames are the indices of the longer BundleSDF trajectories that
    # correspond to the ContactNets trajectories, which include only the
    # autonomous dynamics of the object dropping under gravity and colliding
    # with the table.
    relative_start_frames = np.array([file_utils.load_field_from_yaml(
        object, toss_i, 'start_frame') for toss_i in range(
            start_toss, end_toss+1)])
    relative_end_frames = np.array([file_utils.load_field_from_yaml(
        object, toss_i, 'end_frame') for toss_i in range(
            start_toss, end_toss+1)])

    # Do the geometry conversion.
    geom_converter = GeometryConverterBundleSDFToPLL(
        tracking_bundlesdf_id=bundlesdf_id, nerf_bundlesdf_id=nerf_bundlesdf_id,
        start_toss=start_toss, end_toss=end_toss, object=object,
        cycle_iteration=cycle_iteration, plot=show
    )
    geom_converter.process_and_save()

    # Do the trajectory conversion.
    traj_converter = TrajectoryConverterBundleSDFToPLL(
        tracking_bundlesdf_id=bundlesdf_id,
        nerf_bundlesdf_id=nerf_bundlesdf_id,
        relative_start_frames=relative_start_frames,
        relative_end_frames=relative_end_frames,
        start_ros_times=start_ros_times, start_toss=start_toss,
        end_toss=end_toss, object=object, cycle_iteration=cycle_iteration,
        bsdf_only=bsdf_only, cam_trans=cam_trans,
        cam_rot_axis_angle=cam_rot_axis_angle, frame_rate=30, z_table=z_table,
        plot=show, bsdf_offset_frames=offset_frames,
    )
    traj_converter.do_process()
    traj_converter.plot_trajectory(full_trajectory=True)
    traj_converter.plot_trajectory(full_trajectory=False)
    save_tagslam = not bsdf_only and offset_frames == 1
    traj_converter.save_data(save_tagslam=save_tagslam, save_bundlesdf=True)

    # Create an overlay video.
    if make_videos:
        overlay_generator = OverlayVideoGenerator(
            vision_asset=vision_asset, tracking_bundlesdf_id=bundlesdf_id,
            nerf_bundlesdf_id=nerf_bundlesdf_id, bsdf_only=bsdf_only,
            cycle_iteration=cycle_iteration, remote=remote,
            bsdf_offset_frames=offset_frames,
        )
        overlay_generator.make_overlay_video()
        overlay_generator.make_optimized_keyframe_overlay_images()
    else:
        print('Skipping overlay video creation.')

    # Generate the SDF slice images.
    if make_videos:
        sdf_slice_generator = SDFSliceViewer(
            vision_asset=vision_asset, tracking_bundlesdf_id=bundlesdf_id,
            nerf_bundlesdf_id=nerf_bundlesdf_id, cycle_iteration=cycle_iteration,
            remote=remote
        )
        sdf_slice_generator.visualization()
    else:
        print('Skipping slice video creation.')


if __name__ == '__main__':
    main_command()  # pylint: disable=no-value-for-parameter
