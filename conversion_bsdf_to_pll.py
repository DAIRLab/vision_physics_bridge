"""This file performs output conversions from BundleSDF trajectories to input
formats required by PLL.

TODO:  This functionality has only been checked for single toss datasets, e.g.
cube_2.  Still to be tested on multi-toss experiments.
"""

# import argparse
import click
import os.path as op
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from scipy import signal
from pyquaternion import Quaternion
import matplotlib.pyplot as plt
import pdb
import math
from typing import Tuple, List

import file_utils
import math_utils


FILTER_ORIENTATIONS = True
FILTER_POSITIONS = True
FILTER_LINEAR_VELOCITIES = True
FILTER_ANGULAR_VELOCITIES = True

FILTER_TYPE = 'median'              # Can be median or savgol.
SAVGOL_FILTER_WINDOW_LENGTH = 15
SAVGOL_FILTER_POLYORDER = 3
MEDIAN_FILTER_KERNEL_SIZE = 3

YAML_PATH = file_utils.PROCESSING_YAML_FILE


def rotvecfix(rv):
    for i in range(rv.shape[0]-1):
        rvi = rv[i,:]
        rvip1 = rv[i+1,:]
        theta = np.linalg.norm(rvip1)
        if theta > 0.0:
            rnew = rvip1*(1 - 2*math.pi/theta)
            if np.linalg.norm(rvi - rnew) < np.linalg.norm(rvi - rvip1):
                rv[i+1,:] = rnew
    return rv

def smooth_positions(positions, window_size=5):
    """
    Smooths positions using a moving average.
    :param positions: Nx3 array of positions.
    :param window_size: Size of the moving average window.
    :return: Smoothed Nx3 array of positions.
    """
    smoothed_positions = np.zeros_like(positions)
    half_window = window_size // 2

    for i in range(positions.shape[0]):
        start_idx = max(0, i - half_window)
        end_idx = min(positions.shape[0], i + half_window)
        smoothed_positions[i] = np.mean(positions[start_idx:end_idx], axis=0)

    return smoothed_positions

def smooth_quaternions_pyquat(quats, alpha=0.5):
    """
    Smooth quaternions using Slerp with pyquaternion.
    :param quats: Nx4 array of quaternions, xyzw
    :param alpha: Interpolation factor (0.0 <= alpha <= 1.0).
    :return: Smoothed Nx4 array of quaternions.
    """
    quats = math_utils.xyzw2wxyz(quats)
    smoothed_quats = np.zeros_like(quats)
    smoothed_quats[0] = quats[0]

    for i in range(1, len(quats)):
        q0 = Quaternion(quats[i-1])
        q1 = Quaternion(quats[i])
        smoothed = Quaternion.slerp(q0, q1, alpha)
        smoothed_quats[i] = [smoothed.w, smoothed.x, smoothed.y, smoothed.z]
    smoothed_quats = math_utils.wxyz2xyzw(smoothed_quats)
    return smoothed_quats #xyzw


class ConverterBundleSDFToPLL:
    """Class for processing pose data from BundleSDF.  Can load corresponding
    poses from TagSLAM, convert between BundleSDF and TagSLAM origins and
    between camera and world frames.

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
    def __init__(self, bundlesdf_id: str, pll_id: str, start_toss: int,
                 end_toss: int, object: str, cycle_iteration: int,
                 cam_trans: np.ndarray, cam_rot_axis_angle: np.ndarray,
                 frame_rate: int, z_table: float, relative_start_frames: list,
                 relative_end_frames: list, start_ros_times: list,
                 plot: bool = False) -> None:
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
            iteration_num:  The cycle iteration number of the BundleSDF/
                ContactNets cycle.
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
        # self.start_frames = start_frames
        # self.end_frames = end_frames
        self.start_toss = start_toss
        self.end_toss = end_toss
        self.object = object
        self.iteration_num = cycle_iteration

        self.bundlesdf_id = bundlesdf_id
        self.pll_id = pll_id

        self.plot = plot
        self.cam_trans = cam_trans
        self.cam_rot_axis_angle = cam_rot_axis_angle
        self.frame_rate = frame_rate
        self.z_table = z_table

        self.dataset = f'{self.object}_{self.start_toss}'
        self.dataset += f'-{self.end_toss}' if \
            self.start_toss != self.end_toss else ''

        self._set_up_directories()

        # Load the full pose trajectories from TagSLAM and BundleSDF.
        self._load_poses()

        # Compute the absolute start and end frames for each toss.
        self._get_absolute_frames(start_ros_times, relative_start_frames,
                                  relative_end_frames)

    def _get_absolute_frames(self, start_ros_times, relative_start_frames,
                             relative_end_frames) -> None:
        """Compute the absolute start and end frames for each toss."""
        self.start_frames = math_utils.convert_relative_frames_to_absolute(
            relative_start_frames, self.bundlesdf_full_times, start_ros_times)
        self.end_frames = math_utils.convert_relative_frames_to_absolute(
            relative_end_frames, self.bundlesdf_full_times, start_ros_times)

    def _set_up_directories(self) -> None:
        """Given the stored object and start/end toss numbers, loads the
        following attributes:
            - self.tagslam_dir
            - self.bundlesdf_dir
            - self.annotated_dir
        """
        self.tagslam_dir = file_utils.tagslam_pose_dir(
            self.dataset, check_exists=True)
        self.bundlesdf_dir = file_utils.bundlesdf_pose_dir(
            self.dataset, cycle_iteration=self.iteration_num,
            bundlesdf_id=self.bundlesdf_id)
        self.annotated_dir = file_utils.bundlesdf_annotated_poses_dir(
            self.dataset)
        
    def _load_poses(self) -> None:
        """Load the timestamped poses reported from TagSLAM and BundleSDF,
        saving the results in attributes:
            - self.tagslam_full_times
            - self.tagslam_full_poses
            - self.bundlesdf_full_times
            - self.bundlesdf_full_poses
        
        The TagSLAM information and BundleSDF time information are generated by
        rosbag_processor.py's extract_synchronized_images_and_tagslam_poses.  
        The TagSLAM time and poses are in the ground truth pose directory's file
        tagslam.txt.  The BundleSDF information comes from the BundleSDF output
        directory's XXXX.txt files, which are generated by running BundleSDF and
        are timestamped according to the BundleSDF times, which come from
        the bundlesdf_timestamps.txt file.

        Inputs:
            bsdf_times (N,)
        """
        self._load_tagslam_poses()
        self._load_bundlesdf_poses()

        print(f'\nTarget frame rate: {self.frame_rate}\n')

        tagslam_full_dts = np.mean(
            self.tagslam_full_times[1:] - self.tagslam_full_times[:-1]
        )
        print(f'TagSLAM full trajectory information:' + \
              f'\n\t{self.tagslam_full_poses.shape=}' + \
              f'\n\t{self.tagslam_full_times[0]=}' + \
              f'\n\tAverage frame rate (full): {1/tagslam_full_dts}\n')

        bsdf_full_dts = np.mean(
            self.bundlesdf_full_times[1:] - self.bundlesdf_full_times[:-1]
        )
        print(f'BundleSDF full trajectory information:' + \
              f'\n\t{self.bundlesdf_full_poses.shape=}' + \
              f'\n\t{self.bundlesdf_full_times[0]=}' + \
              f'\n\tAverage frame rate (full): {1/bsdf_full_dts}\n')
        
    def _load_tagslam_poses(self) -> None:
        """Load all the poses reported by TagSLAM.  These are in world
        coordinates of the TagSLAM body origin with the following ordering:
            [x, y, z, qx, qy, qz, qw]
        """
        tagslam_data = np.loadtxt(op.join(self.tagslam_dir, 'tagslam.txt'))

        self.tagslam_full_times = tagslam_data[:, 0]
        self.tagslam_full_poses = tagslam_data[:, 1:]

    def _load_bundlesdf_poses(self) -> None:
        """Load all the poses reported by BundleSDF.  These are in world
        coordinates of the **TagSLAM body origin (converted from BundleSDF
        origin in camera coordinates via
        math_utils.transform_bundletrack_output) with the following ordering:
            [x, y, z, qx, qy, qz, qw]
        """
        bundlesdf_poses = []
        bundlesdf_times = np.loadtxt(
            op.join(op.dirname(self.tagslam_dir), 'bundlesdf_timestamps.txt'))

        # Add 1 for range bounds because BundleSDF poses are 1-indexed.
        for i in range(1, bundlesdf_times.shape[0] + 1):
            trans_mat = np.loadtxt(op.join(self.bundlesdf_dir, "%04i.txt" % i))
            trans_mat = \
                math_utils.transform_bundletrack_origin_to_tagslam_origin(
                    pred_pose=trans_mat,
                    bsdf_output_pose_dir=self.bundlesdf_dir,
                    annotated_poses_dir=self.annotated_dir,
                    translation=self.cam_trans,
                    axis_vec=self.cam_rot_axis_angle,
                    to_world=True
                )
            pos_quat = math_utils.trans_mat_to_pos_quat(trans_mat).reshape(7)
            bundlesdf_poses.append(pos_quat)

        self.bundlesdf_full_poses = np.array(bundlesdf_poses)
        self.bundlesdf_full_times = np.array(bundlesdf_times)

    def _estimate_linear_velocities(
            self, ts, ps, filter=FILTER_LINEAR_VELOCITIES) -> np.ndarray:
        """From times and positions, estimate the linear velocities at each time
        step.

        Args:
            ts (N,)
            ps (N, 3)
            filter:  whether or not to filter the result.

        Outputs:
            vs (N, 3)
        """
        # Do some input checking.
        assert ts.ndim == 1, f'{ts.shape=} not of expected size (N,).'
        assert ps.shape == (ts.shape[0], 3), f'{ps.shape=} not of expected ' + \
            f'size ({ts.shape[0]}, 3).'

        # Compute time differences.
        t_start = ts[0]
        t = ts - t_start
        tdiff = np.tile((t[1:] - t[:-1]).reshape([-1, 1]), [1, 3])

        # Compute velocities based on differences in position over time step.
        pdiff = ps[1:, :] - ps[:-1, :]
        vs = pdiff / tdiff

        # Repeat first row so that \delta p = v' \delta t.
        vs = np.vstack((vs[[0], :], vs))

        if filter:
            if FILTER_TYPE == 'savgol':
                vs = signal.savgol_filter(
                    vs, window_length=SAVGOL_FILTER_WINDOW_LENGTH,
                    polyorder=SAVGOL_FILTER_POLYORDER, axis=0)
            elif FILTER_TYPE == 'median':
                for i in range(3):
                    vs[:, i] = signal.medfilt(
                        vs[:, i], kernel_size=MEDIAN_FILTER_KERNEL_SIZE)
            else:
                raise NotImplementedError

        return vs

    def _estimate_angular_velocities(
            self, ts, qs, filter=FILTER_ANGULAR_VELOCITIES) -> np.ndarray:
        """From times and orientations, estimate the angular velocities at each
        time step.  These angular velocities are reported in body frame.

        Args:
            ts (N,)
            qs (N, 4)
            filter:  whether or not to filter the result.

        Outputs:
            ws (N, 3)
        """
        # Do some input checking.
        assert ts.ndim == 1, f'{ts.shape=} not of expected size (N,).'
        assert qs.shape == (ts.shape[0], 4), f'{qs.shape=} not of expected ' + \
            f'size ({ts.shape[0]}, 4).'

        # Compute time differences.
        t_start = ts[0]
        t = ts - t_start
        tdiff = np.tile((t[1:] - t[:-1]).reshape([-1, 1]), [1, 3])

        # Compute velocities based on differences in orientation over time step.
        rot_t = Rotation.from_quat(qs)
        rot_rel = rot_t[:-1].inv() * rot_t[1:]
        rel_vecs = rot_rel.as_rotvec()
        ws = rel_vecs / tdiff     # (N, 3)
        
        # Repeat first row so that \delta q = w' \delta t.
        ws = np.vstack((ws[[0], :], ws))

        if filter:
            if FILTER_TYPE == 'savgol':
                ws = signal.savgol_filter(
                    ws, window_length=SAVGOL_FILTER_WINDOW_LENGTH,
                    polyorder=SAVGOL_FILTER_POLYORDER, axis=0)
            elif FILTER_TYPE == 'median':
                for i in range(3):
                    ws[:, i] = signal.medfilt(
                        ws[:, i], kernel_size=MEDIAN_FILTER_KERNEL_SIZE)
            else:
                raise NotImplementedError

        return ws

    def _filter_quaternions(self, quat):
        """Filter and process reported quaternions."""
        # Do some input checking.
        assert quat.shape[1] == 4, f'{quat.shape=} not of expected size (N, 4).'
        assert quat.ndim == 2, f'{quat.shape=} not of expected size (N, 4).'

        quat /= np.linalg.norm(quat, axis=1).reshape(-1,1)
        rot_t = Rotation.from_quat(quat)

        # Fix and filter quaternions.
        rvecs = rotvecfix(rot_t.as_rotvec())
        for i in range(3):
            # Always use medfilt no matter FILTER_TYPE.
            rvecs[:, i] = signal.medfilt(
                rvecs[:, i], kernel_size=MEDIAN_FILTER_KERNEL_SIZE)
        rot_t = rot_t.from_rotvec(rvecs)

        return rot_t.as_quat()
    
    def _process_poses(
            self, poses, times, filter_rot=FILTER_ORIENTATIONS,
            filter_pos=FILTER_POSITIONS,
            filter_lin_vel=FILTER_LINEAR_VELOCITIES,
            filter_ang_vel=FILTER_ANGULAR_VELOCITIES,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate contactnets-format trajectory components.  This requires
        estimating velocities from differences in pose and converting everything
        to PLL format.  The PLL format is in:
	        [ quaternion  position  angular_velocity  linear_velocity ]
        where:
            - quaternion: 		[qw, qx, qy, qz]
            - position:  		[x, y, z] in meters
            - angular_velocity:	[wx, wy, wz] in rad/second in body frame
            - linear_velocity: 	[vx, vy, vz] in meters/second

        Args:
            poses (N, 7):  Poses in order of [x, y, z, qx, qy, qz, qw].
            times (N,)

        Outputs:
            qs_wxyz (N, 4)
            ps (N, 3)
            ws (N, 3)
            vs (N, 3)
        """
        # Do some input checking.
        assert times.ndim == 1, f'{times.shape=} not of expected size (N,).'
        assert poses.shape == (times.shape[0], 7), f'{poses.shape=} not of ' + \
            f'expected size ({times.shape[0]}, 7).'

        # Split the poses into positions and quaternions.
        ps = poses[:, :3]
        qs_xyzw = poses[:, 3:7]
        t = times

        # Do orientation filtering if desired.
        if filter_rot:
            qs_xyzw = self._filter_quaternions(qs_xyzw)

        # Do position filtering if desired.
        if filter_pos:
            if FILTER_TYPE == 'savgol':
                ps = signal.savgol_filter(
                    ps, window_length=SAVGOL_FILTER_WINDOW_LENGTH,
                    polyorder=SAVGOL_FILTER_POLYORDER, axis=0)
            elif FILTER_TYPE == 'median':
                for i in range(3):
                    ps[:, i] = signal.medfilt(
                        ps[:, i], kernel_size=MEDIAN_FILTER_KERNEL_SIZE)
            else:
                raise NotImplementedError

        # Subtract out table height so z=0 corresponds to being on the table.
        ps[:, 2] -= self.z_table

        # Calculate derivatives.
        vs = self._estimate_linear_velocities(ts=t, ps=ps,
                                              filter=filter_lin_vel)
        ws = self._estimate_angular_velocities(ts=t, qs=qs_xyzw,
                                              filter=filter_ang_vel)

        # Package into PLL format.
        qs_wxyz = math_utils.xyzw2wxyz(qs_xyzw)
        return qs_wxyz, ps, ws, vs
        
    def do_process(self) -> None:
        """Generate contactnets-format trajectories for both TagSLAM and
        BundleSDF outputs.  This requires estimating velocities from differences
        in pose, converting everything to PLL format, and trimming the
        trajectories to view the autonomous dynamics during the toss only.  The
        PLL format is in:
	        [ quaternion  position  angular_velocity  linear_velocity ]
        where:
            - quaternion: 		[qw, qx, qy, qz]
            - position:  		[x, y, z] in meters
            - angular_velocity:	[wx, wy, wz] in rad/second in body frame
            - linear_velocity: 	[vx, vy, vz] in meters/second

        This method stores the full state trajectories in attributes:
            - self.tagslam_full_processed_states
            - self.bundlesdf_full_processed_states

        Then this method additionally stores the trimmed trajectories:
            - self.tagslam_toss_processed_states: List[np.ndarray(N, 13)]
            - self.bundlesdf_toss_processed_states: List[np.ndarray(N, 13)]
            - self.tagslam_toss_times: List[np.ndarray(N,)]
            - self.bundlesdf_toss_times: List[np.ndarray(N,)]
        """
        # Process TagSLAM and BundleSDF data.
        q_ts, p_ts, w_ts, v_ts = self._process_poses(
            self.tagslam_full_poses, self.tagslam_full_times)
        q_bsdf, p_bsdf, w_bsdf, v_bsdf = self._process_poses(
            self.bundlesdf_full_poses, self.bundlesdf_full_times)
        
        self.tagslam_full_processed_states = np.concatenate(
            (q_ts, p_ts, w_ts, v_ts), axis=1)
        self.bundlesdf_full_processed_states = np.concatenate(
            (q_bsdf, p_bsdf, w_bsdf, v_bsdf), axis=1)
        
        # Store trimmed trajectories for toss only.
        self._trim_processed_trajectories()

        # Print information about the trimmed trajectories.
        for i in range(len(self.start_frames)):
            print(f'\n=================== TOSS {i} ===================')
            tagslam_toss_dts = np.mean(self.tagslam_toss_times[i][1:] - \
                                       self.tagslam_toss_times[i][:-1])
            print(f'TagSLAM toss {i} trajectory information:' + \
                f'\n\t{self.tagslam_toss_processed_states[i].shape=}' + \
                f'\n\t{self.tagslam_toss_times[i][0]=}' + \
                f'\n\tAverage frame rate (toss): {1/tagslam_toss_dts}\n')

            bsdf_toss_dts = np.mean(self.bundlesdf_toss_times[i][1:] - \
                                    self.bundlesdf_toss_times[i][:-1])
            print(f'BundleSDF toss {i} trajectory information:' + \
                f'\n\t{self.bundlesdf_toss_processed_states[i].shape=}' + \
                f'\n\t{self.bundlesdf_toss_times[i][0]=}' + \
                f'\n\tAverage frame rate (toss): {1/bsdf_toss_dts}\n')

    def _trim_processed_trajectories(self) -> None:
        """After trajectories are already processed, store trimmed versions of
        them corresponding to autonomous dynamics throughout a toss.  This
        stores the following attributes:
            - self.tagslam_toss_processed_states: List[np.ndarray(N, 13)]
            - self.bundlesdf_toss_processed_states: List[np.ndarray(N, 13)]
            - self.tagslam_toss_times: List[np.ndarray(N,)]
            - self.bundlesdf_toss_times: List[np.ndarray(N,)]
        """
        self.bundlesdf_toss_processed_states = []
        self.bundlesdf_toss_times = []
        self.tagslam_toss_processed_states = []
        self.tagslam_toss_times = []

        for i in range(len(self.start_frames)):
            # Need to do one less than provided start and end frames because
            # loaded data in 1-indexed directory but provided 0-indexed
            # start_frame and end_frame.
            b_start = self.start_frames[i] - 1
            b_end = self.end_frames[i] - 1
            self.bundlesdf_toss_processed_states.append(
                self.bundlesdf_full_processed_states[b_start:b_end])
            self.bundlesdf_toss_times.append(
                self.bundlesdf_full_times[b_start:b_end])

            # Find the start and end frames that are most synchronized in time
            # with those pre-selected for BundleSDF trajectories.
            t_start = np.argmin(
                (self.tagslam_full_times - self.bundlesdf_toss_times[i][0])**2)
            t_end = t_start + (b_end - b_start)
            self.tagslam_toss_processed_states.append(
                self.tagslam_full_processed_states[t_start:t_end])
            self.tagslam_toss_times.append(
                self.tagslam_full_times[t_start:t_end])
        
    def plot_trajectory(self, full_trajectory: bool = True) -> None:
        """Visualize the BundleSDF and TagSLAM trajectories overlayed on a set
        of plots."""
        def do_plot(q_ts, p_ts, w_ts, v_ts, t_ts,
                    q_bsdf, p_bsdf, w_bsdf, v_bsdf, t_bsdf, toss_num, full):
            first_t = min(min(t_ts), min(t_bsdf))
            t_ts -= first_t
            t_bsdf -= first_t

            fig, ax = plt.subplots(4, 4, figsize=(15, 15), sharex='all',
                                sharey='row')
            ax[0, 0].plot(t_ts, p_ts[:, 0], label='TagSLAM')
            ax[0, 0].plot(t_bsdf, p_bsdf[:, 0], label='BundleSDF')
            ax[0, 0].set_title('X Position')
            ax[0, 1].plot(t_ts, p_ts[:, 1], label='TagSLAM')
            ax[0, 1].plot(t_bsdf, p_bsdf[:, 1], label='BundleSDF')
            ax[0, 1].set_title('Y Position')
            ax[0, 2].plot(t_ts, p_ts[:, 2], label='TagSLAM')
            ax[0, 2].plot(t_bsdf, p_bsdf[:, 2], label='BundleSDF')
            ax[0, 2].set_title('Z Position')

            ax[1, 0].plot(t_ts, q_ts[:, 0], label='TagSLAM')
            ax[1, 0].plot(t_bsdf, q_bsdf[:, 0], label='BundleSDF')
            ax[1, 0].set_title('W Quaternion')
            ax[1, 1].plot(t_ts, q_ts[:, 1], label='TagSLAM')
            ax[1, 1].plot(t_bsdf, q_bsdf[:, 1], label='BundleSDF')
            ax[1, 1].set_title('X Quaternion')
            ax[1, 2].plot(t_ts, q_ts[:, 2], label='TagSLAM')
            ax[1, 2].plot(t_bsdf, q_bsdf[:, 2], label='BundleSDF')
            ax[1, 2].set_title('Y Quaternion')
            ax[1, 3].plot(t_ts, q_ts[:, 3], label='TagSLAM')
            ax[1, 3].plot(t_bsdf, q_bsdf[:, 3], label='BundleSDF')
            ax[1, 3].set_title('Z Quaternion')

            ax[2, 0].plot(t_ts, v_ts[:, 0], label='TagSLAM')
            ax[2, 0].plot(t_bsdf, v_bsdf[:, 0], label='BundleSDF')
            ax[2, 0].set_title('X Velocity')
            ax[2, 1].plot(t_ts, v_ts[:, 1], label='TagSLAM')
            ax[2, 1].plot(t_bsdf, v_bsdf[:, 1], label='BundleSDF')
            ax[2, 1].set_title('Y Velocity')
            ax[2, 2].plot(t_ts, v_ts[:, 2], label='TagSLAM')
            ax[2, 2].plot(t_bsdf, v_bsdf[:, 2], label='BundleSDF')
            ax[2, 2].set_title('Z Velocity')

            ax[3, 0].plot(t_ts, w_ts[:, 0], label='TagSLAM')
            ax[3, 0].plot(t_bsdf, w_bsdf[:, 0], label='BundleSDF')
            ax[3, 0].set_title('X Angular Velocity')
            ax[3, 1].plot(t_ts, w_ts[:, 1], label='TagSLAM')
            ax[3, 1].plot(t_bsdf, w_bsdf[:, 1], label='BundleSDF')
            ax[3, 1].set_title('Y Angular Velocity')
            ax[3, 2].plot(t_ts, w_ts[:, 2], label='TagSLAM')
            ax[3, 2].plot(t_bsdf, w_bsdf[:, 2], label='BundleSDF')
            ax[3, 2].set_title('Z Angular Velocity')

            ax[0, 0].set_ylabel('Position [m]')
            ax[1, 0].set_ylabel('Orientation')
            ax[2, 0].set_ylabel('Linear Velocity [m/s]')
            ax[3, 0].set_ylabel('Angular Velocity [rad/s]')
            ax[3, 0].set_xlabel('Time from first pose [s]')
            ax[3, 1].set_xlabel('Time from first pose [s]')
            ax[3, 2].set_xlabel('Time from first pose [s]')
            ax[3, 3].set_xlabel('Time from first pose [s]')

            handles, labels = ax[3,2].get_legend_handles_labels()
            fig.legend(handles, labels, loc='lower center')

            fig.suptitle(title)

            # Save the figure in the PLL assets directory where the trajectory
            # data goes.
            pll_bsdf_asset_dir = file_utils.contactnets_input_dir_bundlesdf(
                dataset=self.dataset, iteration=self.iteration_num,
                bundlesdf_id=self.bundlesdf_id, full=full_trajectory)
            plot_name = '' if full else f'toss_{toss_num}_'
            plot_name += f'{self.bundlesdf_id}.png'
            plt.savefig(op.join(pll_bsdf_asset_dir, plot_name))
            print(f'Saved plot to {op.join(pll_bsdf_asset_dir, plot_name)}.\n')

            if self.plot:
                plt.show()
            plt.close()

        if full_trajectory:
            q_ts = self.tagslam_full_processed_states[:, 0:4]
            p_ts = self.tagslam_full_processed_states[:, 4:7]
            w_ts = self.tagslam_full_processed_states[:, 7:10]
            v_ts = self.tagslam_full_processed_states[:, 10:13]
            t_ts = self.tagslam_full_times

            q_bsdf = self.bundlesdf_full_processed_states[:, 0:4]
            p_bsdf = self.bundlesdf_full_processed_states[:, 4:7]
            w_bsdf = self.bundlesdf_full_processed_states[:, 7:10]
            v_bsdf = self.bundlesdf_full_processed_states[:, 10:13]
            t_bsdf = self.bundlesdf_full_times

            title='Full Trajectory Results'
            do_plot(q_ts, p_ts, w_ts, v_ts, t_ts,
                    q_bsdf, p_bsdf, w_bsdf, v_bsdf, t_bsdf,
                    toss_num=None, full=full_trajectory)

        else:
            for i in range(len(self.tagslam_toss_times)):
                q_ts = self.tagslam_toss_processed_states[i][:, 0:4]
                p_ts = self.tagslam_toss_processed_states[i][:, 4:7]
                w_ts = self.tagslam_toss_processed_states[i][:, 7:10]
                v_ts = self.tagslam_toss_processed_states[i][:, 10:13]
                t_ts = self.tagslam_toss_times[i]

                q_bsdf = self.bundlesdf_toss_processed_states[i][:, 0:4]
                p_bsdf = self.bundlesdf_toss_processed_states[i][:, 4:7]
                w_bsdf = self.bundlesdf_toss_processed_states[i][:, 7:10]
                v_bsdf = self.bundlesdf_toss_processed_states[i][:, 10:13]
                t_bsdf = self.bundlesdf_toss_times[i]

                title=f'Toss {i + self.start_toss} Trajectory Results'
                do_plot(q_ts, p_ts, w_ts, v_ts, t_ts,
                        q_bsdf, p_bsdf, w_bsdf, v_bsdf, t_bsdf,
                        toss_num=i+self.start_toss, full=full_trajectory)

    def save_data(self, save_tagslam: bool = False,
                  save_bundlesdf: bool = False) -> None:
        """Stores data as .pt files in the ContactNets input directory."""
        toss_filenames = [f'{toss_i}.pt' for toss_i in range(
            self.start_toss, self.end_toss+1)]

        print('Saving files summary:')

        if save_tagslam:
            # Save full trajectory.
            full_tagslam_dir = file_utils.contactnets_input_dir_tagslam(
                dataset=self.dataset, full=True)
            torch.save(
                torch.tensor(self.tagslam_full_processed_states),
                op.join(full_tagslam_dir, 'tagslam.pt'))
            print(f"\t{op.join(full_tagslam_dir, 'tagslam.pt')}")

            # Do toss trajectories.
            toss_tagslam_dir = file_utils.contactnets_input_dir_tagslam(
                dataset=self.dataset, full=False)
            for i in range(len(toss_filenames)):
                torch.save(
                    torch.tensor(self.tagslam_toss_processed_states[i]),
                    op.join(toss_tagslam_dir, toss_filenames[i]))
                print(f'\t{op.join(toss_tagslam_dir, toss_filenames[i])}')

        if save_bundlesdf:
            # Save full trajectory.
            full_bundlesdf_dir = file_utils.contactnets_input_dir_bundlesdf(
                dataset=self.dataset, iteration=self.iteration_num,
                bundlesdf_id=self.bundlesdf_id, full=True)
            traj_filename = f'{self.bundlesdf_id}.pt'
            torch.save(
                torch.tensor(self.bundlesdf_full_processed_states),
                op.join(full_bundlesdf_dir, traj_filename))
            print(f'\t{op.join(full_bundlesdf_dir, traj_filename)}')

            # Do toss trajectories.
            toss_bundlesdf_dir = file_utils.contactnets_input_dir_bundlesdf(
                dataset=self.dataset, iteration=self.iteration_num,
                bundlesdf_id=self.bundlesdf_id, full=False)
            for i in range(len(toss_filenames)):
                torch.save(
                    torch.tensor(self.bundlesdf_toss_processed_states[i]),
                    op.join(toss_bundlesdf_dir, toss_filenames[i]))
                print(f'\t{op.join(toss_bundlesdf_dir, toss_filenames[i])}')


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
@click.option('--cycle-iteration',
              type=int,
              default=1,
              help="BundleSDF iteration number (0 means use TagSLAM poses).")

def main_command(vision_asset: str, bundlesdf_id: str, cycle_iteration: int):
    # First decode the system and start/end tosses from the provided asset
    # directory.
    assert '_' in vision_asset, f'Invalid asset directory: {vision_asset}.'
    object = vision_asset.split('_')[0]

    start_toss = int(vision_asset.split('_')[1].split('-')[0])
    end_toss = start_toss if '-' not in vision_asset else \
        int(vision_asset.split('-')[1])
    assert start_toss <= end_toss, f'Invalid toss range: {start_toss} ' + \
        f'-{end_toss} inferred from {vision_asset=}.'
    
    # Locate all the related files and directories for the given vision asset.
    rosbag_number = file_utils.load_rosbag_number_from_yaml(
        object, start_toss, second_toss_number=end_toss)

    # Decode the BundleSDF run ID and find if there's an associated PLL run ID.
    if bundlesdf_id[:13] != 'bundlesdf_id_':
        bundlesdf_id = f'bundlesdf_id_{bundlesdf_id}'
    pll_id = file_utils.bundlesdf_run_associated_pll_run_id(
        dataset=vision_asset, bundlesdf_id=bundlesdf_id,
        cycle_iteration=cycle_iteration)

    print(f'Processing toss {vision_asset} in raw_{rosbag_number}.bag from ' + \
          f'BundleSDF run ID {bundlesdf_id} with associated PLL ID {pll_id}.\n')

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

    # Do the conversion.
    converter = ConverterBundleSDFToPLL(
        bundlesdf_id=bundlesdf_id, pll_id=pll_id,
        relative_start_frames=relative_start_frames,
        relative_end_frames=relative_end_frames,
        start_ros_times=start_ros_times, start_toss=start_toss,
        end_toss=end_toss, object=object, cycle_iteration=cycle_iteration,
        cam_trans=cam_trans, cam_rot_axis_angle=cam_rot_axis_angle,
        frame_rate=30, z_table=z_table, plot=True
    )

    converter.do_process()
    converter.plot_trajectory(full_trajectory=True)
    converter.plot_trajectory(full_trajectory=False)
    converter.save_data(save_tagslam=True, save_bundlesdf=True)


if __name__ == '__main__':
    main_command()  # pylint: disable=no-value-for-parameter
