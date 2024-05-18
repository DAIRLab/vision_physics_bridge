"""This file performs output conversions from BundleSDF trajectories to input
formats required by PLL.

TODO:  This functionality has only been checked for single toss datasets, e.g.
cube_2.  Still to be tested on multi-toss experiments.
"""

# import argparse
import click
import os
import os.path as op
import numpy as np
import pdb
import torch
from scipy import signal
from scipy.spatial.transform import Rotation
from pyquaternion import Quaternion
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pdb
import math
import trimesh
from typing import Tuple

import file_utils
import math_utils

from overlay_videos import OverlayVideoGenerator
from vis_utils import SDFSliceViewer


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


class TrajectoryConverterBundleSDFToPLL:
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
    def __init__(self, tracking_bundlesdf_id: str, nerf_bundlesdf_id: str,
                 start_toss: int, end_toss: int, object: str,
                 cycle_iteration: int, cam_trans: np.ndarray,
                 cam_rot_axis_angle: np.ndarray, frame_rate: int,
                 z_table: float, relative_start_frames: list,
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
        self.tagslam_dir = file_utils.synchronized_tagslam_pose_dir(
            self.dataset, check_exists=True)
        self.bundlesdf_dir = file_utils.bundlesdf_pose_dir(
            self.dataset, cycle_iteration=self.iteration_num,
            bundlesdf_id=self.tracking_bundlesdf_id)
        self.annotated_dir = file_utils.bundlesdf_annotated_poses_dir(
            self.dataset)

    def _load_poses(self) -> None:
        """Load the timestamped poses reported from TagSLAM and BundleSDF,
        saving the results in attributes:
            - self.tagslam_full_times
            - self.tagslam_full_poses
            - self.bundlesdf_full_times
            - self.bundlesdf_b_full_poses (BundleSDF body origin)
            - self.bundlesdf_t_full_poses (TagSLAM body origin)
            - self.keyframe_full_times
            - self.keyframe_b_full_poses (BundleSDF body origin)
            - self.keyframe_t_full_poses (TagSLAM body origin)
        
        The TagSLAM information and BundleSDF time information are generated by
        rosbag_processor.py's extract_synchronized_images_and_tagslam_poses.  
        The TagSLAM time and poses are in the ground truth pose directory's file
        synced_tagslam.txt.  The BundleSDF information comes from the BundleSDF
        output directory's XXXX.txt files, which are generated by running
        BundleSDF and are timestamped according to the BundleSDF times, which
        come from the bundlesdf_timestamps.txt file.

        Inputs:
            bsdf_times (N,)
        """
        self._load_tagslam_poses()
        self._load_bundlesdf_poses()
        self._load_bundlesdf_keyframe_poses()

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
              f'\n\t{self.bundlesdf_b_full_poses.shape=}' + \
              f'\n\t{self.bundlesdf_full_times[0]=}' + \
              f'\n\tAverage frame rate (full): {1/bsdf_full_dts}\n')

        print(f'BundleSDF keyframe full trajectory information:' + \
              f'\n\t{self.keyframe_b_full_poses.shape=}' + \
              f'\n\t{self.keyframe_full_times[0]=}\n')
        
    def _load_tagslam_poses(self) -> None:
        """Load all the poses reported by TagSLAM.  These are in world
        coordinates of the TagSLAM body origin with the following ordering:
            [x, y, z, qx, qy, qz, qw]
        """
        tagslam_data = np.loadtxt(
            op.join(self.tagslam_dir, 'synced_tagslam.txt'))

        self.tagslam_full_times = tagslam_data[:, 0]
        self.tagslam_full_poses = tagslam_data[:, 1:]

    def _load_bundlesdf_poses(self) -> None:
        """Load all the poses reported by BundleSDF.  Store these in two
        formats:  of the BundleSDF body origin in world frame, and of the
        TagSLAM body origin in world frame (converted from BundleSDF
        origin in camera coordinates via
        math_utils.transform_bundletrack_output).  All have the following
        ordering:
            [x, y, z, qx, qy, qz, qw]
        """
        bundlesdf_b_poses = []
        bundlesdf_t_poses = []
        bundlesdf_times = np.loadtxt(
            op.join(op.dirname(self.tagslam_dir), 'bundlesdf_timestamps.txt'))

        # Add 1 for range bounds because BundleSDF poses are 1-indexed.
        for i in range(1, bundlesdf_times.shape[0] + 1):
            trans_mat = np.loadtxt(op.join(self.bundlesdf_dir, "%04i.txt" % i))
            trans_mat_b = math_utils.camera_to_world(
                trans_mat, translation=self.cam_trans,
                axis_vec=self.cam_rot_axis_angle)
            trans_mat_t = \
                math_utils.transform_bundletrack_origin_to_tagslam_origin(
                    pred_pose=trans_mat,
                    bsdf_output_pose_dir=self.bundlesdf_dir,
                    annotated_poses_dir=self.annotated_dir,
                    translation=self.cam_trans,
                    axis_vec=self.cam_rot_axis_angle,
                    to_world=True
                )
            pos_quat_b = math_utils.trans_mat_to_pos_quat(
                trans_mat_b).reshape(7)
            pos_quat_t = math_utils.trans_mat_to_pos_quat(
                trans_mat_t).reshape(7)
            bundlesdf_b_poses.append(pos_quat_b)
            bundlesdf_t_poses.append(pos_quat_t)

        self.bundlesdf_b_full_poses = np.array(bundlesdf_b_poses)
        self.bundlesdf_t_full_poses = np.array(bundlesdf_t_poses)
        self.bundlesdf_full_times = np.array(bundlesdf_times)

    def _load_bundlesdf_keyframe_poses(self) -> None:
        """Load all the optimized keyframe poses reported by BundleSDF after
        NeRF training.  Store these in two formats:  of the BundleSDF body
        origin in world frame, and of the TagSLAM body origin in world frame
        (converted from BundleSDF origin in camera coordinates via
        math_utils.transform_bundletrack_output).  All have the following
        ordering:
            [x, y, z, qx, qy, qz, qw]
        """
        # Get the keyframe indices from the BundleSDF tracking results' last
        # frame directory, in keyframes.yml.
        keyframe_idx = \
            file_utils.load_keyframe_indices_from_nerf_results_yml(
                self.dataset, self.iteration_num, self.tracking_bundlesdf_id)
        self.keyframe_idx = [i-1 for i in keyframe_idx]

        # Get the adjusted keyframe poses from the BundleSDF NeRF results'
        # poses_after_nerf.txt.
        keyframe_tfs = \
            file_utils.load_optimized_keyframe_poses_from_nerf_results(
                dataset=self.dataset, cycle_iteration=self.iteration_num,
                tracking_bundlesdf_id=self.tracking_bundlesdf_id,
                nerf_bundlesdf_id=self.nerf_bundlesdf_id
            )

        keyframe_b_poses = []
        keyframe_t_poses = []
        # Convert BundleSDF keyframes 1-indexing to 0-indexing.
        keyframe_times = [self.bundlesdf_full_times[i-1] for i in keyframe_idx]

        for bsdf_pose in keyframe_tfs:
            trans_mat = bsdf_pose
            trans_mat_b = math_utils.camera_to_world(
                trans_mat, translation=self.cam_trans,
                axis_vec=self.cam_rot_axis_angle)
            trans_mat_t = \
                math_utils.transform_bundletrack_origin_to_tagslam_origin(
                    pred_pose=trans_mat,
                    bsdf_output_pose_dir=self.bundlesdf_dir,
                    annotated_poses_dir=self.annotated_dir,
                    translation=self.cam_trans,
                    axis_vec=self.cam_rot_axis_angle,
                    to_world=True
                )
            pos_quat_b = math_utils.trans_mat_to_pos_quat(
                trans_mat_b).reshape(7)
            pos_quat_t = math_utils.trans_mat_to_pos_quat(
                trans_mat_t).reshape(7)
            keyframe_b_poses.append(pos_quat_b)
            keyframe_t_poses.append(pos_quat_t)

        self.keyframe_b_full_poses = np.array(keyframe_b_poses)
        self.keyframe_t_full_poses = np.array(keyframe_t_poses)
        self.keyframe_full_times = np.array(keyframe_times)

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
    
    def _make_quaternions_consistent(self, quat):
        # Do some input checking.
        assert quat.shape[1] == 4, f'{quat.shape=} not of expected size (N, 4).'
        assert quat.ndim == 2, f'{quat.shape=} not of expected size (N, 4).'

        quat /= np.linalg.norm(quat, axis=1).reshape(-1,1)
        rot_t = Rotation.from_quat(quat)

        # Fix and filter quaternions.
        rvecs = rotvecfix(rot_t.as_rotvec())
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
        else:
            qs_xyzw = self._make_quaternions_consistent(qs_xyzw)

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
            - self.bundlesdf_b_full_processed_states
            - self.bundlesdf_t_full_processed_states

        Then this method additionally stores the trimmed trajectories:
            - self.tagslam_toss_processed_states: List[np.ndarray(N, 13)]
            - self.bundlesdf_b_toss_processed_states: List[np.ndarray(N, 13)]
            - self.bundlesdf_t_toss_processed_states: List[np.ndarray(N, 13)]
            - self.tagslam_toss_times: List[np.ndarray(N,)]
            - self.bundlesdf_toss_times: List[np.ndarray(N,)]
        """
        # Process TagSLAM and BundleSDF data.
        q_ts, p_ts, w_ts, v_ts = self._process_poses(
            self.tagslam_full_poses, self.tagslam_full_times)
        q_bsdf_b, p_bsdf_b, w_bsdf_b, v_bsdf_b = self._process_poses(
            self.bundlesdf_b_full_poses, self.bundlesdf_full_times)
        q_bsdf_t, p_bsdf_t, w_bsdf_t, v_bsdf_t = self._process_poses(
            self.bundlesdf_t_full_poses, self.bundlesdf_full_times)
        q_key_b, p_key_b, w_key_b, v_key_b = self._process_poses(
            self.keyframe_b_full_poses, self.keyframe_full_times,
            filter_rot=False, filter_pos=False, filter_lin_vel=False,
            filter_ang_vel=False)
        q_key_t, p_key_t, w_key_t, v_key_t = self._process_poses(
            self.keyframe_t_full_poses, self.keyframe_full_times,
            filter_rot=False, filter_pos=False, filter_lin_vel=False,
            filter_ang_vel=False)

        self.tagslam_full_processed_states = np.concatenate(
            (q_ts, p_ts, w_ts, v_ts), axis=1)
        self.bundlesdf_b_full_processed_states = np.concatenate(
            (q_bsdf_b, p_bsdf_b, w_bsdf_b, v_bsdf_b), axis=1)
        self.bundlesdf_t_full_processed_states = np.concatenate(
            (q_bsdf_t, p_bsdf_t, w_bsdf_t, v_bsdf_t), axis=1)
        self.keyframe_b_full_processed_states = np.concatenate(
            (q_key_b, p_key_b, w_key_b, v_key_b), axis=1)
        self.keyframe_t_full_processed_states = np.concatenate(
            (q_key_t, p_key_t, w_key_t, v_key_t), axis=1)
        
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
                f'\n\t{self.bundlesdf_b_toss_processed_states[i].shape=}' + \
                f'\n\t{self.bundlesdf_toss_times[i][0]=}' + \
                f'\n\tAverage frame rate (toss): {1/bsdf_toss_dts}\n')

            print(f'BundleSDF keyframe toss {i} trajectory information:' + \
                f'\n\t{self.keyframe_b_toss_processed_states[i].shape=}' + \
                f'\n\t{self.keyframe_toss_times[i][0]=}\n')

    def _trim_processed_trajectories(self) -> None:
        """After trajectories are already processed, store trimmed versions of
        them corresponding to autonomous dynamics throughout a toss.  This
        stores the following attributes:
            - self.tagslam_toss_processed_states: List[np.ndarray(N, 13)]
            - self.bundlesdf_b_toss_processed_states: List[np.ndarray(N, 13)]
            - self.bundlesdf_t_toss_processed_states: List[np.ndarray(N, 13)]
            - self.tagslam_toss_times: List[np.ndarray(N,)]
            - self.bundlesdf_toss_times: List[np.ndarray(N,)]
        """
        self.bundlesdf_b_toss_processed_states = []
        self.bundlesdf_t_toss_processed_states = []
        self.bundlesdf_toss_times = []
        self.keyframe_b_toss_processed_states = []
        self.keyframe_t_toss_processed_states = []
        self.keyframe_toss_times = []
        self.tagslam_toss_processed_states = []
        self.tagslam_toss_times = []

        for i in range(len(self.start_frames)):
            # Need to do one less than provided start and end frames because
            # loaded data in 1-indexed directory but provided 0-indexed
            # start_frame and end_frame.
            b_start = self.start_frames[i] - 1
            b_end = self.end_frames[i] - 1
            self.bundlesdf_b_toss_processed_states.append(
                self.bundlesdf_b_full_processed_states[b_start:b_end])
            self.bundlesdf_t_toss_processed_states.append(
                self.bundlesdf_t_full_processed_states[b_start:b_end])
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
            
            # Keep any keyframe data in the same time range as the toss.
            key_start = np.argmin(
                (self.keyframe_full_times - self.bundlesdf_toss_times[i][0])**2)
            key_end = np.argmin(
                (self.keyframe_full_times - self.bundlesdf_toss_times[i][-1]
                 )**2)
            self.keyframe_b_toss_processed_states.append(
                self.keyframe_b_full_processed_states[key_start:key_end])
            self.keyframe_t_toss_processed_states.append(
                self.keyframe_t_full_processed_states[key_start:key_end])
            self.keyframe_toss_times.append(
                self.keyframe_full_times[key_start:key_end])

    def plot_trajectory(self, full_trajectory: bool = True) -> None:
        """Visualize the BundleSDF and TagSLAM trajectories overlayed on a set
        of plots."""
        def do_plot(q_ts, p_ts, w_ts, v_ts, t_ts,
                    q_bsdf_t, p_bsdf_t, w_bsdf_t, v_bsdf_t,
                    q_bsdf_b, p_bsdf_b, w_bsdf_b, v_bsdf_b, t_bsdf,
                    q_key_t, p_key_t, _w_key_t, _v_key_t,
                    q_key_b, p_key_b, _w_key_b, _v_key_b, t_key,
                    toss_num, full):
            first_t = min(min(t_ts), min(t_bsdf))
            t_ts -= first_t
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

            ax[0, 0].plot(t_ts, p_ts[:, 0], label='TagSLAM')
            ax[0, 0].plot(t_bsdf, p_bsdf_t[:, 0], label='BundleSDF, T Origin')
            ax[0, 0].plot(t_bsdf, p_bsdf_b[:, 0], label='BundleSDF, B Origin',
                          linestyle='--')
            ax[0, 0].scatter(t_key, p_key_t[:, 0], c='orange', s=20,
                             label='Keyframes, T Origin')
            ax[0, 0].scatter(t_key, p_key_b[:, 0], c='green', s=20,
                             label='Keyframes, B Origin')
            ax[0, 0].set_title('X Position')
            ax[0, 1].plot(t_ts, p_ts[:, 1], label='TagSLAM')
            ax[0, 1].plot(t_bsdf, p_bsdf_t[:, 1], label='BundleSDF, T Origin')
            ax[0, 1].plot(t_bsdf, p_bsdf_b[:, 1], label='BundleSDF, B Origin',
                          linestyle='--')
            ax[0, 1].scatter(t_key, p_key_t[:, 1], c='orange', s=20,
                             label='Keyframes, T Origin')
            ax[0, 1].scatter(t_key, p_key_b[:, 1], c='green', s=20,
                             label='Keyframes, B Origin')
            ax[0, 1].set_title('Y Position')
            ax[0, 2].plot(t_ts, p_ts[:, 2], label='TagSLAM')
            ax[0, 2].plot(t_bsdf, p_bsdf_t[:, 2], label='BundleSDF, T Origin')
            ax[0, 2].plot(t_bsdf, p_bsdf_b[:, 2], label='BundleSDF, B Origin',
                          linestyle='--')
            ax[0, 2].scatter(t_key, p_key_t[:, 2], c='orange', s=20,
                             label='Keyframes, T Origin')
            ax[0, 2].scatter(t_key, p_key_b[:, 2], c='green', s=20,
                             label='Keyframes, B Origin')
            ax[0, 2].set_title('Z Position')

            ax[1, 0].plot(t_ts, q_ts[:, 0], label='TagSLAM')
            ax[1, 0].plot(t_bsdf, q_bsdf_t[:, 0], label='BundleSDF, T Origin')
            ax[1, 0].plot(t_bsdf, q_bsdf_b[:, 0], label='BundleSDF, B Origin',
                          linestyle='--')
            ax[1, 0].scatter(t_key, q_key_t[:, 0], c='orange', s=20,
                             label='Keyframes, T Origin')
            ax[1, 0].scatter(t_key, q_key_b[:, 0], c='green', s=20,
                             label='Keyframes, B Origin')
            ax[1, 0].set_title('W Quaternion')
            ax[1, 1].plot(t_ts, q_ts[:, 1], label='TagSLAM')
            ax[1, 1].plot(t_bsdf, q_bsdf_t[:, 1], label='BundleSDF, T Origin')
            ax[1, 1].plot(t_bsdf, q_bsdf_b[:, 1], label='BundleSDF, B Origin',
                          linestyle='--')
            ax[1, 1].scatter(t_key, q_key_t[:, 1], c='orange', s=20,
                             label='Keyframes, T Origin')
            ax[1, 1].scatter(t_key, q_key_b[:, 1], c='green', s=20,
                             label='Keyframes, B Origin')
            ax[1, 1].set_title('X Quaternion')
            ax[1, 2].plot(t_ts, q_ts[:, 2], label='TagSLAM')
            ax[1, 2].plot(t_bsdf, q_bsdf_t[:, 2], label='BundleSDF, T Origin')
            ax[1, 2].plot(t_bsdf, q_bsdf_b[:, 2], label='BundleSDF, B Origin',
                          linestyle='--')
            ax[1, 2].scatter(t_key, q_key_t[:, 2], c='orange', s=20,
                             label='Keyframes, T Origin')
            ax[1, 2].scatter(t_key, q_key_b[:, 2], c='green', s=20,
                             label='Keyframes, B Origin')
            ax[1, 2].set_title('Y Quaternion')
            ax[1, 3].plot(t_ts, q_ts[:, 3], label='TagSLAM')
            ax[1, 3].plot(t_bsdf, q_bsdf_t[:, 3], label='BundleSDF, T Origin')
            ax[1, 3].plot(t_bsdf, q_bsdf_b[:, 3], label='BundleSDF, B Origin',
                          linestyle='--')
            ax[1, 3].scatter(t_key, q_key_t[:, 3], c='orange', s=20,
                             label='Keyframes, T Origin')
            ax[1, 3].scatter(t_key, q_key_b[:, 3], c='green', s=20,
                             label='Keyframes, B Origin')
            ax[1, 3].set_title('Z Quaternion')

            ax[2, 0].plot(t_ts, v_ts[:, 0], label='TagSLAM')
            ax[2, 0].plot(t_bsdf, v_bsdf_t[:, 0], label='BundleSDF, T Origin')
            ax[2, 0].plot(t_bsdf, v_bsdf_b[:, 0], label='BundleSDF, B Origin',
                          linestyle='--')
            ax[2, 0].set_title('X Velocity')
            ax[2, 1].plot(t_ts, v_ts[:, 1], label='TagSLAM')
            ax[2, 1].plot(t_bsdf, v_bsdf_t[:, 1], label='BundleSDF, T Origin')
            ax[2, 1].plot(t_bsdf, v_bsdf_b[:, 1], label='BundleSDF, B Origin',
                          linestyle='--')
            ax[2, 1].set_title('Y Velocity')
            ax[2, 2].plot(t_ts, v_ts[:, 2], label='TagSLAM')
            ax[2, 2].plot(t_bsdf, v_bsdf_t[:, 2], label='BundleSDF, T Origin')
            ax[2, 2].plot(t_bsdf, v_bsdf_b[:, 2], label='BundleSDF, B Origin',
                          linestyle='--')
            ax[2, 2].set_title('Z Velocity')

            ax[3, 0].plot(t_ts, w_ts[:, 0], label='TagSLAM')
            ax[3, 0].plot(t_bsdf, w_bsdf_t[:, 0], label='BundleSDF, T Origin')
            ax[3, 0].plot(t_bsdf, w_bsdf_b[:, 0], label='BundleSDF, B Origin',
                          linestyle='--')
            ax[3, 0].set_title('X Angular Velocity')
            ax[3, 1].plot(t_ts, w_ts[:, 1], label='TagSLAM')
            ax[3, 1].plot(t_bsdf, w_bsdf_t[:, 1], label='BundleSDF, T Origin')
            ax[3, 1].plot(t_bsdf, w_bsdf_b[:, 1], label='BundleSDF, B Origin',
                          linestyle='--')
            ax[3, 1].set_title('Y Angular Velocity')
            ax[3, 2].plot(t_ts, w_ts[:, 2], label='TagSLAM')
            ax[3, 2].plot(t_bsdf, w_bsdf_t[:, 2], label='BundleSDF, T Origin')
            ax[3, 2].plot(t_bsdf, w_bsdf_b[:, 2], label='BundleSDF, B Origin',
                          linestyle='--')
            ax[3, 2].set_title('Z Angular Velocity')

            # Include an orientation error plot in the empty upper right.
            q_errors = math_utils.quaternion_error(q_ts, q_bsdf_t)
            q_errors *= 180 / np.pi

            # Include the orientation error for the keyframes.
            tagslam_time_indices = [np.argmin((t_ts - t)**2) for t in t_key]
            q_key_errors = math_utils.quaternion_error(
                q_ts[tagslam_time_indices], q_key_t)
            q_key_errors *= 180 / np.pi
            ax[0, 3].plot(t_ts, q_errors, color='r',
                          label='TagSLAM-to-BundleSDF T')
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
                dataset=self.dataset, iteration=self.iteration_num,
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
                cycle_iteration=self.iteration_num
            )
            plot_name = f'{prefix}.png' if full else \
                f'{prefix}_toss_{toss_num}.png'
            plt.savefig(op.join(dir, plot_name))
            print(f'Saved plot to {op.join(dir, plot_name)}.\n')

            if self.plot:
                plt.show()
            plt.close()

        if full_trajectory:
            q_ts = self.tagslam_full_processed_states[:, 0:4]
            p_ts = self.tagslam_full_processed_states[:, 4:7]
            w_ts = self.tagslam_full_processed_states[:, 7:10]
            v_ts = self.tagslam_full_processed_states[:, 10:13]
            t_ts = self.tagslam_full_times

            q_bsdf_b = self.bundlesdf_b_full_processed_states[:, 0:4]
            p_bsdf_b = self.bundlesdf_b_full_processed_states[:, 4:7]
            w_bsdf_b = self.bundlesdf_b_full_processed_states[:, 7:10]
            v_bsdf_b = self.bundlesdf_b_full_processed_states[:, 10:13]
            q_bsdf_t = self.bundlesdf_t_full_processed_states[:, 0:4]
            p_bsdf_t = self.bundlesdf_t_full_processed_states[:, 4:7]
            w_bsdf_t = self.bundlesdf_t_full_processed_states[:, 7:10]
            v_bsdf_t = self.bundlesdf_t_full_processed_states[:, 10:13]
            t_bsdf = self.bundlesdf_full_times

            q_key_b = self.keyframe_b_full_processed_states[:, 0:4]
            p_key_b = self.keyframe_b_full_processed_states[:, 4:7]
            w_key_b = self.keyframe_b_full_processed_states[:, 7:10]
            v_key_b = self.keyframe_b_full_processed_states[:, 10:13]
            q_key_t = self.keyframe_t_full_processed_states[:, 0:4]
            p_key_t = self.keyframe_t_full_processed_states[:, 4:7]
            w_key_t = self.keyframe_t_full_processed_states[:, 7:10]
            v_key_t = self.keyframe_t_full_processed_states[:, 10:13]
            t_key = self.keyframe_full_times

            title='Full Trajectory Results'
            do_plot(q_ts, p_ts, w_ts, v_ts, t_ts,
                    q_bsdf_t, p_bsdf_t, w_bsdf_t, v_bsdf_t,
                    q_bsdf_b, p_bsdf_b, w_bsdf_b, v_bsdf_b, t_bsdf,
                    q_key_t, p_key_t, w_key_t, v_key_t,
                    q_key_b, p_key_b, w_key_b, v_key_b, t_key,
                    toss_num=None, full=full_trajectory)

        else:
            for i in range(len(self.tagslam_toss_times)):
                q_ts = self.tagslam_toss_processed_states[i][:, 0:4]
                p_ts = self.tagslam_toss_processed_states[i][:, 4:7]
                w_ts = self.tagslam_toss_processed_states[i][:, 7:10]
                v_ts = self.tagslam_toss_processed_states[i][:, 10:13]
                t_ts = self.tagslam_toss_times[i]

                q_bsdf_b = self.bundlesdf_b_toss_processed_states[i][:, 0:4]
                p_bsdf_b = self.bundlesdf_b_toss_processed_states[i][:, 4:7]
                w_bsdf_b = self.bundlesdf_b_toss_processed_states[i][:, 7:10]
                v_bsdf_b = self.bundlesdf_b_toss_processed_states[i][:, 10:13]
                q_bsdf_t = self.bundlesdf_t_toss_processed_states[i][:, 0:4]
                p_bsdf_t = self.bundlesdf_t_toss_processed_states[i][:, 4:7]
                w_bsdf_t = self.bundlesdf_t_toss_processed_states[i][:, 7:10]
                v_bsdf_t = self.bundlesdf_t_toss_processed_states[i][:, 10:13]
                t_bsdf = self.bundlesdf_toss_times[i]

                q_key_b = self.keyframe_b_toss_processed_states[i][:, 0:4]
                p_key_b = self.keyframe_b_toss_processed_states[i][:, 4:7]
                w_key_b = self.keyframe_b_toss_processed_states[i][:, 7:10]
                v_key_b = self.keyframe_b_toss_processed_states[i][:, 10:13]
                q_key_t = self.keyframe_t_toss_processed_states[i][:, 0:4]
                p_key_t = self.keyframe_t_toss_processed_states[i][:, 4:7]
                w_key_t = self.keyframe_t_toss_processed_states[i][:, 7:10]
                v_key_t = self.keyframe_t_toss_processed_states[i][:, 10:13]
                t_key = self.keyframe_toss_times[i]

                title=f'Toss {i + self.start_toss} Trajectory Results'
                do_plot(q_ts, p_ts, w_ts, v_ts, t_ts,
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
        tracking results."""
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
                bundlesdf_id=self.tracking_bundlesdf_id, full=True)
            traj_filename = f'{self.tracking_bundlesdf_id}.pt'
            torch.save(
                torch.tensor(self.bundlesdf_b_full_processed_states),
                op.join(full_bundlesdf_dir, traj_filename))
            print(f'\t{op.join(full_bundlesdf_dir, traj_filename)}')

            # Do toss trajectories.
            toss_bundlesdf_dir = file_utils.contactnets_input_dir_bundlesdf(
                dataset=self.dataset, iteration=self.iteration_num,
                bundlesdf_id=self.tracking_bundlesdf_id, full=False)
            for i in range(len(toss_filenames)):
                torch.save(
                    torch.tensor(self.bundlesdf_b_toss_processed_states[i]),
                    op.join(toss_bundlesdf_dir, toss_filenames[i]))
                print(f'\t{op.join(toss_bundlesdf_dir, toss_filenames[i])}')

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
@click.option('--make-overlay/--skip-overlay',
              type=bool,
              default=True,
              help="whether to make an overlay video.")
@click.option('--remote/--local',
              default=False,
              help="whether to run on a remote server.")
@click.option('--show/--noshow',
              default=True,
              help="whether to show the plots.")


def main_command(vision_asset: str, bundlesdf_id: str, nerf_bundlesdf_id: str,
                 cycle_iteration: int, make_overlay: bool, remote: bool,
                 show: bool):
    # First decode the system and start/end tosses from the provided asset
    # directory.
    assert cycle_iteration > 0, f'Invalid cycle iteration: {cycle_iteration}.'
    assert '_' in vision_asset, f'Invalid asset directory: {vision_asset}.'
    object = vision_asset.split('_')[0]

    start_toss = int(vision_asset.split('_')[1].split('-')[0])
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
        cam_trans=cam_trans, cam_rot_axis_angle=cam_rot_axis_angle,
        frame_rate=30, z_table=z_table, plot=show
    )
    traj_converter.do_process()
    traj_converter.plot_trajectory(full_trajectory=True)
    traj_converter.plot_trajectory(full_trajectory=False)
    traj_converter.save_data(save_tagslam=True, save_bundlesdf=True)

    # Create an overlay video.
    if make_overlay:
        overlay_generator = OverlayVideoGenerator(
            vision_asset=vision_asset, tracking_bundlesdf_id=bundlesdf_id,
            nerf_bundlesdf_id=nerf_bundlesdf_id,
            cycle_iteration=cycle_iteration, remote=remote
        )
        overlay_generator.make_overlay_video()
        overlay_generator.make_optimized_keyframe_overlay_images()
    else:
        print('Skipping overlay video creation.')

    # Generate the SDF slice images.
    sdf_slice_generator = SDFSliceViewer(
        vision_asset=vision_asset, tracking_bundlesdf_id=bundlesdf_id,
        nerf_bundlesdf_id=nerf_bundlesdf_id, cycle_iteration=cycle_iteration
    )
    sdf_slice_generator.visualization()


if __name__ == '__main__':
    main_command()  # pylint: disable=no-value-for-parameter
