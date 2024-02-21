import argparse
import os.path as op
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from scipy import signal
from pyquaternion import Quaternion
import matplotlib.pyplot as plt
import pdb
import math
from typing import Tuple

import file_utils
import math_utils
import rosbag_processor
import sync_data


FILTER_ORIENTATIONS = True
FILTER_POSITIONS = True
FILTER_LINEAR_VELOCITIES = True
FILTER_ANGULAR_VELOCITIES = True

FILTER_TYPE = 'median'              # Can be median or savgol.
SAVGOL_FILTER_WINDOW_LENGTH = 15
SAVGOL_FILTER_POLYORDER = 3
MEDIAN_FILTER_KERNEL_SIZE = 3

ADJUST_POSITION = True

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


class DatasetManagement:
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
    def __init__(self, start_frame: int, end_frame: int,
                 timestamps: np.ndarray, toss_id: int, toss_type: str,
                 iteration_num: int, cam_trans: np.ndarray,
                 cam_rot_axis_angle: np.ndarray, frame_rate: int,
                 z_shift: float, plot: bool = False) -> None:
        """Prepare for processing data from TagSLAM and BundleSDF.

        Args:
            start_frame:  BundleSDF frame index at which a PLL toss begins.
            end_frame:  BundleSDF frame index at which a PLL toss ends.
            timestamps (N,):  BundleSDF timestamps.
            toss_id:  Toss index.
            toss_type:  The tossed object name.
            iteration_num:  The cycle iteration number of the BundleSDF/
                ContactNets cycle.
            cam_trans (3,):  Position of the camera in world coordinates.
            cam_rot_axis_angle (3,):  Orientation of the camera is world
                coordinates in Rodrigues form.
            frame_rate:  The expected frame rate of the data.  This isn't used
                for processing but can be used to manually inspect that the
                data's timestamps result in a similar frame rate as expected.
            z_shift:  Amount to shift the z positions throughout the trajectory.
            plot:  Whether to show the overlay plot of BundleSDF and TagSLAM
                trajectories.
        """
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.toss_id = toss_id
        self.toss_type = toss_type
        self.iteration_num = iteration_num

        self.plot = plot
        self.cam_trans = cam_trans
        self.cam_rot_axis_angle = cam_rot_axis_angle
        self.frame_rate = frame_rate
        self.z_shift = z_shift

        self._set_up_directories()

        # Load the poses from TagSLAM and BundleSDF.
        self._load_poses(bsdf_times=timestamps)

    def _set_up_directories(self) -> None:
        """Given the stored toss_type and toss_id, loads the following
        attributes:
            - self.tagslam_dir
            - self.bundlesdf_dir
            - self.annotated_dir
            - self.contactnets_dir
        """
        dataset = f'{self.toss_type}_{self.toss_id}'

        self.tagslam_dir = file_utils.tagslam_pose_dir(dataset)
        self.bundlesdf_dir = file_utils.bundlesdf_pose_dir(dataset)
        self.annotated_dir = file_utils.bundlesdf_annotated_poses_dir(dataset)
        self.contactnets_dir = file_utils.contactnets_input_dir(self.toss_type)
        
    def _load_poses(self, bsdf_times: np.ndarray) -> None:
        """Load the timestamped poses reported from TagSLAM and BundleSDF,
        saving the results in attributes:
            - self.tagslam_full_times
            - self.tagslam_full_poses
            - self.bundlesdf_full_times
            - self.bundlesdf_full_poses
        
        The TagSLAM information comes from the ground truth pose directory's
        file tagslam.txt, which is generated by rosbag_processor.py's
        extract_time_versus_poses.  The BundleSDF information comes from the
        BundleSDF output directory's XXXX.txt files, which are generated by
        running BundleSDF and are timestamped according to this method's
        provided bsdf_times (which come from the timestamps associated with the
        RGBD images).

        Inputs:
            bsdf_times (N,)
        """
        self._load_tagslam_poses()
        self._load_bundlesdf_poses(bsdf_times)

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

    def _load_bundlesdf_poses(self, timestamps: np.ndarray) -> None:
        """Load all the poses reported by BundleSDF.  These are in world
        coordinates of the **TagSLAM body origin (converted from BundleSDF
        origin in cameracoordinates via math_utils.transform_bundletrack_output)
        with the following ordering:
            [x, y, z, qx, qy, qz, qw]
        """
        bundlesdf_poses, bundlesdf_times = [], []

        for i in range(1, len(timestamps)):
            trans_mat = np.loadtxt(op.join(self.bundlesdf_dir, "%04i.txt" % i))
            trans_mat = math_utils.transform_bundletrack_output(
                trans_mat, self.bundlesdf_dir, self.annotated_dir,
                self.cam_trans, self.cam_rot_axis_angle, to_world=True
            )
            pos_quat = math_utils.trans_mat_to_pos_quat(trans_mat).reshape(7)
            bundlesdf_poses.append(pos_quat)
            bundlesdf_times.append(timestamps[i])

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
            adjust_position=ADJUST_POSITION
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

        # Adjust position height if desired.
        if adjust_position:
            self.z_shift -= ps[-1, 2]
        ps[:, 2] += self.z_shift

        # Calculate derivatives.
        vs = self._estimate_linear_velocities(ts=t, ps=ps, filter=filter_lin_vel)
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
            - self.tagslam_toss_processed_states
            - self.bundlesdf_toss_processed_states
            - self.tagslam_toss_times
            - self.bundlesdf_toss_times
        """
        # Process TagSLAM and BundleSDF data.
        q_ts, p_ts, w_ts, v_ts = self._process_poses(
            self.tagslam_full_poses, self.tagslam_full_times,
            adjust_position=True)
        q_bsdf, p_bsdf, w_bsdf, v_bsdf = self._process_poses(
            self.bundlesdf_full_poses, self.bundlesdf_full_times,
            adjust_position=False)
        
        self.tagslam_full_processed_states = np.concatenate(
            (q_ts, p_ts, w_ts, v_ts), axis=1)
        self.bundlesdf_full_processed_states = np.concatenate(
            (q_bsdf, p_bsdf, w_bsdf, v_bsdf), axis=1)
        
        # Store trimmed trajectories for toss only.
        self._trim_processed_trajectories()

        # Print information about the trimmed trajectories.
        tagslam_toss_dts = np.mean(
            self.tagslam_toss_times[1:] - self.tagslam_toss_times[:-1]
        )
        print(f'TagSLAM toss trajectory information:' + \
              f'\n\t{self.tagslam_toss_processed_states.shape=}' + \
              f'\n\t{self.tagslam_toss_times[0]=}' + \
              f'\n\tAverage frame rate (toss): {1/tagslam_toss_dts}\n')

        bsdf_toss_dts = np.mean(
            self.bundlesdf_toss_times[1:] - self.bundlesdf_toss_times[:-1]
        )
        print(f'BundleSDF toss trajectory information:' + \
              f'\n\t{self.bundlesdf_toss_processed_states.shape=}' + \
              f'\n\t{self.bundlesdf_toss_times[0]=}' + \
              f'\n\tAverage frame rate (toss): {1/bsdf_toss_dts}\n')

    def _trim_processed_trajectories(self) -> None:
        """After trajectories are already processed, store trimmed versions of
        them corresponding to autonomous dynamics throughout a toss."""
        # Need to do one less than provided start and end frames because loaded
        # data in 1-indexed directory but provided 0-indexed start_frame and
        # end_frame.
        b_start = self.start_frame-1
        b_end = self.end_frame-1
        self.bundlesdf_toss_processed_states = \
            self.bundlesdf_full_processed_states[b_start:b_end]
        self.bundlesdf_toss_times = self.bundlesdf_full_times[b_start:b_end]

        # Find the start and end frames that are most synchronized in time with
        # those pre-selected for BundleSDF trajectories.
        t_start = np.argmin(
            (self.tagslam_full_times - self.bundlesdf_toss_times[0])**2)
        t_end = t_start + (b_end - b_start)
        self.tagslam_toss_processed_states = \
            self.tagslam_full_processed_states[t_start:t_end]
        self.tagslam_toss_times = self.tagslam_full_times[t_start:t_end]
        
    def plot_trajectory(self, full_trajectory: bool = True) -> None:
        """Visualize the BundleSDF and TagSLAM trajectories overlayed on a set
        of plots."""
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

        else:
            q_ts = self.tagslam_toss_processed_states[:, 0:4]
            p_ts = self.tagslam_toss_processed_states[:, 4:7]
            w_ts = self.tagslam_toss_processed_states[:, 7:10]
            v_ts = self.tagslam_toss_processed_states[:, 10:13]
            t_ts = self.tagslam_toss_times

            q_bsdf = self.bundlesdf_toss_processed_states[:, 0:4]
            p_bsdf = self.bundlesdf_toss_processed_states[:, 4:7]
            w_bsdf = self.bundlesdf_toss_processed_states[:, 7:10]
            v_bsdf = self.bundlesdf_toss_processed_states[:, 10:13]
            t_bsdf = self.bundlesdf_toss_times

            title='Toss Trajectory Results'

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
        # plt.savefig(f'bundlesdf_{TOSS_TYPE}_traj_{TOSS_ID}_tagslam.png')
        # print(f'Saved fig bundlesdf_{TOSS_TYPE}_traj_{TOSS_ID}_tagslam.png')
        if self.plot:
            plt.show()
        plt.close()

    def save_data(self, save_tagslam: bool = False,
                  save_bundlesdf: bool = False) -> None:
        """Stores data as .pt files in the ContactNets input directory."""
        traj_filename = f'{self.toss_id - 1}.pt'

        print('Saving files summary:')

        if save_tagslam:
            full_tagslam_dir = file_utils.contactnets_input_dir_tagslam(
                self.toss_type, full=True)
            toss_tagslam_dir = file_utils.contactnets_input_dir_tagslam(
                self.toss_type, full=False)
            torch.save(
                torch.tensor(self.tagslam_full_processed_states),
                op.join(full_tagslam_dir, traj_filename))
            torch.save(
                torch.tensor(self.tagslam_toss_processed_states),
                op.join(toss_tagslam_dir, traj_filename))
            print(f'\t{op.join(full_tagslam_dir, traj_filename)}.')
            print(f'\t{op.join(toss_tagslam_dir, traj_filename)}.')

        if save_bundlesdf:
            full_bundlesdf_dir = file_utils.contactnets_input_dir_bundlesdf(
                self.toss_type, iteration=self.iteration_num, full=True)
            toss_bundlesdf_dir = file_utils.contactnets_input_dir_bundlesdf(
                self.toss_type, iteration=self.iteration_num, full=False)
            torch.save(
                torch.tensor(self.bundlesdf_full_processed_states),
                op.join(full_bundlesdf_dir, traj_filename))
            torch.save(
                torch.tensor(self.bundlesdf_toss_processed_states),
                op.join(toss_bundlesdf_dir, traj_filename))
            print(f'\t{op.join(full_bundlesdf_dir, traj_filename)}.')
            print(f'\t{op.join(toss_bundlesdf_dir, traj_filename)}.')


#######################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--toss_id",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--type",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--zshift",
        type=float,
        default=0.05148739950625105,
        help="Offset from the table to the origin of the data"
    )
    parser.add_argument(
        "--iteration",
        type=int,
        default=1,
        help="BundleSDF/ContactNets cycle iteration"
    )
    args = parser.parse_args()

    toss_id = args.toss_id
    toss_type = args.type
    z_shift = args.zshift
    iteration_num = args.iteration

    dataset = f'{toss_type}_{toss_id}'
    rosbag_number = file_utils.load_dataset_from_yaml(toss_type,
                                                      toss_id)
    depth_bag_file = f"./rosbags/raw_{rosbag_number}.bag"
    odom_bag_file = f"./rosbags/odom_{rosbag_number}.bag"
    odom_ros_topic = f"/tagslam/odom/body_{toss_type}"
    tagslam_dir = file_utils.tagslam_pose_dir(dataset)
    annotated_poses_dir = file_utils.bundlesdf_annotated_poses_dir(dataset)

    print(f'Processing toss {toss_type}_{toss_id} in raw_{rosbag_number}.bag')

    # Get the camera extrinsics.
    cam_trans, cam_rot_axis_angle = file_utils.load_camera_extrinsics(toss_type)
    
    # Start/end times are for the start and end of a BundleSDF trajectory, which
    # starts with the object unmoving on the table, includes the toss wind-up
    # and execution, and ends with the object unmoving on the table again.
    start_time = file_utils.load_toss_time_from_yaml(toss_type, 
                                                     toss_id, 'start_time')
    end_time = file_utils.load_toss_time_from_yaml(toss_type,
                                                   toss_id, 'end_time')
    
    # Start/end frames are the indices of the longer BundleSDF trajectories that
    # correspond to the ContactNets trajectories, which include only the
    # autonomous dynamics of the object dropping under gravity and colliding
    # with the table.
    start_frame = file_utils.load_field_from_yaml(toss_type, toss_id,
                                                  'start_frame')
    end_frame = file_utils.load_field_from_yaml(toss_type, toss_id,
                                                'end_frame')

    # Rosbag processor extracts times associated with eventual BundleSDF poses
    # based on the times for every depth image from the depth bag.  The below
    # call additionally writes a tagslam.txt file that grabs TagSLAM poses from
    # the odom bag and their associated timestamps.
    bundletrack_time = rosbag_processor.extract_time_versus_poses(
        start_time, end_time, depth_bag_file, odom_bag_file, odom_ros_topic,
        tagslam_dir, save=True).reshape(-1,)

    # Write annotated_poses/0000.txt file, which stores the first pose of the
    # TagSLAM origin in camera frame (obtained by converting TagSLAM output).
    rosbag_processor.save_initial_tagslam_pose_in_camera_frame(
        tagslam_world_poses_dir=tagslam_dir,
        annotated_poses_dir=annotated_poses_dir, cam_trans=cam_trans,
        cam_rot_axis_angle=cam_rot_axis_angle
    )

    dataset = DatasetManagement(
        start_frame, end_frame, bundletrack_time, toss_id, toss_type,
        iteration_num, cam_trans, cam_rot_axis_angle, frame_rate=30,
        z_shift=z_shift, plot=True
    )

    dataset.do_process()
    dataset.plot_trajectory(full_trajectory=True)
    dataset.plot_trajectory(full_trajectory=False)

    dataset.save_data(save_tagslam=True, save_bundlesdf=True)
