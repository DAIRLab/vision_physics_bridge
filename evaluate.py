"""An evaluation script to inspect and analyze the results from the BundleSDF-
PLL cyclic pipeline."""

import click
import matplotlib.pyplot as plt
import numpy as np
import os
import os.path as op
import pdb
import torch
from torch import Tensor
import trimesh

import eval_utils, file_utils, icp, math_utils, mesh_processing

from conversion_bsdf_to_pll import TrajectoryConverterBundleSDFToPLL
from eval_utils import MultibodyLearnableSystem



INERTIA_THETA_KEY = 'multibody_terms.lagrangian_terms.inertial_parameters'
FRICTION_KEY = 'multibody_terms.contact_terms.friction_params'


def traverse_run_history_from_bsdf(
        vision_asset: str, bundlesdf_id: str, cycle_iteration: int) -> dict:
    """Return a dictionary of of the run history of a given run of BundleSDF.
    The dictionary will initially have keys of cycle_iteration_{i} for the given
    cycle iteration.  Each entry of dict[f'cycle_iteration_{i}'] is another
    dictionary with keys 'bundlesdf_id' and 'pll_id'.  The highest cycle
    iteration of BundleSDF will have a 'pll_id' of None.

    Here's an example of the returned dictionary for a BundleSDF run that
    terminated after cycle iteration 3:
    
        {'cycle_iteration_1': {'bundlesdf_id': bundlesdf_id_X,
                               'pll_id': pll_id_X},
         'cycle_iteration_2': {'bundlesdf_id': bundlesdf_id_Y,
                               'pll_id': pll_id_Y},
         'cycle_iteration_3': {'bundlesdf_id': bundlesdf_id_Z,
                               'pll_id': None}
        }
    """
    history_dict = {f'cycle_iteration_{cycle_iteration}': {
        'bundlesdf_id': bundlesdf_id, 'pll_id': None}}

    # First, handle the cycle = 1 case, which means there should be no former
    # associated PLL run.
    if cycle_iteration == 1:
        return history_dict

    # For higher cycles, find the associated PLL run.

    # Get the PLL ID from the BundleSDF run's config_nerf.yml.
    bundlesdf_results_dir = file_utils.bundlesdf_run_results_dir(
        vision_asset, cycle_iteration=cycle_iteration, bundlesdf_id=bundlesdf_id
    )
    pll_id = file_utils.load_pll_id_from_bundlesdf_yml(bundlesdf_results_dir)
    assert pll_id is not None, f'{cycle_iteration=} but found no PLL ID for' + \
        f' {vision_asset=} with {bundlesdf_id=}.'

    history_dict[f'cycle_iteration_{cycle_iteration-1}'] = {
        'bundlesdf_id': None, 'pll_id': pll_id
    }

    # Traverse one level lower.
    pll_output_dir = file_utils.contactnets_output_dir(
        vision_asset, cycle_iteration=cycle_iteration-1, pll_id=pll_id)
    former_bundlesdf_id = file_utils.load_bundlesdf_id_from_pll_json(
        pll_output_dir)
    former_history_dict = traverse_run_history_from_bsdf(
        vision_asset, former_bundlesdf_id, cycle_iteration-1)

    # Take care to update the history dictionary.
    for cycle_key, cycle_info in former_history_dict.items():
        if cycle_key not in history_dict.keys():
            history_dict[cycle_key] = cycle_info
        else:
            assert history_dict[cycle_key]['bundlesdf_id'] is None, \
                f'Found a BundleSDF ID for {cycle_key=} in {vision_asset=} ' + \
                f'with {cycle_iteration=}.'
            history_dict[cycle_key]['bundlesdf_id'] = cycle_info['bundlesdf_id']

    return history_dict

def traverse_run_history_from_pll(
        vision_asset: str, pll_id: str, cycle_iteration: int) -> dict:
    """From a PLL ID, create run history dictionary with same format as from
    traverse_run_history_from_bsdf.  The highest cycle iteration of PLL will
    have the provided PLL ID."""
    # Get the BundleSDF ID associated with the PLL run.
    pll_output_dir = file_utils.contactnets_output_dir(
        vision_asset, cycle_iteration=cycle_iteration, pll_id=pll_id)
    bundlesdf_id = file_utils.load_bundlesdf_id_from_pll_json(
        pll_output_dir)

    # If at lowest cycle iteration, history only includes the BundleSDF ID used
    # to provide PLL with trajectories.
    if cycle_iteration == 1:
        history_dict = {f'cycle_iteration_{cycle_iteration}': {
            'bundlesdf_id': bundlesdf_id, 'pll_id': pll_id}}

    # Otherwise, use the BundleSDF ID to traverse lower.
    else:
        history_dict = traverse_run_history_from_bsdf(
            vision_asset, bundlesdf_id, cycle_iteration)
        history_dict[f'cycle_iteration_{cycle_iteration}']['pll_id'] = pll_id

    return history_dict


class TrajectoryPerformanceEvaluator:
    """Trajectory-related metrics."""
    def __init__(self, vision_asset: str, history: dict, nerf_bundlesdf_id: str,
                 bsdf_only: bool):
        # First decode the system and start/end tosses from the provided asset
        # directory.
        self.object = vision_asset.split('_')[0]

        start_toss = int(vision_asset.split('_')[1].split('-')[0])
        end_toss = start_toss if '-' not in vision_asset else \
            int(vision_asset.split('-')[1])
        assert start_toss <= end_toss, f'Invalid toss range: {start_toss} ' + \
                f'-{end_toss} inferred from {vision_asset=}.'
        self.start_toss = start_toss
        self.end_toss = end_toss

        last_bsdf_iteration = 1
        for cycle in history.keys():
            cycle_num = cycle.split('_')[-1]
            if int(cycle_num) > last_bsdf_iteration:
                last_bsdf_iteration = int(cycle_num)
        self.last_bsdf_iteration = last_bsdf_iteration
        self.last_tracking_bsdf_id = history[
            f'cycle_iteration_{last_bsdf_iteration}']['bundlesdf_id']

        self.vision_asset = vision_asset
        self.nerf_bundlesdf_id = nerf_bundlesdf_id
        self.history = history
        self.bsdf_only = bsdf_only

        self.pll_id = history[
            f'cycle_iteration_{last_bsdf_iteration}']['pll_id']
        if self.pll_id is not None:
            self.pll_last_tracking_bsdf_id = self.last_tracking_bsdf_id
            self.last_tracking_bsdf_id = None
            self.nerf_bundlesdf_id = None

        self.eval_dir = file_utils.evaluation_subdir(
            dataset=self.vision_asset, cycle_iteration=self.last_bsdf_iteration,
            tracking_bundlesdf_id=self.last_tracking_bsdf_id,
            nerf_bundlesdf_id=self.nerf_bundlesdf_id, pll_id=self.pll_id
        )

    def get_tracking_trajectories(self):
        """Creates the following attributes, all of which are dictionaries with
        keys e.g. 'cycle_iteration_1' and values that are described below:
            - bundlesdf_full_times:  (N,)
            - bundlesdf_toss_times:  List of length n of (M_i,) arrays
            - bsdf_b_full_states:  (N, 13)
            - bsdf_b_toss_states:  List of length n of (M_i, 13) arrays

        If not self.bsdf_only, also creates the following attributes with the
        same structure as above:
            - tagslam_full_times:  (N,)
            - tagslam_toss_times:  List of length n of (M_i,) arrays
            - tagslam_full_states:  (N, 13)
            - bsdf_t_full_states:  (N, 13)
            - tagslam_toss_states:  List of length n of (M_i, 13) arrays
            - bsdf_t_toss_states:  List of length n of (M_i, 13) arrays
        """
        # Prepare to get the non-TagSLAM-related information.
        self.bundlesdf_full_times = {}
        self.bundlesdf_toss_times = {}
        self.bsdf_b_full_states = {}
        self.bsdf_b_toss_states = {}

        # Prepare to get the TagSLAM-related information, if available.
        if not self.bsdf_only:
            self.tagslam_full_times = {}
            self.tagslam_toss_times = {}
            self.tagslam_b_full_states = {}
            self.tagslam_b_toss_states = {}

        # Go through every cycle iteration and get the trajectory information.
        for cycle_label, cycle_info in self.history.items():
            bundlesdf_id = cycle_info['bundlesdf_id']
            cycle_iteration = int(cycle_label.split('_')[-1])

            # Start times are for the start and end of a BundleSDF trajectory,
            # which starts with the object unmoving on the table, includes the
            # toss windup and execution, and ends with the object unmoving on
            # the table again.
            start_ros_times = np.array([file_utils.load_toss_time_from_yaml(
                self.object, toss_i, 'start_time', as_ros_time=True) for \
                    toss_i in range(self.start_toss, self.end_toss+1)])

            # Start/end frames are the indices of the longer BundleSDF
            # trajectories that correspond to the ContactNets trajectories,
            # which include only the autonomous dynamics of the object dropping
            # under gravity and colliding with the table.
            relative_start_frames = np.array([file_utils.load_field_from_yaml(
                self.object, toss_i, 'start_frame') for toss_i in range(
                    self.start_toss, self.end_toss+1)])
            relative_end_frames = np.array([file_utils.load_field_from_yaml(
                self.object, toss_i, 'end_frame') for toss_i in range(
                    self.start_toss, self.end_toss+1)])

            # Get the table height.  Use the average if using multiple tosses.
            table_heights = np.array([
                file_utils.load_table_z_height(self.object, toss) for toss in
                range(self.start_toss, self.end_toss+1)
            ])
            z_table = np.mean(table_heights)

            # Get the camera intrinsics and extrinsics.
            cam_trans, cam_axis_vec = \
                file_utils.load_camera_extrinsics(self.object)

            traj_conv = TrajectoryConverterBundleSDFToPLL(
                tracking_bundlesdf_id=bundlesdf_id,
                nerf_bundlesdf_id=bundlesdf_id,  # TODO want to change?
                relative_start_frames=relative_start_frames,
                relative_end_frames=relative_end_frames,
                start_ros_times=start_ros_times, start_toss=self.start_toss,
                end_toss=self.end_toss, object=self.object,
                cycle_iteration=cycle_iteration, bsdf_only=self.bsdf_only,
                cam_trans=cam_trans, cam_rot_axis_angle=cam_axis_vec,
                frame_rate=30, z_table=z_table, plot=False
            )
            traj_conv.do_process()

            # Get the non-TagSLAM-related information.
            self.bundlesdf_full_times[cycle_label] = \
                traj_conv.bundlesdf_full_times
            self.bundlesdf_toss_times[cycle_label] = \
                traj_conv.bundlesdf_toss_times
            self.bsdf_b_full_states[cycle_label] = \
                traj_conv.bundlesdf_b_full_processed_states
            self.bsdf_b_toss_states[cycle_label] = \
                traj_conv.bundlesdf_b_toss_processed_states

            # Get the TagSLAM-related information, if available.
            if not self.bsdf_only:
                self.tagslam_full_times[cycle_label] = \
                    traj_conv.tagslam_full_times
                self.tagslam_toss_times[cycle_label] = \
                    traj_conv.tagslam_toss_times
                self.tagslam_b_full_states[cycle_label] = \
                    traj_conv.tagslam_b_full_processed_states
                self.tagslam_b_toss_states[cycle_label] = \
                    traj_conv.tagslam_b_toss_processed_states

    def _get_learned_pll_system(self):
        """Get a PLL system with the learned parameters, including geometry from
        BundleSDF."""
        if not hasattr(self, 'learned_pll_system'):
            # Create the learned system.
            dynamics_predictor = DynamicsPredictor(
                self.vision_asset, self.history, self.nerf_bundlesdf_id,
                self.bsdf_only)
            dynamics_predictor._create_pll_sim_system()
            self.learned_pll_system = dynamics_predictor.pll_system

        return self.learned_pll_system

    def _get_aligned_true_cloud(self):
        if not hasattr(self, 'aligned_true_cloud'):
            self._get_true_geometry_pll_system()
        return self.aligned_true_cloud

    def _write_aligned_true_geometry_obj(self, obj_name: str):
        """Use the MeshProcessor class to align the ground truth mesh to the
        BundleSDF-generated mesh."""
        if self.pll_id is None:
            tracking_bundlesdf_id = self.last_tracking_bsdf_id
            nerf_bundlesdf_id = self.nerf_bundlesdf_id
        else:
            tracking_bundlesdf_id = self.pll_last_tracking_bsdf_id
            nerf_bundlesdf_id = self.pll_last_tracking_bsdf_id

        mesh_processor = mesh_processing.MeshProcessor(
            vision_asset=self.vision_asset,
            tracking_bundlesdf_id=tracking_bundlesdf_id,
            nerf_bundlesdf_id=nerf_bundlesdf_id,
            cycle_iteration=self.last_bsdf_iteration
        )
        mesh_processor.align_true_to_learned_mesh_with_icp(
            show=False, save_dir=self.eval_dir, obj_name=obj_name)

        # Store the points sampled on the true aligned geometry.
        point_cloud_object = mesh_processor.aligned_true_cloud
        self.aligned_true_cloud = np.asarray(point_cloud_object.points)

    def _get_true_geometry_pll_system(self):
        """Get a PLL system with the learned parameters but true geometry."""
        if not hasattr(self, 'true_geom_pll_system'):
            if self.last_bsdf_iteration > 1:
                pll_iteration = self.last_bsdf_iteration - 1
                pll_id = self.history[f'cycle_iteration_{pll_iteration}'][
                    'pll_id']
                pll_results_dir = file_utils.contactnets_output_dir(
                    dataset=self.vision_asset, cycle_iteration=pll_iteration,
                    pll_id=pll_id)

                # Get the old URDF from the PLL results.
                old_urdf_path = op.join(
                    pll_results_dir, 'urdfs', 'with_bundlesdf_mesh.urdf')

            else:
                # Use the original URDF.
                old_urdf_path = file_utils.template_urdf_filepath()

            # First create a URDF.
            new_urdf_path = op.join(self.eval_dir, 'bsdf_mesh_pll_params.urdf')
            os.system(f'cp {old_urdf_path} {new_urdf_path}')

            old_obj_path = file_utils.object_scan_filepath(self.object)
            new_obj_path = op.join(self.eval_dir, 'true_geom_aligned.obj')
            os.system(f'cp {old_obj_path} {new_obj_path}')

            # Align the true geometry to the BSDF geometry.
            self._write_aligned_true_geometry_obj(
                obj_name='true_geom_aligned.obj')

            # Overwrite the geometry in the URDF to refer to the new obj.
            eval_utils.overwrite_mesh_name_in_urdf(
                new_urdf_path, 'true_geom_aligned.obj')
            print(f'Wrote URDF to {new_urdf_path}')

            # Create the system.
            self.true_geom_pll_system = \
                eval_utils.create_multibody_learnable_system(new_urdf_path)

        return self.true_geom_pll_system

    def compute_metrics(self):
        """Metrics to include:
            - positional error over trajectory
            - orientation error over trajectory
            - ADD (requires mesh)
            - ADD-S (requires mesh)
            - penetration, 2 ways (requires mesh)

        Note:  All trajectories are in PLL format, which is:
        [ qw qx qy qz  x y z  wx wy wz  vx vy vz ]

        Creates the following attributes, all of which are dictionaries with
        keys e.g. 'cycle_iteration_1' and values that are described below:
            - bundlesdf_full_times:  (N,)
            - bundlesdf_toss_times:  List of length n of (M_i,) arrays
            - bsdf_b_full_states:  (N, 13)
            - bsdf_b_toss_states:  List of length n of (M_i, 13) arrays

        If not self.bsdf_only, also creates the following attributes with the
        same structure as above:
            - tagslam_full_times:  (N,)
            - tagslam_toss_times:  List of length n of (M_i,) arrays
            - tagslam_full_states:  (N, 13)
            - bsdf_t_full_states:  (N, 13)
            - tagslam_toss_states:  List of length n of (M_i, 13) arrays
            - bsdf_t_toss_states:  List of length n of (M_i, 13) arrays
        """
        # Do a test with bsdf_t_full_states and tagslam_full_states.
        cycle_key = f'cycle_iteration_{self.last_bsdf_iteration}'
        target_traj = Tensor(self.tagslam_full_states[cycle_key])
        pred_traj = Tensor(self.bsdf_t_full_states[cycle_key])

        pos_error = self._compute_pos_error(target_traj, pred_traj)
        rot_error = self._compute_rot_error(target_traj, pred_traj)
        pen_true_geom_error = self._compute_pen_error_true_geom_predicted_traj(
            pred_traj)
        pen_true_traj_error = self._compute_pen_error_true_geom_predicted_traj(
            target_traj)
        add_error = self._compute_add_error(target_traj, pred_traj)
        adds_error = self._compute_adds_error(target_traj, pred_traj)
        pdb.set_trace()

        self.visualize_metrics(pos_error, rot_error, pen_true_geom_error,
                               pen_true_traj_error, add_error, adds_error)

    def _compute_pos_error(self, target_traj: Tensor, pred_traj: Tensor):
        return TrajectoryMetrics.position_error(target_traj, pred_traj)

    def _compute_rot_error(self, target_traj: Tensor, pred_traj: Tensor):
        return TrajectoryMetrics.rotation_error(target_traj, pred_traj)

    def _compute_pen_error_true_geom_predicted_traj(self, pred_traj: Tensor):
        """Returns the mean penetration of the real geometry at each point over
        the predicted trajectory."""
        true_geom_system = self._get_true_geometry_pll_system()
        return TrajectoryMetrics.penetration(pred_traj, true_geom_system)

    def _compute_pen_error_learned_geom_real_traj(self, target_traj: Tensor):
        """Returns the penetration of the learned geometry at each point over
        the target trajectory."""
        pred_geom_system = self._get_learned_pll_system()
        return TrajectoryMetrics.penetration(target_traj, pred_geom_system)

    def _compute_add_error(self, target_traj: Tensor, pred_traj: Tensor):
        """Returns the average distance of 1-to-1 mapped object surface points
        from one pose to another pose, at each point in the trajectories.  These
        surface points are sampled from the true geometry."""
        assert hasattr(self, 'aligned_true_cloud'), f'Need to run ' + \
            f'self._write_aligned_true_geometry_obj() first so ' + \
            f'aligned_true_cloud attribute exists.'

        return TrajectoryMetrics.add_error(
            target_traj, pred_traj, self.aligned_true_cloud)

    def _compute_adds_error(self, target_traj: Tensor, pred_traj: Tensor):
        """Returns the average distance from each point sampled on the true
        geometry at a predicted pose to the nearest point at a true pose, at
        each point in the trajectories."""
        assert hasattr(self, 'aligned_true_cloud'), f'Need to run ' + \
            f'self._write_aligned_true_geometry_obj() first so ' + \
            f'aligned_true_cloud attribute exists.'

        return TrajectoryMetrics.adds_error(
            target_traj, pred_traj, self.aligned_true_cloud)

    def visualize_metrics(self, pos_error, rot_error, pen_true_geom_error,
                          pen_true_traj_error, add_error, adds_error):
        """Visualize the metrics."""
        plt.ion()
        fig = plt.figure()

        ax = fig.add_subplot(221)
        ax.plot(pos_error, label='Position error')
        ax.legend()
        ax.set_ylabel('Position [m]')

        ax = fig.add_subplot(222)
        ax.plot(rot_error*180/np.pi, label='Rotation error')
        ax.legend()
        ax.set_ylabel('Rotation [deg]')

        ax = fig.add_subplot(223)
        ax.plot(add_error, label='ADD error')
        ax.plot(adds_error, label='ADD-S error')
        ax.legend()
        ax.set_ylabel('Average distance [m]')

        ax = fig.add_subplot(224)
        ax.plot(pen_true_geom_error,
                label='True geometry on predicted trajectory')
        ax.plot(pen_true_traj_error,
                label='Learned geometry on true trajectory')
        ax.legend()
        ax.set_ylabel('Penetration [m]')

        pdb.set_trace()

    def store_tracking_metrics(self, results: dict) -> None:
        """Store the tracking metrics in the results dictionary."""
        cycle_key = f'cycle_iteration_{self.last_bsdf_iteration}'

        # First do everything against BundleSDF.
        sub_results = results['tracking_metrics']['against_bundlesdf']

        penetration_true_geom = sub_results[
            'penetration_true_geom_estimated_traj']
        for toss_key, subsub_results in penetration_true_geom.items():
            if 'toss' in toss_key:
                toss_num = int(toss_key.split('_')[1])
                traj = self.bsdf_b_toss_states[cycle_key][
                    toss_num - self.start_toss]
            else:
                traj = self.bsdf_b_full_states[cycle_key]
            over_traj = TrajectoryMetrics.penetration(
                traj, self._get_true_geometry_pll_system())
            subsub_results['mean'] = over_traj.mean().item()
            subsub_results['traj'] = over_traj.tolist()

        penetration_learned_geom = sub_results[
            'penetration_learned_geom_estimated_traj']
        for toss_key, subsub_results in penetration_learned_geom.items():
            if 'toss' in toss_key:
                toss_num = int(toss_key.split('_')[1])
                traj = self.bsdf_b_toss_states[cycle_key][
                    toss_num - self.start_toss]
            else:
                traj = self.bsdf_b_full_states[cycle_key]
            over_traj = TrajectoryMetrics.penetration(
                traj, self._get_learned_pll_system())
            subsub_results['mean'] = over_traj.mean().item()
            subsub_results['traj'] = over_traj.tolist()

        # Second do everything against TagSLAM.
        if not 'against_tagslam' in results['tracking_metrics'].keys():
            return
        assert hasattr(self, 'tagslam_b_full_states'), f'Expected to have ' + \
            f'TagSLAM trajectories stored since {results.keys()=}.'

        sub_results = results['tracking_metrics']['against_tagslam']

        position_results = sub_results['position_error']
        for toss_key, subsub_results in position_results.items():
            if 'toss' in toss_key:
                toss_num = int(toss_key.split('_')[1])
                bsdf_traj = self.bsdf_b_toss_states[cycle_key][
                    toss_num - self.start_toss]
                tagslam_traj = self.tagslam_b_toss_states[cycle_key][
                    toss_num - self.start_toss]
            else:
                bsdf_traj = self.bsdf_b_full_states[cycle_key]
                tagslam_traj = self.tagslam_b_full_states[cycle_key]
            over_traj = TrajectoryMetrics.position_error(
                bsdf_traj, tagslam_traj)
            subsub_results['mean'] = over_traj.mean().item()
            subsub_results['traj'] = over_traj.tolist()

        rotation_results = sub_results['rotation_error']
        for toss_key, subsub_results in rotation_results.items():
            if 'toss' in toss_key:
                toss_num = int(toss_key.split('_')[1])
                bsdf_traj = self.bsdf_b_toss_states[cycle_key][
                    toss_num - self.start_toss]
                tagslam_traj = self.tagslam_b_toss_states[cycle_key][
                    toss_num - self.start_toss]
            else:
                bsdf_traj = self.bsdf_b_full_states[cycle_key]
                tagslam_traj = self.tagslam_b_full_states[cycle_key]
            over_traj = TrajectoryMetrics.rotation_error(
                bsdf_traj, tagslam_traj)
            subsub_results['mean'] = over_traj.mean().item()
            subsub_results['traj'] = over_traj.tolist()

        add_error = sub_results['add_error']
        for toss_key, subsub_results in add_error.items():
            if 'toss' in toss_key:
                toss_num = int(toss_key.split('_')[1])
                bsdf_traj = self.bsdf_b_toss_states[cycle_key][
                    toss_num - self.start_toss]
                tagslam_traj = self.tagslam_b_toss_states[cycle_key][
                    toss_num - self.start_toss]
            else:
                bsdf_traj = self.bsdf_b_full_states[cycle_key]
                tagslam_traj = self.tagslam_b_full_states[cycle_key]
            over_traj = TrajectoryMetrics.add_error(
                bsdf_traj, tagslam_traj, self._get_aligned_true_cloud())
            subsub_results['mean'] = over_traj.mean().item()
            subsub_results['traj'] = over_traj.tolist()

        adds_error = sub_results['adds_error']
        for toss_key, subsub_results in adds_error.items():
            if 'toss' in toss_key:
                toss_num = int(toss_key.split('_')[1])
                bsdf_traj = self.bsdf_b_toss_states[cycle_key][
                    toss_num - self.start_toss]
                tagslam_traj = self.tagslam_b_toss_states[cycle_key][
                    toss_num - self.start_toss]
            else:
                bsdf_traj = self.bsdf_b_full_states[cycle_key]
                tagslam_traj = self.tagslam_b_full_states[cycle_key]
            over_traj = TrajectoryMetrics.adds_error(
                bsdf_traj, tagslam_traj, self._get_aligned_true_cloud())
            subsub_results['mean'] = over_traj.mean().item()
            subsub_results['traj'] = over_traj.tolist()

        penetration_learned_geom = sub_results[
            'penetration_learned_geom_tagslam_traj']
        for toss_key, subsub_results in penetration_learned_geom.items():
            if 'toss' in toss_key:
                toss_num = int(toss_key.split('_')[1])
                traj = self.tagslam_b_toss_states[cycle_key][
                    toss_num - self.start_toss]
            else:
                traj = self.tagslam_b_full_states[cycle_key]
            over_traj = TrajectoryMetrics.penetration(
                traj, self._get_learned_pll_system())
            subsub_results['mean'] = over_traj.mean().item()
            subsub_results['traj'] = over_traj.tolist()


class DynamicsPredictor:
    """Generate dynamics predictions."""
    def __init__(self, vision_asset: str, history: dict, nerf_bundlesdf_id: str,
                 bsdf_only: bool):
        # First decode the system and start/end tosses from the provided asset
        # directory.
        self.object = vision_asset.split('_')[0]

        start_toss = int(vision_asset.split('_')[1].split('-')[0])
        end_toss = start_toss if '-' not in vision_asset else \
            int(vision_asset.split('-')[1])
        assert start_toss <= end_toss, f'Invalid toss range: {start_toss} ' + \
                f'-{end_toss} inferred from {vision_asset=}.'
        self.start_toss = start_toss
        self.end_toss = end_toss

        self.vision_asset = vision_asset
        self.history = history
        self.bsdf_only = bsdf_only

        last_bsdf_iteration = 1
        for cycle in self.history.keys():
            cycle_num = cycle.split('_')[-1]
            if int(cycle_num) > last_bsdf_iteration:
                last_bsdf_iteration = int(cycle_num)
        self.last_bsdf_iteration = last_bsdf_iteration
        self.last_tracking_bsdf_id = self.history[
            f'cycle_iteration_{last_bsdf_iteration}']['bundlesdf_id']
        self.last_nerf_bsdf_id = nerf_bundlesdf_id

        self.pll_id = history[
            f'cycle_iteration_{last_bsdf_iteration}']['pll_id']
        if self.pll_id is not None:
            self.pll_last_tracking_bsdf_id = self.last_tracking_bsdf_id
            self.last_tracking_bsdf_id = None
            self.last_nerf_bsdf_id = None

        learned_params = self._look_up_latest_pll_results()
        learned_params.update(self._look_up_latest_bundlesdf_results())
        self.learned_params = learned_params

        self.eval_dir = file_utils.evaluation_subdir(
            dataset=self.vision_asset, cycle_iteration=self.last_bsdf_iteration,
            tracking_bundlesdf_id=self.last_tracking_bsdf_id,
            nerf_bundlesdf_id=self.last_nerf_bsdf_id, pll_id=self.pll_id
        )

    def _look_up_latest_pll_results(self):
        """Also stores self.pll_results_dir."""
        if self.last_bsdf_iteration == 1:
            self.pll_results_dir = file_utils.contactnets_output_dir(
                dataset=self.vision_asset,
                cycle_iteration=self.last_bsdf_iteration,
                pll_id=self.pll_id
            )

            print(f'No prior PLL results to look up for {self.vision_asset=}' +\
                  f' with {self.history=}.')
            return {}

        pll_iteration = self.last_bsdf_iteration - 1
        pll_id = self.history[f'cycle_iteration_{pll_iteration}']['pll_id']
        self.pll_results_dir = file_utils.contactnets_output_dir(
            dataset=self.vision_asset, cycle_iteration=pll_iteration,
            pll_id=pll_id)

        # TODO BIBIT decide if config/stats are necessary/useful
        config, stats, checkpoint = eval_utils.get_pll_config_stats_checkpoint(
            self.pll_results_dir)

        best_system_state = checkpoint['best_learned_system_state']
        params_dict = self._get_physical_parameters(
            best_system_state, self.pll_results_dir)
        # run_dict['learned_params'] = params_dict

        # init_params_dict = get_init_physical_parameters(
        #     system, body_names, checkpoint, wandb_api)
        # run_dict['initial_params'] = init_params_dict
        return params_dict

    def _look_up_latest_bundlesdf_results(self):
        """Also stores self.nerf_results_dir."""
        if self.pll_id is not None:
            return {}

        self.nerf_results_dir = file_utils.bundlesdf_nerf_results_dir(
            dataset=self.vision_asset, cycle_iteration=self.last_bsdf_iteration,
            tracking_bundlesdf_id=self.last_tracking_bsdf_id,
            nerf_bundlesdf_id=self.last_nerf_bsdf_id
        )
        geometry = trimesh.load(
            op.join(self.nerf_results_dir, 'textured_mesh.obj'), force='mesh')

        bsdf_params = {}
        bsdf_params['bsdf_geometry'] = geometry
        bsdf_params['bsdf_geometry_hull'] = geometry.convex_hull
        return bsdf_params

    def _get_physical_parameters(
            self, system_state: dict, pll_results_dir: str) -> dict:
        """Extract the physical parameters PLL learned from the given system
        state dictionary.  This dictionary will contain keys:
            - friction:  The friction coefficient.
            - mass:  The mass of the object.
            - com_x:  The x-coordinate of the center of mass.
            - com_y:  The y-coordinate of the center of mass.
            - com_z:  The z-coordinate of the center of mass.
            - I_xx:  The moment of inertia about the x-axis.
            - I_yy:  The moment of inertia about the y-axis.
            - I_zz:  The moment of inertia about the z-axis.
            - I_xy:  The moment of inertia about the xy-plane.
            - I_xz:  The moment of inertia about the xz-plane.
            - I_yz:  The moment of inertia about the yz-plane.
            - geometry:  The geometry of the object as a trimesh object.
        """
        learned_params = {}

        # =========== FRICTION: The friction is a single parameter, stored at
        # index 0 at the friction key value (index 1 is the ground).
        learned_params['friction'] = system_state[FRICTION_KEY][0].item()

        # =========== INERTIA: Interpret inertia into individual parts.
        inertia_theta = system_state[INERTIA_THETA_KEY]
        inertia_pi_cm = eval_utils.convert_inertia_theta_to_pi_cm(
            inertia_theta).squeeze()

        # Sadly the mass that is stored in this checkpoint is slightly
        # incorrect since it was not manually overwritten to be the original
        # value.  We can look this up in the URDF.
        urdf_path = op.join(
            pll_results_dir, 'urdfs', 'with_bundlesdf_mesh.urdf')
        mass = eval_utils.get_mass_from_urdf(urdf_path)

        # Reminder, pi_cm format is:
        # [m, m * p_x, m * p_y, m * p_z, I_xx, I_yy, I_zz, I_xy, I_xz, I_yz]

        # Divide out the mass.
        inertia_pi_cm[1:4] /= mass

        # Store the human-interpretable inertia parameters -- these should
        # exactly match the URDF.
        learned_params['mass'] = mass
        learned_params['com_x'] = inertia_pi_cm[1].item()
        learned_params['com_y'] = inertia_pi_cm[2].item()
        learned_params['com_z'] = inertia_pi_cm[3].item()
        learned_params['I_xx'] = inertia_pi_cm[4].item()
        learned_params['I_yy'] = inertia_pi_cm[5].item()
        learned_params['I_zz'] = inertia_pi_cm[6].item()
        learned_params['I_xy'] = inertia_pi_cm[7].item()
        learned_params['I_xz'] = inertia_pi_cm[8].item()
        learned_params['I_yz'] = inertia_pi_cm[9].item()

        # =========== GEOMETRY: Extract the geometry from the obj file.
        obj_path = op.join(pll_results_dir, 'urdfs', 'test.obj')
        if not op.exists(obj_path):
            obj_path = op.join(pll_results_dir, 'urdfs', 'test_best.obj')
        learned_params['pll_geometry'] = trimesh.load(obj_path, force='mesh')

        return learned_params

    def _create_pll_sim_system(self):
        """Create a PLL MultibodyLearnableSystem, which can be simulated."""
        # First create a URDF.  This should be the same as the last PLL URDF,
        # possibly with the geometry replaced by the new BSDF geometry if the
        # last run was BundleSDF and not PLL.
        if self.last_bsdf_iteration > 1:
            old_urdf_path = op.join(
                self.pll_results_dir, 'urdfs', 'with_bundlesdf_mesh.urdf')
        else:
            old_urdf_path = file_utils.template_urdf_filepath()

        new_urdf_path = op.join(self.eval_dir, 'bsdf_mesh_pll_params.urdf')
        os.system(f'cp {old_urdf_path} {new_urdf_path}')

        if self.pll_id is None:
            old_obj_path = op.join(self.nerf_results_dir, 'textured_mesh.obj')
            new_mesh_name = 'bsdf_mesh.obj'
        else:
            old_obj_path = op.join(self.pll_results_dir, 'urdfs', 'test.obj')
            if not op.exists(old_obj_path):
                old_obj_path = op.join(
                    self.pll_results_dir, 'urdfs', 'test_best.obj')
            new_mesh_name = 'pll_mesh.obj'
        new_obj_path = op.join(self.eval_dir, new_mesh_name)
        os.system(f'cp {old_obj_path} {new_obj_path}')

        # Overwrite the geometry in the URDF to refer to the new obj.
        eval_utils.overwrite_mesh_name_in_urdf(new_urdf_path, new_mesh_name)
        print(f'Wrote URDF to {new_urdf_path}')

        # Create the system.
        self.pll_system = eval_utils.create_multibody_learnable_system(
            new_urdf_path)

        # Export the URDF.
        self.pll_system.generate_updated_urdfs()

    def _get_tracked_trajectories(self):
        # Get the BundleSDF trajectories for each toss.
        if self.pll_id is not None:
            last_bsdf_id = self.pll_last_tracking_bsdf_id
        else:
            last_bsdf_id = self.last_tracking_bsdf_id
        self.bundlesdf_trajs = eval_utils.get_bundlesdf_trajectories_pll_format(
            self.vision_asset, cycle_iteration=self.last_bsdf_iteration,
            bundlesdf_id=last_bsdf_id
        )

        # Get ground truth trajectories from TagSLAM.
        tagslam_trajs = eval_utils.get_pll_tagslam_trajectories_pll_format(
            object=self.object)

        if tagslam_trajs is not None:
            # Convert the TagSLAM trajectories to be represented with respect to
            # the BundleSDF body origin.
            tagslam_trajs_of_b_origin = {}
            for toss_key, tagslam_traj in tagslam_trajs.items():
                # Get synchronized BundleSDF and TagSLAM poses.
                if toss_key in self.bundlesdf_trajs.keys():
                    print(f'Can synchronize toss {toss_key} with BundleSDF.')
                    b_mat = math_utils.pll_format_to_trans_mat(
                        self.bundlesdf_trajs[toss_key][0])
                    t_mat = math_utils.pll_format_to_trans_mat(tagslam_traj[0])

                else:
                    print(f'Need to synchronize at start for toss {toss_key}.')
                    b_mat, t_mat = \
                        eval_utils.get_synced_bsdf_tagslam_toss_poses(
                            vision_asset=self.vision_asset,
                            bundlesdf_id=last_bsdf_id,
                            cycle_iteration=self.last_bsdf_iteration,
                            desired_toss_num=toss_key
                        )

                # Do the conversion.
                tagslam_trajs_of_b_origin[toss_key] = \
                    math_utils.transform_t_origin_to_b_origin_pll_format(
                        full_tagslam_trajectory=tagslam_traj,
                        synced_bsdf_pose=b_mat,
                        synced_tagslam_pose=t_mat
                    )

            self.tagslam_b_trajs = tagslam_trajs_of_b_origin

    def generate_rollout_trajectories(self):
        """Generate rollouts for the object using the learned parameters."""
        # Create the simulation system.
        self._create_pll_sim_system()       # gets self.pll_system
        self._get_tracked_trajectories()    # gets self.bundlesdf_trajs,
                                            # self.tagslam_b_trajs (if exists)

        # Get the predictions.
        pred_trajs_of_b_origin = {}
        trajs = self.bundlesdf_trajs if not hasattr(self, 'tagslam_b_trajs') \
            else self.tagslam_b_trajs

        for toss_key, target_traj in trajs.items():
            start_adjust = file_utils.load_field_from_yaml(
                object=self.object, toss_number=toss_key, key='start_adjust')
            pred_trajs_of_b_origin[toss_key] = \
                eval_utils.get_pll_rollout_trajectory(
                    system=self.pll_system, target_traj=Tensor(target_traj),
                    start_adjust=start_adjust
                )

        # Store the targets and predictions.
        self.predicted_trajs = pred_trajs_of_b_origin

    def generate_single_step_predictions(self):
        """Generate single-step predictions for the object using the learned
        parameters."""
        # Get the predictions.
        pred_bsdf_steps = {}
        target_bsdf_steps = {}
        bundlesdf_trajs = self.bundlesdf_trajs

        for toss_key, target_bsdf_traj in bundlesdf_trajs.items():
            start_adjust = file_utils.load_field_from_yaml(
                object=self.object, toss_number=toss_key, key='start_adjust')
            pred_bsdf_steps[toss_key], target_bsdf_steps[toss_key] = \
                eval_utils.get_pll_single_step_predictions_and_targets(
                    system=self.pll_system, full_traj=Tensor(target_bsdf_traj),
                    start_adjust=start_adjust
                )

        # Store the targets and predictions.
        self.single_step_bsdf_predictions = pred_bsdf_steps
        self.single_step_bsdf_targets = target_bsdf_steps

        # Do the same thing against TagSLAM, if available.
        if not hasattr(self, 'tagslam_b_trajs'):
            return

        pred_tagslam_steps = {}
        target_tagslam_steps = {}
        bundlesdf_trajs = self.tagslam_b_trajs

        for toss_key, target_bsdf_traj in bundlesdf_trajs.items():
            start_adjust = file_utils.load_field_from_yaml(
                object=self.object, toss_number=toss_key, key='start_adjust')
            pred_tagslam_steps[toss_key], target_tagslam_steps[toss_key] = \
                eval_utils.get_pll_single_step_predictions_and_targets(
                    system=self.pll_system, full_traj=Tensor(target_bsdf_traj),
                    start_adjust=start_adjust
                )

        # Store the targets and predictions.
        self.single_step_tagslam_predictions = pred_tagslam_steps
        self.single_step_tagslam_targets = target_tagslam_steps

    def save_predictions(self):
        """Save the target and prediction trajectories to the evaluation
        directory."""
        print(f'Saving trajectories to {self.eval_dir}:')

        for toss_num, pred_traj in self.predicted_trajs.items():
            filename = f'predicted_toss_{toss_num}.pt'
            torch.save(pred_traj, op.join(self.eval_dir, filename))
            print(f'\t{filename}')

        for toss_num, target_traj in self.bundlesdf_trajs.items():
            filename = f'bundlesdf_toss_{toss_num}.pt'
            torch.save(target_traj, op.join(self.eval_dir, filename))
            print(f'\t{filename}')

        for toss_num, step_targets in self.single_step_bsdf_targets.items():
            filename = f'step_target_bsdf_toss_{toss_num}.pt'
            torch.save(step_targets, op.join(self.eval_dir, filename))
            print(f'\t{filename}')

        for toss_num, step_preds in self.single_step_bsdf_predictions.items():
            filename = f'step_prediction_bsdf_toss_{toss_num}.pt'
            torch.save(step_preds, op.join(self.eval_dir, filename))
            print(f'\t{filename}')

        if hasattr(self, 'tagslam_b_trajs'):
            for toss_num, target_traj in self.tagslam_b_trajs.items():
                filename = f'tagslam_b_toss_{toss_num}.pt'
                torch.save(target_traj, op.join(self.eval_dir, filename))
                print(f'\t{filename}')

            for toss_num, step_targets in \
                self.single_step_tagslam_targets.items():
                filename = f'step_target_tagslam_toss_{toss_num}.pt'
                torch.save(step_targets, op.join(self.eval_dir, filename))
                print(f'\t{filename}')

            for toss_num, step_preds in \
                self.single_step_tagslam_predictions.items():
                filename = f'step_prediction_tagslam_toss_{toss_num}.pt'
                torch.save(step_preds, op.join(self.eval_dir, filename))
                print(f'\t{filename}')

    def make_prediction_video(self):
        prediction_overlay = eval_utils.PredictionOverlayGenerator(
            vision_asset=self.vision_asset,
            history=self.history,
            nerf_bundlesdf_id=self.last_nerf_bsdf_id
        )
        prediction_overlay.make_overlay_video()

    def store_dynamics_metrics(
            self, results: dict, traj_evaluator: TrajectoryPerformanceEvaluator
    ) -> None:
        """Store the dynamics prediction metrics in the results dictionary."""
        # First do everything against BundleSDF.
        ### Rollout metrics.
        sub_results = results['dynamics_rollout_metrics']['against_bundlesdf']

        position_results = sub_results['position_error']
        for toss_key, subsub_results in position_results.items():
            toss_num = int(toss_key.split('_')[1])
            over_traj = TrajectoryMetrics.position_error(
                self.bundlesdf_trajs[toss_num], self.predicted_trajs[toss_num])
            subsub_results['mean'] = over_traj.mean().item()
            subsub_results['traj'] = over_traj.tolist()

        rotation_error = sub_results['rotation_error']
        for toss_key, subsub_results in rotation_error.items():
            toss_num = int(toss_key.split('_')[1])
            over_traj = TrajectoryMetrics.rotation_error(
                self.bundlesdf_trajs[toss_num], self.predicted_trajs[toss_num])
            subsub_results['mean'] = over_traj.mean().item()
            subsub_results['traj'] = over_traj.tolist()

        add_error = sub_results['add_error']
        for toss_key, subsub_results in add_error.items():
            toss_num = int(toss_key.split('_')[1])
            over_traj = TrajectoryMetrics.add_error(
                self.bundlesdf_trajs[toss_num], self.predicted_trajs[toss_num],
                traj_evaluator._get_aligned_true_cloud()
            )
            subsub_results['mean'] = over_traj.mean().item()
            subsub_results['traj'] = over_traj.tolist()

        adds_error = sub_results['adds_error']
        for toss_key, subsub_results in adds_error.items():
            toss_num = int(toss_key.split('_')[1])
            over_traj = TrajectoryMetrics.adds_error(
                self.bundlesdf_trajs[toss_num], self.predicted_trajs[toss_num],
                traj_evaluator._get_aligned_true_cloud()
            )
            subsub_results['mean'] = over_traj.mean().item()
            subsub_results['traj'] = over_traj.tolist()

        penetration_true_geom = sub_results[
            'penetration_true_geom_predicted_traj']
        for toss_key, subsub_results in penetration_true_geom.items():
            toss_num = int(toss_key.split('_')[1])
            over_traj = TrajectoryMetrics.penetration(
                self.predicted_trajs[toss_num],
                traj_evaluator._get_true_geometry_pll_system()
            )
            subsub_results['mean'] = over_traj.mean().item()
            subsub_results['traj'] = over_traj.tolist()

        ### Single-step metrics.
        sub_results = results['dynamics_single_step_metrics'][
            'against_bundlesdf']

        position_results = sub_results['position_error']
        for toss_key, subsub_results in position_results.items():
            toss_num = int(toss_key.split('_')[1])
            over_traj = TrajectoryMetrics.position_error(
                self.single_step_bsdf_targets[toss_num],
                self.single_step_bsdf_predictions[toss_num])
            subsub_results['mean'] = over_traj.mean().item()
            subsub_results['traj'] = over_traj.tolist()

        rotation_error = sub_results['rotation_error']
        for toss_key, subsub_results in rotation_error.items():
            toss_num = int(toss_key.split('_')[1])
            over_traj = TrajectoryMetrics.rotation_error(
                self.single_step_bsdf_targets[toss_num],
                self.single_step_bsdf_predictions[toss_num])
            subsub_results['mean'] = over_traj.mean().item()
            subsub_results['traj'] = over_traj.tolist()

        add_error = sub_results['add_error']
        for toss_key, subsub_results in add_error.items():
            toss_num = int(toss_key.split('_')[1])
            over_traj = TrajectoryMetrics.add_error(
                self.single_step_bsdf_targets[toss_num],
                self.single_step_bsdf_predictions[toss_num],
                traj_evaluator._get_aligned_true_cloud()
            )
            subsub_results['mean'] = over_traj.mean().item()
            subsub_results['traj'] = over_traj.tolist()

        adds_error = sub_results['adds_error']
        for toss_key, subsub_results in adds_error.items():
            toss_num = int(toss_key.split('_')[1])
            over_traj = TrajectoryMetrics.adds_error(
                self.single_step_bsdf_targets[toss_num],
                self.single_step_bsdf_predictions[toss_num],
                traj_evaluator._get_aligned_true_cloud()
            )
            subsub_results['mean'] = over_traj.mean().item()
            subsub_results['traj'] = over_traj.tolist()

        penetration_true_geom = sub_results[
            'penetration_true_geom_predicted_traj']
        for toss_key, subsub_results in penetration_true_geom.items():
            toss_num = int(toss_key.split('_')[1])
            over_traj = TrajectoryMetrics.penetration(
                self.single_step_bsdf_predictions[toss_num],
                traj_evaluator._get_true_geometry_pll_system()
            )
            subsub_results['mean'] = over_traj.mean().item()
            subsub_results['traj'] = over_traj.tolist()

        # Second do everything against TagSLAM.
        if not 'against_tagslam' in results['dynamics_rollout_metrics'].keys():
            return
        assert hasattr(self, 'tagslam_b_trajs'), f'Expected to have ' + \
            f'TagSLAM trajectories stored since {results.keys()=}.'

        ### Rollout metrics.
        sub_results = results['dynamics_rollout_metrics']['against_tagslam']

        position_results = sub_results['position_error']
        for toss_key, subsub_results in position_results.items():
            toss_num = int(toss_key.split('_')[1])
            over_traj = TrajectoryMetrics.position_error(
                self.tagslam_b_trajs[toss_num], self.predicted_trajs[toss_num])
            subsub_results['mean'] = over_traj.mean().item()
            subsub_results['traj'] = over_traj.tolist()

        rotation_error = sub_results['rotation_error']
        for toss_key, subsub_results in rotation_error.items():
            toss_num = int(toss_key.split('_')[1])
            over_traj = TrajectoryMetrics.rotation_error(
                self.tagslam_b_trajs[toss_num], self.predicted_trajs[toss_num])
            subsub_results['mean'] = over_traj.mean().item()
            subsub_results['traj'] = over_traj.tolist()

        add_error = sub_results['add_error']
        for toss_key, subsub_results in add_error.items():
            toss_num = int(toss_key.split('_')[1])
            over_traj = TrajectoryMetrics.add_error(
                self.tagslam_b_trajs[toss_num], self.predicted_trajs[toss_num],
                traj_evaluator._get_aligned_true_cloud()
            )
            subsub_results['mean'] = over_traj.mean().item()
            subsub_results['traj'] = over_traj.tolist()

        adds_error = sub_results['adds_error']
        for toss_key, subsub_results in adds_error.items():
            toss_num = int(toss_key.split('_')[1])
            over_traj = TrajectoryMetrics.adds_error(
                self.tagslam_b_trajs[toss_num], self.predicted_trajs[toss_num],
                traj_evaluator._get_aligned_true_cloud()
            )
            subsub_results['mean'] = over_traj.mean().item()
            subsub_results['traj'] = over_traj.tolist()

        ### Single-step metrics.
        sub_results = results['dynamics_single_step_metrics']['against_tagslam']

        position_results = sub_results['position_error']
        for toss_key, subsub_results in position_results.items():
            toss_num = int(toss_key.split('_')[1])
            over_traj = TrajectoryMetrics.position_error(
                self.single_step_tagslam_targets[toss_num],
                self.single_step_tagslam_predictions[toss_num])
            subsub_results['mean'] = over_traj.mean().item()
            subsub_results['traj'] = over_traj.tolist()

        rotation_error = sub_results['rotation_error']
        for toss_key, subsub_results in rotation_error.items():
            toss_num = int(toss_key.split('_')[1])
            over_traj = TrajectoryMetrics.rotation_error(
                self.single_step_tagslam_targets[toss_num],
                self.single_step_tagslam_predictions[toss_num])
            subsub_results['mean'] = over_traj.mean().item()
            subsub_results['traj'] = over_traj.tolist()

        add_error = sub_results['add_error']
        for toss_key, subsub_results in add_error.items():
            toss_num = int(toss_key.split('_')[1])
            over_traj = TrajectoryMetrics.add_error(
                self.single_step_tagslam_targets[toss_num],
                self.single_step_tagslam_predictions[toss_num],
                traj_evaluator._get_aligned_true_cloud()
            )
            subsub_results['mean'] = over_traj.mean().item()
            subsub_results['traj'] = over_traj.tolist()

        adds_error = sub_results['adds_error']
        for toss_key, subsub_results in adds_error.items():
            toss_num = int(toss_key.split('_')[1])
            over_traj = TrajectoryMetrics.adds_error(
                self.single_step_tagslam_targets[toss_num],
                self.single_step_tagslam_predictions[toss_num],
                traj_evaluator._get_aligned_true_cloud()
            )
            subsub_results['mean'] = over_traj.mean().item()
            subsub_results['traj'] = over_traj.tolist()


class GeometryEvaluator:
    """Evaluate the learned geometry.  This requires the following to already be
    present in the evaluation directory:
        - true_geom_aligned.obj
        - bsdf_mesh.obj if last run was BundleSDF, else pll_mesh.obj
    """
    def __init__(self, vision_asset: str, history: dict,
                 nerf_bundlesdf_id: str):
        # First decode the latest BundleSDF or PLL run IDs to set up the
        # evaluation directory.
        last_bsdf_iteration = 1
        for cycle in history.keys():
            cycle_num = cycle.split('_')[-1]
            if int(cycle_num) > last_bsdf_iteration:
                last_bsdf_iteration = int(cycle_num)
        last_tracking_bsdf_id = history[
            f'cycle_iteration_{last_bsdf_iteration}']['bundlesdf_id']

        pll_id = history[f'cycle_iteration_{last_bsdf_iteration}']['pll_id']
        if pll_id is not None:
            last_tracking_bsdf_id = None
            nerf_bundlesdf_id = None

        self.eval_dir = file_utils.evaluation_subdir(
            dataset=vision_asset, cycle_iteration=last_bsdf_iteration,
            tracking_bundlesdf_id=last_tracking_bsdf_id,
            nerf_bundlesdf_id=nerf_bundlesdf_id, pll_id=pll_id
        )

        self.pll_id = pll_id

        # Get the meshes.
        self._get_meshes()

    def _get_meshes(self):
        """Loads the learned and ground truth meshes from the evaluation
        directory."""
        # Learned mesh.
        if self.pll_id is None:
            learned_mesh_path = op.join(self.eval_dir, 'bsdf_mesh.obj')
        else:
            learned_mesh_path = op.join(
                self.eval_dir, 'pll_urdf', 'test.obj')
            if not op.exists(learned_mesh_path):
                learned_mesh_path = op.join(
                self.eval_dir, 'pll_urdf', 'test_best.obj')
        assert op.exists(learned_mesh_path), f'GeometryEvaluator requires ' + \
            f'{learned_mesh_path=} to exist, but does not exist.'
        self.learned_mesh = icp.load_mesh_from_obj(learned_mesh_path)

        # Ground truth mesh.
        true_mesh_path = op.join(self.eval_dir, 'true_geom_aligned.obj')
        assert op.exists(true_mesh_path), f'GeometryEvaluator requires ' + \
            f'{true_mesh_path=} to exist, but does not exist.'
        self.true_mesh = icp.load_mesh_from_obj(true_mesh_path)

        # Get each of their convex hulls too.
        self.learned_hull, _ = self.learned_mesh.compute_convex_hull()
        self.true_hull, _ = self.true_mesh.compute_convex_hull()

    def compute_metrics(self):
        self._compute_chamfer_distance()
        self._compute_f_score()
        self._compute_volume_error()

    def _compute_chamfer_distance(self):
        # Sample point clouds on both meshes.
        true_cloud = Tensor(np.asarray(
            self.true_mesh.sample_points_poisson_disk(2000).points))
        learned_cloud = Tensor(np.asarray(
            self.learned_mesh.sample_points_poisson_disk(2000).points))

        # Compute chamfer distance on these clouds.
        self.chamfer_distance = eval_utils.chamfer_distance(
            true_cloud, learned_cloud).item()

        # Do the same thing for the convex hull.
        true_hull_cloud = Tensor(np.asarray(
            self.true_hull.sample_points_poisson_disk(2000).points))
        learned_hull_cloud = Tensor(np.asarray(
            self.learned_hull.sample_points_poisson_disk(2000).points))
        self.hull_chamfer_distance = eval_utils.chamfer_distance(
            true_hull_cloud, learned_hull_cloud).item()

    # TODO BIBIT implement F-score
    def _compute_f_score(self):
        self.f_score = None
        self.hull_f_score = None

    def _compute_volume_error(self):
        # Get the vertices of each mesh.
        true_hull_vertices = Tensor(np.asarray(self.true_hull.vertices))
        learned_hull_vertices = Tensor(np.asarray(self.learned_hull.vertices))

        # Compute convex volume error on these vertices.
        self.convex_volume_error = eval_utils.convex_volume_error(
            true_hull_vertices, learned_hull_vertices).item()

    def store_geometry_metrics(self, results: dict):
        """Store the geometry metrics in the results dictionary."""
        full_geometry_results = results['geometry_metrics']['full_geometry']
        full_geometry_results['chamfer_distance'] = self.chamfer_distance
        full_geometry_results['f_score'] = self.f_score

        hull_geometry_results = results['geometry_metrics']['convex_hull']
        hull_geometry_results['chamfer_distance'] = self.hull_chamfer_distance
        hull_geometry_results['f_score'] = self.hull_f_score
        hull_geometry_results['volume_error'] = self.convex_volume_error


class TrajectoryMetrics:
    """Utility class for trajectory metrics."""

    @staticmethod
    def position_error(target_traj: Tensor, pred_traj: Tensor) -> Tensor:
        """Returns the positional error at each point over the trajectory."""
        target_traj = Tensor(target_traj)
        pred_traj = Tensor(pred_traj)

        target_xyz = target_traj[:, 4:7]
        pred_xyz = pred_traj[:, 4:7]

        pos_diff = target_xyz - pred_xyz
        pos_errors = torch.linalg.norm(pos_diff, dim=1)
        return pos_errors

    @staticmethod
    def rotation_error(target_traj: Tensor, pred_traj: Tensor) -> Tensor:
        """Returns the angular error at each point over the trajectory."""
        target_traj = Tensor(target_traj)
        pred_traj = Tensor(pred_traj)

        target_quat = target_traj[:, :4]
        pred_quat = pred_traj[:, :4]

        quat_errors = math_utils.quaternion_errors(target_quat, pred_quat)
        return quat_errors

    @staticmethod
    def add_error(target_traj: Tensor, pred_traj: Tensor, point_cloud: Tensor
                  ) -> Tensor:
        """Returns the average distance of 1-to-1 mapped object surface points
        from one pose to another pose, at each point in the trajectories.  These
        surface points are sampled from the true geometry."""
        target_traj = Tensor(target_traj)
        pred_traj = Tensor(pred_traj)
        point_cloud = Tensor(point_cloud)

        n_steps = target_traj.shape[0]

        # Convert the poses to homogeneous transformations.
        target_trans_mats = math_utils.batch_pll_format_to_trans_mat(
            target_traj)
        pred_trans_mats = math_utils.batch_pll_format_to_trans_mat(pred_traj)

        # Get the ADD error for every timestep.
        add_errors = torch.zeros((n_steps))

        for i, targ_pred in enumerate(zip(target_trans_mats, pred_trans_mats)):
            targ, pred = targ_pred[0], targ_pred[1]

            # Use the points that were already sampled on the aligned true
            # geometry.
            add_errors[i] = eval_utils.compute_add_tracking_error(
                targ, pred, point_cloud)

        return add_errors

    @staticmethod
    def adds_error(target_traj: Tensor, pred_traj: Tensor, point_cloud: Tensor
                   ) -> Tensor:
        """Returns the average distance from each point sampled on the true
        geometry at a predicted pose to the nearest point at a true pose, at
        each point in the trajectories."""
        target_traj = Tensor(target_traj)
        pred_traj = Tensor(pred_traj)
        point_cloud = Tensor(point_cloud)

        n_steps = target_traj.shape[0]

        # Convert the poses to homogeneous transformations.
        target_trans_mats = math_utils.batch_pll_format_to_trans_mat(
            target_traj)
        pred_trans_mats = math_utils.batch_pll_format_to_trans_mat(pred_traj)

        # Get the ADD error for every timestep.
        adds_errors = torch.zeros((n_steps))

        for i, targ_pred in enumerate(zip(target_trans_mats, pred_trans_mats)):
            targ, pred = targ_pred[0], targ_pred[1]

            # Use the points that were already sampled on the aligned true
            # geometry.
            adds_errors[i] = eval_utils.compute_adds_tracking_error(
                targ, pred, point_cloud)

        return adds_errors

    @staticmethod
    def penetration(traj: Tensor, geometry_system: MultibodyLearnableSystem
                    ) -> Tensor:
        """Returns the penetration of the geometry in the provided
        geometry_system when swept over the provided trajectory."""
        traj = Tensor(traj)

        assert traj.shape[1] == geometry_system.space.n_x

        phi, _J, _p_BiBc_B = geometry_system.multibody_terms.contact_terms(traj)
        phi = phi.detach().clone()
        smallest_phis = phi.min(dim=1).values
        return -torch.clamp_max(smallest_phis, 0)



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
@click.option('--pll-id',
              type=str,
              default=None,
              help="what PLL run ID to look up -- only include if want to " + \
                "evaluate PLL-only baseline.")
@click.option('--cycle-iteration',
              type=int,
              default=1,
              help="BundleSDF iteration number (can't choose 0 since that " + \
                "means use TagSLAM poses).")
@click.option('--do-videos/--skip-videos',
              type=bool,
              default=False,
              help="whether to generate videos.")

def main_command(vision_asset: str, bundlesdf_id: str, nerf_bundlesdf_id: str,
                 pll_id: str, cycle_iteration: int, do_videos: bool):
    assert cycle_iteration > 0, f'Invalid {cycle_iteration=}.'
    assert '_' in vision_asset, f'Invalid {vision_asset=}.'

    if pll_id is None:
        assert bundlesdf_id is not None, f'Need {bundlesdf_id=} if not ' + \
            f'{pll_id=}.'

        # Decode the BundleSDF run ID.
        tracking_bundlesdf_id = bundlesdf_id
        if tracking_bundlesdf_id[:13] != 'bundlesdf_id_':
            tracking_bundlesdf_id = f'bundlesdf_id_{tracking_bundlesdf_id}'
        if nerf_bundlesdf_id is None:
            nerf_bundlesdf_id = tracking_bundlesdf_id
        elif nerf_bundlesdf_id[:13] != 'bundlesdf_id_':
            nerf_bundlesdf_id = f'bundlesdf_id_{nerf_bundlesdf_id}'

        # Obtain the run history.
        history = traverse_run_history_from_bsdf(
            vision_asset, tracking_bundlesdf_id, cycle_iteration)

    else:
        assert bundlesdf_id is None and nerf_bundlesdf_id is None, f'Can ' + \
            f'only have {pll_id=} if not {bundlesdf_id=} or ' + \
            f'{nerf_bundlesdf_id=}.'

        # Decode the PLL run ID.
        if pll_id[:7] != 'pll_id_':
            pll_id = f'pll_id_{pll_id}'

        # Obtain the run history.
        history = traverse_run_history_from_pll(
            vision_asset, pll_id, cycle_iteration)

    # Automatically detect if BundleSDF-only is necessary based on if the object
    # is a tagless one.
    bsdf_only = False
    object = vision_asset.split('_')[0]
    if object in file_utils.TAGLESS_OBJECTS:
        bsdf_only = True
        print(f'Automatically setting {bsdf_only=} for tagless {object=}.')
    else:
        print(f'Using TagSLAM and BundleSDF: {bsdf_only=}')


    print(f'Found run history:')
    for key, val in history.items():
        print(f'\t{key} : {val}')

    # Create an empty results dictionary to be stored as a yaml file.
    last_run_was_bsdf = True if pll_id is None else False
    results = eval_utils.create_empty_results_dict(
        vision_asset, cycle_iteration, last_run_was_bsdf=last_run_was_bsdf)
    results['_overview']['vision_asset'] = vision_asset
    results['_overview']['history'] = history
    results['_overview']['nerf_bundlesdf_id'] = nerf_bundlesdf_id

    ### Pose estimation.
    # Compute tracking metrics if last run was BundleSDF (so PLL ID is None).
    traj_evaluator = TrajectoryPerformanceEvaluator(
        vision_asset, history, nerf_bundlesdf_id, bsdf_only)
    traj_evaluator.get_tracking_trajectories()
    if pll_id is None:
        traj_evaluator.store_tracking_metrics(results)

    ### Dynamics predictions.
    # Compute dynamics metrics if last run was PLL or PLL was ever run.
    if pll_id is not None or cycle_iteration > 1:
        dynamics_predictor = DynamicsPredictor(
            vision_asset, history, nerf_bundlesdf_id, bsdf_only)
        dynamics_predictor.generate_rollout_trajectories()
        dynamics_predictor.generate_single_step_predictions()
        dynamics_predictor.save_predictions()
        dynamics_predictor.store_dynamics_metrics(results, traj_evaluator)
        if do_videos:
            dynamics_predictor.make_prediction_video()
        else:
            print(f'Skipping video generation for {vision_asset=}, ' + \
                f'{bundlesdf_id=}, {nerf_bundlesdf_id=}, {cycle_iteration=}.')

    ### Geometry evaluation.
    geometry_evaluator = GeometryEvaluator(
        vision_asset, history, nerf_bundlesdf_id)
    geometry_evaluator.compute_metrics()
    geometry_evaluator.store_geometry_metrics(results)

    ### Save the results.
    file_utils.save_results_to_yaml(results, geometry_evaluator.eval_dir)


if __name__ == "__main__":
    main_command()  # pylint: disable=no-value-for-parameter
