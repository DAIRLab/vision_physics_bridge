"""Debugging."""

import torch
from torch import Tensor
import matplotlib.pyplot as plt
import pdb
import os.path as op
import open3d as o3d
import numpy as np

import evaluate, eval_utils, file_utils


DEBUG_SINGLE_STEP = True
DEBUG_ROLLOUT = False
DEBUG_CONVEX_HULL = False
DO_CONFIG = False
DO_VELOCITY = False
DO_ACCELERATION = True

if DEBUG_ROLLOUT:
    for toss_i in range(1, 11):
        # Load the trajectories.
        p1l = torch.load(f'evaluation/bottle_1-3_02_02_2/predicted_toss_{toss_i}.pt')
        p1r = torch.load(f'evaluation/bottle_1-3_02_02_2_REMOTE/predicted_toss_{toss_i}.pt')
        t1l = torch.load(f'evaluation/bottle_1-3_02_02_2/tagslam_b_toss_{toss_i}.pt')
        t1r = torch.load(f'evaluation/bottle_1-3_02_02_2_REMOTE/tagslam_b_toss_{toss_i}.pt')
        pll_t1l = torch.load(f'../dair_pll/assets/vision_bottle/bottle_1-10/toss/tagslam_LOCAL/{toss_i}.pt')
        pll_t1r = torch.load(f'../dair_pll/assets/vision_bottle/bottle_1-10/toss/tagslam/{toss_i}.pt')

        if toss_i <= 3:
            b1l = torch.load(f'evaluation/bottle_1-3_02_02_2/bundlesdf_toss_{toss_i}.pt')
            b1r = torch.load(f'evaluation/bottle_1-3_02_02_2_REMOTE/bundlesdf_toss_{toss_i}.pt')
            pll_b1l = torch.load(f'../dair_pll/assets/vision_bottle/bottle_1-3/toss/bundlesdf_iteration_2/bundlesdf_id_02_LOCAL/{toss_i}.pt')
            pll_b1r = torch.load(f'../dair_pll/assets/vision_bottle/bottle_1-3/toss/bundlesdf_iteration_2/bundlesdf_id_02_REMOTE/{toss_i}.pt')
            pll_b1l_from_r = torch.load(f'../dair_pll/assets/vision_bottle/bottle_1-3/toss/bundlesdf_iteration_2/bundlesdf_id_02/{toss_i}.pt')

        # Compare.
        plt.ion()
        fig, ax = plt.subplots(2, 4, figsize=(15, 15), sharex='all', sharey='row')
        for i in range(4):
            ax[0, i].plot(t1l[:, i], linewidth=5, label='TagSLAM, local eval')
            ax[0, i].plot(t1r[:, i], label='TagSLAM, remote eval')
            ax[0, i].plot(pll_t1l[:, i], linewidth=5, label='TagSLAM, local PLL assets')
            ax[0, i].plot(pll_t1r[:, i], color='cyan', label='TagSLAM, remote PLL assets')
            ax[0, i].plot(p1l[:, i], linewidth=5, label='Prediction, local eval')
            ax[0, i].plot(p1r[:, i], label='Prediction, remote eval')

            if toss_i <= 3:
                ax[0, i].plot(b1l[:, i], linewidth=5, label='BundleSDF, local eval')
                ax[0, i].plot(b1r[:, i], label='BundleSDF, remote eval')
                ax[0, i].plot(pll_b1l[:, i], linewidth=5, label='BundleSDF, local PLL asset')
                ax[0, i].plot(pll_b1r[:, i], label='BundleSDF, remote PLL asset')
                ax[0, i].plot(pll_b1l_from_r[:, i], label='BundleSDF, revised local PLL asset')

            if i < 3:
                ax[1, i].plot(t1l[:, i+4], linewidth=5, label='TagSLAM, local eval')
                ax[1, i].plot(t1r[:, i+4], label='TagSLAM, remote eval')
                ax[1, i].plot(pll_t1l[:, i+4], linewidth=5, label='TagSLAM, local PLL assets')
                ax[1, i].plot(pll_t1r[:, i+4], color='cyan', label='TagSLAM, remote PLL assets')
                ax[1, i].plot(p1l[:, i+4], linewidth=5, label='Prediction, local eval')
                ax[1, i].plot(p1r[:, i+4], label='Prediction, remote eval')

                if toss_i <= 3:
                    ax[1, i].plot(b1l[:, i+4], linewidth=5, label='BundleSDF, local eval')
                    ax[1, i].plot(b1r[:, i+4], label='BundleSDF, remote eval')
                    ax[1, i].plot(pll_b1l[:, i+4], linewidth=5, label='BundleSDF, local PLL asset')
                    ax[1, i].plot(pll_b1r[:, i+4], label='BundleSDF, remote PLL asset')
                    ax[1, i].plot(pll_b1l_from_r[:, i+4], label='BundleSDF, revised local PLL asset')

        ax[1, 2].legend()
        ax[0, 0].set_title('qw')
        ax[0, 1].set_title('qx')
        ax[0, 2].set_title('qy')
        ax[0, 3].set_title('qz')
        ax[1, 0].set_title('x')
        ax[1, 1].set_title('y')
        ax[1, 2].set_title('z')
        fig.suptitle(f'Toss {toss_i}')
        plt.savefig(f'debugging/single_toss_{toss_i}.png')
        plt.close()
    pdb.set_trace()

if DEBUG_SINGLE_STEP:
    for toss_i in range(1, 11):
        # Load the trajectories.
        st = torch.load(f'evaluation/bottle_1-3_02_02_2/tagslam_b_toss_{toss_i}.pt')[:-1]
        tt = torch.load(f'evaluation/bottle_1-3_02_02_2/step_target_tagslam_toss_{toss_i}.pt')
        pt = torch.load(f'evaluation/bottle_1-3_02_02_2/step_prediction_tagslam_toss_{toss_i}.pt')

        if toss_i <= 3:
            sb = torch.load(f'evaluation/bottle_1-3_02_02_2/bundlesdf_toss_{toss_i}.pt')[:-1]
            tb = torch.load(f'evaluation/bottle_1-3_02_02_2/step_target_bsdf_toss_{toss_i}.pt')
            pb = torch.load(f'evaluation/bottle_1-3_02_02_2/step_prediction_bsdf_toss_{toss_i}.pt')

        # Compare.
        plt.ion()
        if DO_CONFIG:
            fig, ax = plt.subplots(2, 4, figsize=(25, 15), sharex='all', sharey='none')
            for i in range(4):
                ax[0, i].plot(st[:, i], linewidth=2, label='TagSLAM single-step start')
                ax[0, i].plot(tt[:, i], linewidth=2, label='TagSLAM single-step target')
                ax[0, i].plot(pt[:, i], label='TagSLAM-based single-step')

                if toss_i <= 3:
                    ax[0, i].plot(sb[:, i], linewidth=2, label='BundleSDF single-step start')
                    ax[0, i].plot(tb[:, i], linewidth=2, label='BundleSDF single-step target')
                    ax[0, i].plot(pb[:, i], label='BundleSDF-based single-step')

                if i < 3:
                    ax[1, i].plot(st[:, i+4], linewidth=2, label='TagSLAM single-step start')
                    ax[1, i].plot(tt[:, i+4], linewidth=2, label='TagSLAM single-step target')
                    ax[1, i].plot(pt[:, i+4], label='TagSLAM-based single-step')

                    if toss_i <= 3:
                        ax[1, i].plot(sb[:, i+4], linewidth=2, label='BundleSDF single-step start')
                        ax[1, i].plot(tb[:, i+4], linewidth=2, label='BundleSDF single-step target')
                        ax[1, i].plot(pb[:, i+4], label='BundleSDF-based single-step')

            ax[1, 2].legend()
            ax[0, 0].set_title('qw')
            ax[0, 1].set_title('qx')
            ax[0, 2].set_title('qy')
            ax[0, 3].set_title('qz')
            ax[1, 0].set_title('x')
            ax[1, 1].set_title('y')
            ax[1, 2].set_title('z')
            fig.suptitle(f'Toss {toss_i}')
            plt.savefig(f'debugging/single_toss_{toss_i}.png')
            plt.close()

        if DO_VELOCITY:
            # Look at velocities.
            fig, ax = plt.subplots(2, 3, figsize=(25, 15), sharex='all', sharey='none')
            for i in range(3):
                ax[0, i].plot(st[:, i+7], linewidth=2, label='TagSLAM single-step start')
                ax[0, i].plot(tt[:, i+7], linewidth=2, label='TagSLAM single-step target')
                ax[0, i].plot(pt[:, i+7], label='TagSLAM-based single-step')

                if toss_i <= 3:
                    ax[0, i].plot(sb[:, i+7], linewidth=2, label='BundleSDF single-step start')
                    ax[0, i].plot(tb[:, i+7], linewidth=2, label='BundleSDF single-step target')
                    ax[0, i].plot(pb[:, i+7], label='BundleSDF-based single-step')

                ax[1, i].plot(st[:, i+3+7], linewidth=2, label='TagSLAM single-step start')
                ax[1, i].plot(tt[:, i+3+7], linewidth=2, label='TagSLAM single-step target')
                ax[1, i].plot(pt[:, i+3+7], label='TagSLAM-based single-step')

                if toss_i <= 3:
                    ax[1, i].plot(sb[:, i+3+7], linewidth=2, label='BundleSDF single-step start')
                    ax[1, i].plot(tb[:, i+3+7], linewidth=2, label='BundleSDF single-step target')
                    ax[1, i].plot(pb[:, i+3+7], label='BundleSDF-based single-step')

            ax[1, 2].legend()
            ax[0, 0].set_title('wx')
            ax[0, 1].set_title('wy')
            ax[0, 2].set_title('wz')
            ax[1, 0].set_title('vx')
            ax[1, 1].set_title('vy')
            ax[1, 2].set_title('vz')
            fig.suptitle(f'Toss {toss_i}')
            plt.savefig(f'debugging/single_toss_{toss_i}_v.png')
            plt.close()

        if DO_ACCELERATION:
            # Look at accelerations.  First need to compute them.
            DT = 1.0/30.0
            actual_acc_t = (tt[:, 7:13] - st[:, 7:13])/DT
            predicted_acc_t = (pt[:, 7:13] - st[:, 7:13])/DT

            if toss_i <= 3:
                actual_acc_b = (tb[:, 7:13] - sb[:, 7:13])/DT
                predicted_acc_b = (pb[:, 7:13] - sb[:, 7:13])/DT

            fig, ax = plt.subplots(2, 3, figsize=(25, 15), sharex='all', sharey='none')
            for i in range(3):
                ax[0, i].plot(actual_acc_t[:, i], linewidth=2, label='TagSLAM actual')
                ax[0, i].plot(predicted_acc_t[:, i], label='TagSLAM-based predicted')

                if toss_i <= 3:
                    ax[0, i].plot(actual_acc_b[:, i], linewidth=2, label='BundleSDF actual')
                    ax[0, i].plot(predicted_acc_b[:, i], label='BundleSDF-based predicted')

                ax[1, i].plot(actual_acc_t[:, i+3], linewidth=2, label='TagSLAM actual')
                ax[1, i].plot(predicted_acc_t[:, i+3], label='TagSLAM-based predicted')

                if toss_i <= 3:
                    ax[1, i].plot(actual_acc_b[:, i+3], linewidth=2, label='BundleSDF actual')
                    ax[1, i].plot(predicted_acc_b[:, i+3], label='BundleSDF-based predicted')

            ax[1, 2].legend()
            ax[0, 0].set_title('alphax')
            ax[0, 1].set_title('alphay')
            ax[0, 2].set_title('alphaz')
            ax[1, 0].set_title('ax')
            ax[1, 1].set_title('ay')
            ax[1, 2].set_title('az')
            fig.suptitle(f'Toss {toss_i}')
            plt.savefig(f'debugging/single_toss_{toss_i}_a.png')
            plt.close()
        
        eval_dir = file_utils.evaluation_subdir(
            dataset='bottle_1-3', cycle_iteration=2,
            tracking_bundlesdf_id='bundlesdf_id_02',
            nerf_bundlesdf_id='bundlesdf_id_02', create=False)
        urdf_path = op.join(eval_dir, 'bsdf_mesh_pll_params.urdf')
        geometry_system = eval_utils.create_multibody_learnable_system(
            urdf_path=urdf_path)
        

        pdb.set_trace()

if DEBUG_CONVEX_HULL:
    vision_assets = ['cube_1-4', 'bottle_1-3']

    slide_little = np.eye(4)
    slide_little[0, 3] = 0.2
    slide_more = np.eye(4)
    slide_more[0, 3] = 0.4

    for vision_asset in vision_assets:
        # history = evaluate.traverse_run_history_from_bsdf(
        #     vision_asset, bundlesdf_id='bundlesdf_id_02', cycle_iteration=2)
        history = {
            'cycle_iteration_2': {'bundlesdf_id': 'bundlesdf_id_02', 'pll_id': None},
            'cycle_iteration_1': {'bundlesdf_id': 'bundlesdf_id_00', 'pll_id': 'pll_id_00'}}

        geom_eval = evaluate.GeometryEvaluator(
            vision_asset=vision_asset, history=history,
            nerf_bundlesdf_id='bundlesdf_id_02'
        )
        geom_eval._compute_chamfer_distance()
        print(f'\bVISION ASSET: {vision_asset}')
        print(f'Full geometry chamfer distance: {geom_eval.chamfer_distance:.4f}')
        print(f'Convex hull chamfer distance: {geom_eval.hull_chamfer_distance:.4f}')

        # Inspect the meshes.
        true_cloud = geom_eval.true_mesh.sample_points_poisson_disk(2000)
        learned_cloud = geom_eval.learned_mesh.sample_points_poisson_disk(2000)

        true_hull_cloud = geom_eval.true_hull.sample_points_poisson_disk(2000)
        learned_hull_cloud = geom_eval.learned_hull.sample_points_poisson_disk(2000)

        true_cloud_slid = o3d.geometry.PointCloud(true_cloud).transform(slide_little)
        learned_cloud_slid = o3d.geometry.PointCloud(learned_cloud).transform(slide_more)
        true_hull_cloud_slid = o3d.geometry.PointCloud(true_hull_cloud).transform(slide_little)
        learned_hull_cloud_slid = o3d.geometry.PointCloud(learned_hull_cloud).transform(slide_more)

        o3d.visualization.draw_geometries(
            [true_cloud, learned_cloud, true_cloud_slid, learned_cloud_slid],
            window_name='Full geometry:  overlaid-true-learned'
        )

        o3d.visualization.draw_geometries(
            [true_hull_cloud, learned_hull_cloud, true_hull_cloud_slid, learned_hull_cloud_slid],
            window_name='Full geometry:  overlaid-true-learned'
        )

        del history, geom_eval, true_cloud, learned_cloud, true_hull_cloud, \
            learned_hull_cloud, true_cloud_slid, learned_cloud_slid, \
            true_hull_cloud_slid, learned_hull_cloud_slid
