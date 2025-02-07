"""Visualization utilities to observe PLL's contribution to BundleSDF meshes."""

import os
import os.path as op
import pickle
import pdb
import numpy as np
import matplotlib.pyplot as plt
import torch
import trimesh
from tqdm import tqdm

from tempfile import TemporaryDirectory

import file_utils


# TODO
def visualize_two_pts_sdfs(pts1, sdf1, pts2, sdf2, max_n=5000, cval_geo1=True,
                           cval_geo2=True, video_output_file=None,
                           figure_output_file=None):
    '''
    If sdf is provided, use sdf to colorize, 
    else if cval_geo is True, use position to colorize, 
    otherwise use a constant color. 
    '''
    cval_sdf1 = True
    if isinstance(pts1, str):
        pts1 = torch.load(pts1)
    if isinstance(sdf1, str):
        sdf1 = torch.load(sdf1)
    elif sdf1 is None:
        sdf1 = torch.zeros_like(pts1[:,0])
        cval_sdf1 = False

    cval_sdf2 = True
    if isinstance(pts2, str):
        pts2 = torch.load(pts2)
    if isinstance(sdf2, str):
        sdf2 = torch.load(sdf2)
    elif sdf2 is None:
        sdf2 = torch.zeros_like(pts2[:,0])
        cval_sdf2 = False
    
    if pts2.ndim == 4:
        sample_id = 0 #150
        pts2 = pts2[sample_id].reshape(-1, 3)
        sdf2 = sdf2[sample_id].reshape(-1)

    print(f"Before subsample: {pts1.shape=}, {sdf1.shape=}, {pts2.shape=}, {sdf2.shape=}")
    if max_n > 0:
        if pts1.shape[0]>max_n:
            idxs = np.random.permutation(pts1.shape[0])
            pts1 = pts1[idxs[:max_n]]
            sdf1 = sdf1[idxs[:max_n]]
        if pts2.shape[0]>max_n:
            idxs = np.random.permutation(pts2.shape[0])
            pts2 = pts2[idxs[:max_n]]
            sdf2 = sdf2[idxs[:max_n]]
            
    print(f"After subsample: {pts1.shape=}, {sdf1.shape=}, {pts2.shape=}, {sdf2.shape=}")
    
    if cval_sdf1:
        cmap1 = 'coolwarm'
        cval1 = sdf1
    elif cval_geo1:
        cmap1 = 'viridis'
        # cval1 = pts1.sum(1)
        cval1 = pts1[:, 2]
    else:
        cmap1 = 'Greens'
        cval1 = sdf1
        
    if cval_sdf2:
        # cmap2 = 'twilight'
        cmap2 = 'coolwarm'
        cval2 = sdf2
    elif cval_geo2:
        cmap2 = 'plasma'
        # cval2 = pts2.sum(1)
        cval2 = pts2[:, 1]
    else:
        cmap2 = 'Greys'
        cval2 = sdf2

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    pts1 = pts1.cpu().detach().numpy()
    pts2 = pts2.cpu().detach().numpy()
    # ax.scatter(pts1[:,0], pts1[:,1], pts1[:,2], color='red', s=1)
    colored1 = ax.scatter(pts1[:, 0], pts1[:, 1], pts1[:, 2], c=cval1,
                              cmap=cmap1, marker='o', vmin=-1, vmax=1,
                              label='pts1', s=1)
    # ax.scatter(pts2[:, 0], pts2[:, 1], pts2[:, 2], c='green', 
    #            marker='.', s=1)
    colored2 = ax.scatter(pts2[:, 0], pts2[:, 1], pts2[:, 2], c=cval2, cmap=cmap2, vmin=-1, vmax=1,
               marker='x', label='pts2', s=1)
    # ax.scatter(rendered_pts[:, 0], rendered_pts[:, 1], rendered_pts[:, 2])
    # Because both scatter series are using the 'viridis' color map, the
    # colorbar will share a mapping for both series.
    if cval_sdf1:
        cbar = fig.colorbar(colored1)
        cbar.set_label('sdf1')
    elif cval_geo1:
        cbar = fig.colorbar(colored1)
        cbar.set_label('pts1')

    if cval_sdf2:
        cbar = fig.colorbar(colored2)
        cbar.set_label('sdf2')
    elif cval_geo2:
        cbar = fig.colorbar(colored2)
        cbar.set_label('pts2')

    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.legend()

    # Set equal aspect ratio.
    ax.set_box_aspect([np.ptp(arr) for arr in \
                       [ax.get_xlim(), ax.get_ylim(), ax.get_zlim()]])

    if video_output_file is not None:
        with TemporaryDirectory(prefix="sdf-slice-") as tmpdir:
            print(f'Storing temporary files at {tmpdir}')

            for i in tqdm(range(180)):
                ax.view_init(elev=30., azim=i*2, roll=0)
                fig.canvas.draw()
                fig.canvas.flush_events()
                plt.savefig(op.join(tmpdir, f'{i:07d}.png'))

            os.system(f'ffmpeg -y -r 30 -i {tmpdir}/%07d.png -vcodec ' + \
                      f'libx264 -preset slow -crf 18 {video_output_file}')
            print(f'Saved slice video to {video_output_file}.')
        
        if figure_output_file is not None:
            pickle.dump(fig, open(figure_output_file, 'wb'))
            print(f'Saved slice figure to {figure_output_file}.')
            print(f'''\nNote:  This interactive file can be opened via:
                  
import pickle
figx = pickle.load(open('{figure_output_file}', 'rb'))
figx.show()\n''')

    else:
        plt.show()


def visualize_three_pts_sdfs(pts1, sdf1, pts2, sdf2, pts3, sdf3, max_n=5000,
                             cval_geo1=True, cval_geo2=True, cval_geo3=True,
                             video_output_file=None, figure_output_file=None):
    '''
    If sdf is provided, use sdf to colorize, 
    else if cval_geo is True, use position to colorize, 
    otherwise use a constant color. 
    '''
    cval_sdf1 = True
    if isinstance(pts1, str):
        pts1 = torch.load(pts1)
    if isinstance(sdf1, str):
        sdf1 = torch.load(sdf1)
    elif sdf1 is None:
        sdf1 = torch.zeros_like(pts1[:,0])
        cval_sdf1 = False

    cval_sdf2 = True
    if isinstance(pts2, str):
        pts2 = torch.load(pts2)
    if isinstance(sdf2, str):
        sdf2 = torch.load(sdf2)
    elif sdf2 is None:
        sdf2 = torch.zeros_like(pts2[:,0])
        cval_sdf2 = False
    
    cval_sdf3 = True
    if isinstance(pts3, str):
        pts3 = torch.load(pts3)
    if isinstance(sdf3, str):
        sdf3 = torch.load(sdf3)
    elif sdf3 is None:
        sdf3 = torch.zeros_like(pts3[:,0])
        cval_sdf3 = False

    if isinstance(pts2, list):
        pts2 = torch.cat(pts2, 0)
        sdf2 = torch.cat(sdf2, 0)

    if isinstance(pts3, list):
        pts3 = torch.cat(pts3, 0)
        sdf3 = torch.cat(sdf3, 0)

    # sdf3_neg_mask = sdf3 < 0
    # sdf3 = sdf3[sdf3_neg_mask]
    # pts3 = pts3[sdf3_neg_mask]
    # sdf3 = torch.zeros_like(pts3[:,0])
    # cval_sdf3 = False

    if pts3.ndim == 4:
        sample_id = 32 # This is an arbitrary number sampling the 32nd contact point
        pts3 = pts3[sample_id].reshape(-1, 3)
        sdf3 = sdf3[sample_id].reshape(-1)

    print(f"Before subsample: {pts1.shape=}, {sdf1.shape=}, {pts2.shape=}, {sdf2.shape=}, {pts3.shape=}, {sdf3.shape=}")
    if max_n > 0:
        if pts1.shape[0]>max_n:
            idxs = np.random.permutation(pts1.shape[0])
            pts1 = pts1[idxs[:max_n]]
            sdf1 = sdf1[idxs[:max_n]]
        if pts2.shape[0]>max_n:
            idxs = np.random.permutation(pts2.shape[0])
            pts2 = pts2[idxs[:max_n]]
            sdf2 = sdf2[idxs[:max_n]]
        if pts3.shape[0]>max_n:
            idxs = np.random.permutation(pts3.shape[0])
            pts3 = pts3[idxs[:max_n]]
            sdf3 = sdf3[idxs[:max_n]]
            
    print(f"After subsample: {pts1.shape=}, {sdf1.shape=}, {pts2.shape=}, {sdf2.shape=}, {pts3.shape=}, {sdf3.shape=}")
    
    if cval_sdf1:
        cmap1 = 'coolwarm'
        cval1 = sdf1
    elif cval_geo1:
        # cmap1 = 'viridis'
        cmap1 = 'Greys'
        # cval1 = pts1.sum(1)
        cval1 = pts1[:, 2]
    else:
        cmap1 = 'Greens'
        cval1 = sdf1
        
    if cval_sdf2:
        cmap2 = 'twilight'
        # cmap2 = 'coolwarm'
        cval2 = sdf2
    elif cval_geo2:
        cmap2 = 'plasma'
        # cval2 = pts2.sum(1)
        cval2 = pts2[:, 1]
    else:
        cmap2 = 'Greys'
        cval2 = sdf2

    if cval_sdf3:
        # cmap3 = 'hsv'
        # cmap3 = 'twilight'
        cmap3 = 'coolwarm'
        cval3 = sdf3
        print("using sdf3")
    elif cval_geo3:
        # cmap3 = 'plasma'
        cmap3 = 'viridis'
        # cval2 = pts2.sum(1)
        cval3 = pts3[:, 0]
        print("using y")
    else:
        cmap3 = 'Oranges'
        cval3 = sdf3
        print("using zero")

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    pts1 = pts1.cpu().detach().numpy()
    pts2 = pts2.cpu().detach().numpy()
    pts3 = pts3.cpu().detach().numpy()
    # ax.scatter(pts1[:,0], pts1[:,1], pts1[:,2], color='red', s=1)
    colored1 = ax.scatter(pts1[:, 0], pts1[:, 1], pts1[:, 2], c=cval1,
                              cmap=cmap1, marker='o', vmin=-1, vmax=1,
                              label='pts1', s=1)
    # ax.scatter(pts2[:, 0], pts2[:, 1], pts2[:, 2], c='green', 
    #            marker='.', s=1)
    colored2 = ax.scatter(pts2[:, 0], pts2[:, 1], pts2[:, 2], c=cval2, cmap=cmap2, vmin=-1, vmax=1,
               marker='*', label='pts2', s=2)
    colored3 = ax.scatter(pts3[:, 0], pts3[:, 1], pts3[:, 2], c=cval3, cmap=cmap3, vmin=-1, vmax=1,
                marker='x', label='pts3', s=1)
    # ax.scatter(rendered_pts[:, 0], rendered_pts[:, 1], rendered_pts[:, 2])
    # Because both scatter series are using the 'viridis' color map, the
    # colorbar will share a mapping for both series.
    if cval_sdf1:
        cbar = fig.colorbar(colored1)
        cbar.set_label('sdf1')
    elif cval_geo1:
        cbar = fig.colorbar(colored1)
        cbar.set_label('pts1')

    if cval_sdf2:
        cbar = fig.colorbar(colored2)
        cbar.set_label('sdf2')
    elif cval_geo2:
        cbar = fig.colorbar(colored2)
        cbar.set_label('pts2')

    if cval_sdf3:
        cbar = fig.colorbar(colored3)
        cbar.set_label('sdf3')
    elif cval_geo3:
        cbar = fig.colorbar(colored3)
        cbar.set_label('pts3')

    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.legend()

    # Set equal aspect ratio.
    ax.set_box_aspect([np.ptp(arr) for arr in \
                       [ax.get_xlim(), ax.get_ylim(), ax.get_zlim()]])

    if video_output_file is not None:
        with TemporaryDirectory(prefix="sdf-slice-") as tmpdir:
            print(f'Storing temporary files at {tmpdir}')

            for i in tqdm(range(180)):
                ax.view_init(elev=30., azim=i*2, roll=0)
                fig.canvas.draw()
                fig.canvas.flush_events()
                plt.savefig(op.join(tmpdir, f'{i:07d}.png'))

            os.system(f'ffmpeg -y -r 30 -i {tmpdir}/%07d.png -vcodec ' + \
                      f'libx264 -preset slow -crf 18 {video_output_file}')
            print(f'Saved slice video to {video_output_file}.')
        
        if figure_output_file is not None:
            pickle.dump(fig, open(figure_output_file, 'wb'))
            print(f'Saved slice figure to {figure_output_file}.')
            print(f'''\nNote:  This interactive file can be opened via:
                  
import pickle
figx = pickle.load(open('{figure_output_file}', 'rb'))
figx.show()\n''')

    else:
        plt.show()


class SDFSliceViewer:
    """Generate visuals to inspect a slice of the trained SDF, including
    plotting other relevant points like points sampled on the generated mesh and
    contact points given during SDF training."""
    def __init__(self, vision_asset: str, tracking_bundlesdf_id: str,
                 nerf_bundlesdf_id: str, cycle_iteration: int, remote: bool = False):
        # First decode the system and start/end tosses from the provided asset
        # directory.
        assert cycle_iteration > 0, f'Invalid {cycle_iteration=}.'
        assert '_' in vision_asset, f'Invalid {vision_asset=}.'

        start_toss = int(vision_asset.split('_')[-1].split('-')[0])
        end_toss = start_toss if '-' not in vision_asset else \
            int(vision_asset.split('-')[1])
        assert start_toss <= end_toss, f'Invalid toss range: {start_toss} ' + \
                f'-{end_toss} inferred from {vision_asset=}.'

        # Decode the BundleSDF run ID.
        if tracking_bundlesdf_id[:13] != 'bundlesdf_id_':
            tracking_bundlesdf_id = f'bundlesdf_id_{tracking_bundlesdf_id}'
        if nerf_bundlesdf_id is None:
            nerf_bundlesdf_id = tracking_bundlesdf_id
        elif nerf_bundlesdf_id[:13] != 'bundlesdf_id_':
            nerf_bundlesdf_id = f'bundlesdf_id_{nerf_bundlesdf_id}'
        
        self.vision_asset = vision_asset
        self.tracking_bundlesdf_id = tracking_bundlesdf_id
        self.nerf_bundlesdf_id = nerf_bundlesdf_id
        self.cycle_iteration = cycle_iteration

        self.remote = remote

    def visualization(self):
        # Get output filenames for the raw figure and video.
        video_output_file = file_utils.inspection_3d_slice_video_filepath(
            dataset=self.vision_asset,
            tracking_bundlesdf_id=self.tracking_bundlesdf_id,
            nerf_bundlesdf_id=self.nerf_bundlesdf_id,
            cycle_iteration=self.cycle_iteration
        )
        figure_output_file = file_utils.inspection_3d_slice_figure_filepath(
            dataset=self.vision_asset,
            tracking_bundlesdf_id=self.tracking_bundlesdf_id,
            nerf_bundlesdf_id=self.nerf_bundlesdf_id,
            cycle_iteration=self.cycle_iteration
        )

        # Load the mesh of the normalized SDF-space geometry.
        nerf_results_dir = file_utils.bundlesdf_nerf_results_dir(
            dataset=self.vision_asset,
            tracking_bundlesdf_id=self.tracking_bundlesdf_id,
            nerf_bundlesdf_id=self.nerf_bundlesdf_id,
            cycle_iteration=self.cycle_iteration
        )
        obj_file = op.join(nerf_results_dir, 'mesh_cleaned.obj')

        # Sample points on the mesh.
        mesh_cleaned = trimesh.load(obj_file, force='mesh')
        mesh_cleaned_pts = mesh_cleaned.sample(3000)
        mesh_cleaned_pts = torch.tensor(mesh_cleaned_pts)

        # Load the contact points.
        sdf_dir = op.join(nerf_results_dir, 'sdf_inspection')

        cps_slices_filepath = op.join(sdf_dir, 'cps_slices.pt')
        cps_slices_pred_filepath = op.join(sdf_dir, 'cps_slices_predsdf.pt')

        ### Need x server to run. Either run locally or run remotely with x
        # forward configured.
        change_display = False
        if self.remote and os.environ.get('DISPLAY') != ':99':
            # Run Xvfb to create a virtual display.
            print("Running Xvfb (virtual display) for rendering.")
            import subprocess
            self.xvfb_process = subprocess.Popen(['Xvfb', ':99', '-screen', '0',
                                                  '640x480x24'])
            print(f"Changing display environment from " + \
                  f"{os.environ['DISPLAY']} variable to :99")
            self.old_display = os.environ['DISPLAY']
            os.environ['DISPLAY'] = ':99'
            change_display = True

        cps_near_pcd_filepath = op.join(sdf_dir, 'cps_near_pcd.pt')
        cps_near_sdf_filepath = op.join(sdf_dir, 'cps_near_sdf_pred.pt')

        if op.exists(cps_near_pcd_filepath) and op.exists(cps_near_sdf_filepath):
            visualize_three_pts_sdfs(
                mesh_cleaned_pts, None, cps_near_pcd_filepath,
                cps_near_sdf_filepath, cps_slices_filepath,
                cps_slices_pred_filepath, video_output_file=video_output_file,
                figure_output_file=figure_output_file)
        else:
            visualize_two_pts_sdfs(
                mesh_cleaned_pts, None, cps_slices_filepath,
                cps_slices_pred_filepath, video_output_file=video_output_file,
                figure_output_file=figure_output_file)

        if self.remote and change_display:
            # Restore the original display environment.
            print(f"Restoring display environment to {self.old_display}")
            os.environ['DISPLAY'] = self.old_display
            self.xvfb_process.kill()
            print("Killed Xvfb process.")