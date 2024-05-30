"""Utilities to help with processing ground truth meshes."""

import click
import numpy as np
import open3d as o3d
import os
import os.path as op
import pdb
import trimesh

import file_utils, icp


# ICP hyperparameters
ICP_THRESHOLD = 0.02    # Maximum distance threshold for a point to be
                        # considered during alignment.


class MeshProcessor:
    """Mesh processing to align ground truth and learned meshes.  For shapes
    that are already relatively accurate, ICP works very well.
    
    TODO:  Needs to be tested on partial/bad mesh estimates, shapes from PLL
    only.
    """
    def __init__(self, vision_asset: str, tracking_bundlesdf_id: str,
                 nerf_bundlesdf_id: str, cycle_iteration: int):
        # First decode the system and start/end tosses from the provided asset
        # directory.
        assert cycle_iteration > 0, f'Invalid {cycle_iteration=}.'
        assert '_' in vision_asset, f'Invalid {vision_asset=}.'
        self.object = vision_asset.split('_')[0]

        self.vision_asset = vision_asset
        self.tracking_bundlesdf_id = tracking_bundlesdf_id
        self.nerf_bundlesdf_id = nerf_bundlesdf_id
        self.cycle_iteration = cycle_iteration

        # Get the meshes.
        self._load_meshes()
        self.did_alignment_to_bundlesdf = False
        
    def _load_meshes(self):
        true_obj_file = file_utils.object_scan_filepath(self.object)
        self.true_mesh = icp.load_mesh_from_obj(true_obj_file)

        nerf_results_dir = file_utils.bundlesdf_nerf_results_dir(
            dataset=self.vision_asset, cycle_iteration=self.cycle_iteration,
            tracking_bundlesdf_id=self.tracking_bundlesdf_id,
            nerf_bundlesdf_id=self.nerf_bundlesdf_id
        )
        learned_obj_file = op.join(nerf_results_dir, 'textured_mesh.obj')
        self.learned_mesh = icp.load_mesh_from_obj(learned_obj_file)

    def align_true_to_learned_mesh_with_icp(
            self, show: bool = True, save_dir: str = None,
            obj_name: str = 'true_geom_aligned.obj'):
        """Goal is to create a ground truth mesh that shares the same body
        origin as the learned mesh.  To do this, use ICP to register the true
        mesh to the learned mesh."""
        # Need to do ICP on 3D points instead of the mesh directly.
        true_cloud = self.true_mesh.sample_points_poisson_disk(2000)
        learned_cloud = self.learned_mesh.sample_points_poisson_disk(2000)

        # Visualize the initial alignment.
        if show:
            o3d.visualization.draw_geometries(
                [true_cloud, learned_cloud], window_name="Before ICP")
        if save_dir is not None:
            vis = o3d.visualization.Visualizer()
            vis.create_window(visible=False)
            vis.add_geometry(true_cloud)
            vis.add_geometry(learned_cloud)
            vis.poll_events()
            vis.update_renderer()
            image = vis.capture_screen_float_buffer(do_render=True)
            o3d.io.write_image(
                op.join(save_dir, 'alignment_before_icp.png'),
                o3d.geometry.Image((255 * np.asarray(image)).astype(np.uint8))
            )
            vis.destroy_window()

        # Coarse initial alignment to get the true mesh's cloud centroid to the
        # same location as the learned mesh's cloud centroid.
        initial_transformation = np.eye(4)
        initial_transformation[:3, 3] = learned_cloud.get_center() - \
            true_cloud.get_center()

        # Apply ICP.
        reg_p2p = o3d.pipelines.registration.registration_icp(
            true_cloud, learned_cloud, ICP_THRESHOLD, initial_transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000)
        )

        # Save the transformation matrix.
        print(f'Solved transformation matrix:\n{reg_p2p.transformation}')
        self.true_to_learned_transform = reg_p2p.transformation
        if save_dir is not None:
            np.savetxt(
                op.join(save_dir, 'true_to_learned_tf.txt'),
                reg_p2p.transformation
            )

        # Transform the true mesh's point cloud by the solved transform and view
        # the results.
        true_cloud.transform(reg_p2p.transformation)
        self.aligned_true_cloud = true_cloud
        if show:
            o3d.visualization.draw_geometries(
                [true_cloud, learned_cloud], window_name="After ICP")
        if save_dir is not None:
            vis = o3d.visualization.Visualizer()
            vis.create_window(visible=False)
            vis.add_geometry(true_cloud)
            vis.add_geometry(learned_cloud)
            vis.poll_events()
            vis.update_renderer()
            image = vis.capture_screen_float_buffer(do_render=True)
            o3d.io.write_image(
                op.join(save_dir, 'alignment_after_icp.png'),
                o3d.geometry.Image((255 * np.asarray(image)).astype(np.uint8))
            )
            vis.destroy_window()

        # Save the transformed mesh to file.
        self.true_mesh.transform(self.true_to_learned_transform)
        self.did_alignment_to_bundlesdf = True
        if save_dir is not None:
            o3d.io.write_triangle_mesh(
                op.join(save_dir, obj_name), self.true_mesh,
                write_triangle_uvs=False
            )
            print(f'Saved ground truth mesh transformed to align with ' + \
                f'BundleSDF mesh, as {obj_name} in {save_dir}.')

            material_filepath = op.join(
                save_dir, f'{obj_name.split(".")[0]}.mtl'
            )
            if op.exists(material_filepath):
                os.system(f'rm {material_filepath}')


class UnscaledMeshProcessor(MeshProcessor):
    """Do the same thing as MeshProcessor but from filepaths."""
    def __init__(self, object: str):
        self.object = object
        self._load_meshes()

    def _load_meshes(self):
        bundlesdf_obj_file = op.join(
            file_utils.object_scan_dir(), 'from_bundlesdf',
            f'{self.object}.obj'
        )
        self.bundlesdf_mesh = icp.load_mesh_from_obj(bundlesdf_obj_file)

        wrong_scaling_obj_file = op.join(
            file_utils.object_scan_dir(), 'wrong_scaling', f'{self.object}.obj')
        self.wrong_scaling_mesh = icp.load_mesh_from_obj(wrong_scaling_obj_file)

    def scale_and_align_wrong_to_bundlesdf(self, show=False):
        def compute_scale_factor(mesh1, mesh2):
            bbox1 = mesh1.get_axis_aligned_bounding_box()
            bbox2 = mesh2.get_axis_aligned_bounding_box()

            diagonal1 = np.linalg.norm(bbox1.get_max_bound() - \
                                       bbox1.get_min_bound())
            diagonal2 = np.linalg.norm(bbox2.get_max_bound() - \
                                       bbox2.get_min_bound())

            scale_factor = diagonal1 / diagonal2
            return scale_factor

        inspection_dir = op.join(
            file_utils.object_scan_dir(), 'corrected_scaling')

        last_inlier_rmse = 2.0
        inlier_rmse = 1.0
        round = 1
        while last_inlier_rmse - inlier_rmse > 1e-4:
            print(f'Round {round} Inlier RMSE: {inlier_rmse} (improvement ' + \
                  f'by {last_inlier_rmse-inlier_rmse})')

            # Show initial scaling/alignment.
            if show:
                bsdf_cloud = self.bundlesdf_mesh.sample_points_poisson_disk(2000)
                wrong_cloud = self.wrong_scaling_mesh.sample_points_poisson_disk(2000)
                o3d.visualization.draw_geometries(
                    [bsdf_cloud, wrong_cloud],
                    window_name=f'Before Scaling/ICP, {round=}'
                )

            # Save the before image.
            if round==1:
                bsdf_cloud = self.bundlesdf_mesh.sample_points_poisson_disk(2000)
                wrong_cloud = self.wrong_scaling_mesh.sample_points_poisson_disk(2000)
                vis = o3d.visualization.Visualizer()
                vis.create_window(visible=False)
                vis.add_geometry(wrong_cloud)
                vis.add_geometry(bsdf_cloud)
                vis.poll_events()
                vis.update_renderer()
                image = vis.capture_screen_float_buffer(do_render=True)
                o3d.io.write_image(
                    op.join(inspection_dir, f'{self.object}_before.png'),
                    o3d.geometry.Image((255 * np.asarray(image)).astype(np.uint8))
                )
                vis.destroy_window()

            # First deal with the fact that the mesh is improperly scaled.
            scale_factor = compute_scale_factor(
                self.bundlesdf_mesh, self.wrong_scaling_mesh)
            self.wrong_scaling_mesh.scale(
                scale_factor, center=self.wrong_scaling_mesh.get_center())

            # Need to do ICP on 3D points instead of the mesh directly.
            bsdf_cloud = self.bundlesdf_mesh.sample_points_poisson_disk(2000)
            wrong_cloud = self.wrong_scaling_mesh.sample_points_poisson_disk(2000)

            # Show initial alignment.
            if show:
                o3d.visualization.draw_geometries(
                    [bsdf_cloud, wrong_cloud],
                    window_name=f'Before ICP, {round=}'
                )

            # Coarse initial alignment to get the true mesh's cloud centroid to the
            # same location as the learned mesh's cloud centroid.
            initial_transformation = np.eye(4)
            initial_transformation[:3, 3] = bsdf_cloud.get_center() - \
                wrong_cloud.get_center()

            # Apply ICP.
            reg_p2p = o3d.pipelines.registration.registration_icp(
                wrong_cloud, bsdf_cloud, ICP_THRESHOLD, initial_transformation,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000)
            )

            # Save the transformation matrix.
            self.wrong_to_bundlesdf_transform = reg_p2p.transformation

            # Transform the wrong mesh's point cloud by the solved transform and
            # view the results.
            wrong_cloud.transform(reg_p2p.transformation)
            if show:
                o3d.visualization.draw_geometries(
                    [wrong_cloud, bsdf_cloud],
                    window_name=f'After ICP, {round=}'
                )

            # Transform the mesh.
            self.wrong_scaling_mesh.transform(self.wrong_to_bundlesdf_transform)

            round += 1
            last_inlier_rmse = inlier_rmse
            inlier_rmse = reg_p2p.inlier_rmse

        # Save the after image.
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False)
        vis.add_geometry(wrong_cloud)
        vis.add_geometry(bsdf_cloud)
        vis.poll_events()
        vis.update_renderer()
        image = vis.capture_screen_float_buffer(do_render=True)
        o3d.io.write_image(
            op.join(inspection_dir, f'{self.object}_after.png'),
            o3d.geometry.Image((255 * np.asarray(image)).astype(np.uint8))
        )
        vis.destroy_window()

        print(f'Finished with {round-1} rounds with inlier RMSE ' + \
              f'{inlier_rmse} (improvement by {last_inlier_rmse-inlier_rmse}).')

        # Save the result.
        corrected_filepath = op.join(
            file_utils.object_scan_dir(), 'corrected_scaling',
            f'{self.object}.obj'
        )
        o3d.io.write_triangle_mesh(corrected_filepath, self.wrong_scaling_mesh)
        print(f'Saved ground truth mesh transformed to align with ' + \
              f'BundleSDF mesh, as {self.object}.obj in corrected_scaling.')
        material_filepath = op.join(
            file_utils.object_scan_dir(), 'corrected_scaling',
            f'{self.object}.mtl'
        )
        if op.exists(material_filepath):
            os.system(f'rm {material_filepath}')


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

def main_command(vision_asset: str, bundlesdf_id: str, nerf_bundlesdf_id: str,
                 cycle_iteration: int):
    # Decode the BundleSDF run ID.
    tracking_bundlesdf_id = bundlesdf_id
    if tracking_bundlesdf_id[:13] != 'bundlesdf_id_':
        tracking_bundlesdf_id = f'bundlesdf_id_{tracking_bundlesdf_id}'
    if nerf_bundlesdf_id is None:
        nerf_bundlesdf_id = tracking_bundlesdf_id
    elif nerf_bundlesdf_id[:13] != 'bundlesdf_id_':
        nerf_bundlesdf_id = f'bundlesdf_id_{nerf_bundlesdf_id}'

    mesh_processor = MeshProcessor(
        vision_asset, tracking_bundlesdf_id, nerf_bundlesdf_id, cycle_iteration)

    mesh_processor.align_true_to_learned_mesh_with_icp()
    pdb.set_trace()


if __name__ == '__main__':
    # for object in ['bakingbox', 'gallon', 'greencan', 'crushedcan', 'stapler',
    #                'styrofoam']:
    # for object in ['gallon', 'crushedcan']:
    #     mesh_processor = UnscaledMeshProcessor(object=object)
    #     mesh_processor.scale_and_align_wrong_to_bundlesdf(show=True)

    # Looks good but maybe axis of symmetry misalignments:  bakingbox, greencan
    # Bad ones:  gallon (tried 00, no 01), crushedcan (01 better than 00)
    # Mostly ok:  stapler
    # Great:  styrofoam


    main_command()  # pylint: disable=no-value-for-parameter
