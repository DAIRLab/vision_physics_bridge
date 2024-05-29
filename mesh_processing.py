"""Utilities to help with processing ground truth meshes."""

import click
import numpy as np
import open3d as o3d
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
    
    TODO:  Needs to be tested on partial/bad mesh estimates.
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
        np.savetxt(
            op.join(save_dir, 'true_to_learned_tf.txt'), reg_p2p.transformation)

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
        o3d.io.write_triangle_mesh(
            op.join(save_dir, obj_name), self.true_mesh)
        print(f'Saved ground truth mesh transformed to align with ' + \
              f'BundleSDF mesh, as {obj_name} in {save_dir}.')


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

    # Load a mesh.
    # mesh_file = f'/home/bibit/vision/bundlenets/cnets-data-generation/assets/object_scans/oatly_top_low_res/mesh_lowpoly.obj'

    # trimesh_obj = trimesh.load(mesh_file, force='mesh')

    # # trimesh_obj.show()

    # trimesh_obj.fill_holes()
    # trimesh_obj.update_faces(trimesh_obj.nondegenerate_faces())
    # trimesh_obj.update_faces(trimesh_obj.unique_faces())
    # trimesh_obj.remove_infinite_values()
    # trimesh_obj.remove_unreferenced_vertices()

    # # Optionally, run additional repair steps to fix normals and winding
    # trimesh_obj.fix_normals()
    # trimesh.repair.fix_winding(trimesh_obj)

    # trimesh_obj.show()

    # repaired_mesh_file = mesh_file.replace('.obj', '_repaired.obj')
    # trimesh_obj.export(repaired_mesh_file)



    main_command()  # pylint: disable=no-value-for-parameter
