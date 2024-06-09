"""Utilities to help with processing ground truth meshes."""

import click
import copy
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import os
import os.path as op
import pdb
import shutil
from tempfile import TemporaryDirectory
from tqdm import tqdm

import file_utils, icp


# ICP hyperparameters
ICP_THRESHOLD = 0.02    # Maximum distance threshold for a point to be
                        # considered during alignment.

# Mesh scaling
CAD_MM = 'mm_in_cad'
REAL_INCHES = 'inches_in_real'
SCALING = {
    'bakingbox':    {CAD_MM: 907.717, REAL_INCHES: 9 + 9/16},
    'cardboard':    {CAD_MM: 832.803, REAL_INCHES: 7 + 1/16},
    'croc':         {CAD_MM: 1482.842, REAL_INCHES: 6 + 1/2},
    'crushedcan':   {CAD_MM: 736.847, REAL_INCHES: 4 + 9/16},
    'gallon':       {CAD_MM: 1028.132, REAL_INCHES: 9 + 13/16},
    'greencan':     {CAD_MM: 470.909, REAL_INCHES: 4 + 11/16},
    'oatly':        {CAD_MM: 2186.932, REAL_INCHES: 9 + 5/8},
    'stapler':      {CAD_MM: 407.748, REAL_INCHES: 3 + 1/8},
    'styrofoam':    {CAD_MM: 952.224, REAL_INCHES: 9 + 7/32}
}


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
        self.true_to_learned_transform = reg_p2p.transformation
        print(f'Solved TF matrix:\n{self.true_to_learned_transform}')
        if save_dir is not None:
            np.savetxt(
                op.join(save_dir, 'true_to_learned_tf.txt'),
                reg_p2p.transformation
            )

        # Transform the true mesh's point cloud by the solved transform and view
        # the results.
        true_cloud.transform(self.true_to_learned_transform)
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
            if op.exists(op.join(save_dir, obj_name)):
                print(f'Found {obj_name} already in {save_dir}.')
                pdb.set_trace()
            o3d.io.write_triangle_mesh(
                op.join(save_dir, obj_name), self.true_mesh,
                write_triangle_uvs=False, write_vertex_colors=False
            )
            print(f'Saved ground truth mesh transformed to align with ' + \
                f'BundleSDF mesh, as {obj_name} in {save_dir}.')

            material_filepath = op.join(
                save_dir, f'{obj_name.split(".")[0]}.mtl'
            )
            if op.exists(material_filepath):
                os.system(f'rm {material_filepath}')

    def beautify_meshes(self):
        # Check if the mesh has vertex normals; if not, compute them.
        if not self.learned_mesh.has_vertex_normals():
            self.learned_mesh.compute_vertex_normals()
        # Check if the mesh has face normals; if not, compute them.
        if not self.learned_mesh.has_triangle_normals():
            self.learned_mesh.compute_triangle_normals()

        # Same thing for true mesh.
        if not self.true_mesh.has_vertex_normals():
            self.true_mesh.compute_vertex_normals()
        if not self.true_mesh.has_triangle_normals():
            self.true_mesh.compute_triangle_normals()

    def interactive_align_true_to_learned_mesh_with_icp2(self, save_dir: str):
        self.beautify_meshes()

        # First, save a video of the learned mesh starting from its canonical
        # pose.
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False)
        vis.add_geometry(self.learned_mesh)
        view_control = vis.get_view_control()
        with TemporaryDirectory(prefix="mesh-images-") as tmpdir:
            print(f'Storing temporary files at {tmpdir}')
            # Loop to rotate the mesh and capture images.
            for i in tqdm(range(20)):
                view_control.rotate(1000/20, 0.0)
                vis.poll_events()
                vis.update_renderer()
                image = vis.capture_screen_float_buffer(do_render=True)
                plt.imsave(f"{tmpdir}/image_{i+1:07d}.png", np.asarray(image))
            vis.destroy_window()
            os.system(f'ffmpeg -y -r 30 -i {tmpdir}/image_%07d.png -vcodec ' + \
                      f'libx264 -preset slow -crf 18 {save_dir}/' + \
                      f'reference.mp4')

        # Use an interactive visualizer with key callbacks to do a manual
        # alignment of the true mesh to the learned mesh.  Refer back to the
        # reference video for the target pose to match the learned mesh.
        interactive_vis = InteractiveVisualizer(self.true_mesh)
        interactive_vis.run()

        manual_adjustments = interactive_vis.total_transformation
        print(f'Total transformation: {manual_adjustments}')

        # Double check the aggregate transformation from the little steps is
        # correct.
        interactive_vis.check_total_transformation(self.true_mesh)

        # Now do ICP, starting with this as an initialization.
        # Need to do ICP on 3D points instead of the mesh directly.
        true_cloud = self.true_mesh.sample_points_poisson_disk(2000)
        learned_cloud = self.learned_mesh.sample_points_poisson_disk(2000)

        # Visualize the initial alignment.
        o3d.visualization.draw_geometries(
            [true_cloud, learned_cloud], window_name="Initial Comparison")

        centered_transform = np.eye(4)
        centered_transform[:3, 3] = learned_cloud.get_center() - \
            true_cloud.get_center()
        transform_to_apply = manual_adjustments @ centered_transform
        true_cloud.transform(transform_to_apply)
        # Visualize the initial alignment.
        o3d.visualization.draw_geometries(
            [true_cloud, learned_cloud], window_name="Before ICP")
        true_cloud.transform(np.linalg.inv(transform_to_apply))

        # Apply ICP.
        reg_p2p = o3d.pipelines.registration.registration_icp(
            true_cloud, learned_cloud, ICP_THRESHOLD, transform_to_apply,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000)
        )
        print(f'FITNESS SCORE:  {reg_p2p.fitness}')

        # Transform the true mesh's point cloud by the solved transform and
        # view the results.
        true_cloud.transform(reg_p2p.transformation)
        o3d.visualization.draw_geometries(
            [true_cloud, learned_cloud],
            window_name=f'After ICP, starting from manual adjustments')

        print(f'Solved TF matrix:\n{reg_p2p.transformation}')

        # Save the transformation.
        np.savetxt(
            op.join(save_dir, 'true_to_learned_tf_assist.txt'),
            reg_p2p.transformation
        )

        # Save the transformed mesh to file.
        if op.exists(op.join(save_dir, 'true_geom_aligned_assist.obj')):
            print(f'Found true_geom_aligned_assist.obj already in ' + \
                  f'{save_dir}.')
            pdb.set_trace()
        self.true_mesh.transform(reg_p2p.transformation)
        o3d.io.write_triangle_mesh(
            op.join(save_dir, 'true_geom_aligned_assist.obj'),
            self.true_mesh, write_triangle_uvs=False, write_vertex_colors=False
        )
        print(f'Saved ground truth mesh transformed to align with ' + \
              f'BundleSDF mesh, as true_geom_aligned_assist.obj in ' + \
              f'{save_dir}.')

        material_filepath = op.join(
            save_dir, 'true_geom_aligned_assist.mtl')
        if op.exists(material_filepath):
            os.system(f'rm {material_filepath}')

    def interactive_align_true_to_learned_mesh_with_icp(
            self, save_dir: str = None,
            obj_name: str = 'true_geom_aligned_assist.obj'):
        """Goal is to create a ground truth mesh that shares the same body
        origin as the learned mesh.  To do this, use ICP to register the true
        mesh to the learned mesh."""
        # Need to do ICP on 3D points instead of the mesh directly.
        true_cloud = self.true_mesh.sample_points_poisson_disk(2000)
        learned_cloud = self.learned_mesh.sample_points_poisson_disk(2000)

        # Visualize the initial alignment.
        o3d.visualization.draw_geometries(
            [true_cloud, learned_cloud], window_name="Initial Comparison")
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

        do_manual_adjust = input('Manual adjustment?  ') == 'y'
        adjustments = 0
        total_manual_adjustments = np.eye(4)

        manual_adjs_str = ''

        while do_manual_adjust:
            adjustments += 1
            manual_transformation = np.eye(4)
            rot = input('Rotation (r) or Translation (other):  ') == 'r'
            axis = input('Axis (x, y, z):  ')
            if rot:
                angle = float(input('Angle (degrees):  '))
                manual_adjs_str += f'Adjustment {adjustments}: ' + \
                    f'{angle} about {axis}\n'
                if axis == 'x':
                    manual_transformation[:3, :3] = \
                        o3d.geometry.get_rotation_matrix_from_xyz(
                            (np.radians(angle), 0, 0))
                elif axis == 'y':
                    manual_transformation[:3, :3] = \
                        o3d.geometry.get_rotation_matrix_from_xyz(
                            (0, np.radians(angle), 0))
                elif axis == 'z':
                    manual_transformation[:3, :3] = \
                        o3d.geometry.get_rotation_matrix_from_xyz(
                            (0, 0, np.radians(angle)))
                else:
                    print(f'Invalid axis {axis}.')
            else:
                translation = float(input('Translation (meters):  '))
                manual_adjs_str += f'Adjustment {adjustments}: ' + \
                    f'{translation} along {axis}\n'
                if axis == 'x':
                    manual_transformation[0, 3] = translation
                elif axis == 'y':
                    manual_transformation[1, 3] = translation
                elif axis == 'z':
                    manual_transformation[2, 3] = translation
                else:
                    print(f'Invalid axis {axis}.')

            true_cloud.transform(manual_transformation)
            o3d.visualization.draw_geometries(
                [true_cloud, learned_cloud],
                window_name=f"Manual Adjustment {adjustments}")

            # Keep track of all the manual adjustments.
            total_manual_adjustments = \
                manual_transformation @ total_manual_adjustments
            do_manual_adjust = input('Manual adjustment?  ') == 'y'

        true_cloud.transform(np.linalg.inv(total_manual_adjustments))
        o3d.visualization.draw_geometries(
            [true_cloud, learned_cloud], window_name="Undid manual adjustments")

        # Coarse initial alignment to get the true mesh's cloud centroid to the
        # the same location as the learned mesh's cloud centroid.
        centered_transform = np.eye(4)
        centered_transform[:3, 3] = learned_cloud.get_center() - \
            true_cloud.get_center()
        transform_to_apply = total_manual_adjustments @ centered_transform
        true_cloud.transform(transform_to_apply)
        # Visualize the initial alignment.
        o3d.visualization.draw_geometries(
            [true_cloud, learned_cloud], window_name="Before ICP")
        true_cloud.transform(np.linalg.inv(transform_to_apply))

        # Apply ICP.
        reg_p2p = o3d.pipelines.registration.registration_icp(
            true_cloud, learned_cloud, ICP_THRESHOLD, transform_to_apply,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000)
        )
        print(f'FITNESS SCORE:  {reg_p2p.fitness}')

        # Transform the true mesh's point cloud by the solved transform and
        # view the results.
        true_cloud.transform(reg_p2p.transformation)
        o3d.visualization.draw_geometries(
            [true_cloud, learned_cloud],
            window_name=f'After ICP, with {adjustments} manual adjustments')

        print(f'Solved TF matrix:\n{reg_p2p.transformation}')
        if save_dir is not None:
            np.savetxt(
                op.join(save_dir, 'true_to_learned_tf_assist.txt'),
                reg_p2p.transformation
            )
            with open(op.join(save_dir, 'manual_inputs.txt'), 'w') as txt_file:
                txt_file.write(manual_adjs_str)
            print(f'Wrote to {op.join(save_dir, "manual_inputs.txt")}')


        if save_dir is not None:
            if op.exists(op.join(save_dir, obj_name)):
                print(f'Found {obj_name} already in {save_dir}.')
                pdb.set_trace()
            self.true_mesh.transform(reg_p2p.transformation)
            o3d.io.write_triangle_mesh(
                op.join(save_dir, obj_name), self.true_mesh,
                write_triangle_uvs=False, write_vertex_colors=False
            )
            print(f'Saved ground truth mesh transformed to align with ' + \
                f'BundleSDF mesh, as {obj_name} in {save_dir}.')

            material_filepath = op.join(
                save_dir, f'{obj_name.split(".")[0]}.mtl'
            )
            if op.exists(material_filepath):
                os.system(f'rm {material_filepath}')


class InteractiveVisualizer:
    def __init__(self, mesh):
        self.mesh = copy.deepcopy(mesh)
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window()

        # Add the mesh to the visualizer
        self.vis.add_geometry(self.mesh)

        # Transformation matrix
        self.total_transformation = np.eye(4)

        # Register key callbacks
        self.vis.register_key_callback(ord("W"), self.translate_forward)
        self.vis.register_key_callback(ord("S"), self.translate_backward)
        self.vis.register_key_callback(ord("A"), self.translate_left)
        self.vis.register_key_callback(ord("D"), self.translate_right)
        self.vis.register_key_callback(ord("Q"), self.translate_up)
        self.vis.register_key_callback(ord("E"), self.translate_down)
        self.vis.register_key_callback(ord("J"), self.rotate_left)
        self.vis.register_key_callback(ord("L"), self.rotate_right)
        self.vis.register_key_callback(ord("I"), self.rotate_up)
        self.vis.register_key_callback(ord("K"), self.rotate_down)
        self.vis.register_key_callback(ord("O"), self.rotate_clockwise)
        self.vis.register_key_callback(ord("U"), self.rotate_counterclockwise)

    def translate_forward(self, vis):
        self.new_transformation = np.eye(4)
        self.new_transformation[:3, 3] += [0, 0, -0.1]
        self.update_mesh()

    def translate_backward(self, vis):
        self.new_transformation = np.eye(4)
        self.new_transformation[:3, 3] += [0, 0, 0.1]
        self.update_mesh()

    def translate_left(self, vis):
        self.new_transformation = np.eye(4)
        self.new_transformation[:3, 3] += [-0.1, 0, 0]
        self.update_mesh()

    def translate_right(self, vis):
        self.new_transformation = np.eye(4)
        self.new_transformation[:3, 3] += [0.1, 0, 0]
        self.update_mesh()

    def translate_up(self, vis):
        self.new_transformation = np.eye(4)
        self.new_transformation[:3, 3] += [0, 0.1, 0]
        self.update_mesh()

    def translate_down(self, vis):
        self.new_transformation = np.eye(4)
        self.new_transformation[:3, 3] += [0, -0.1, 0]
        self.update_mesh()

    def rotate_left(self, vis):
        self.new_transformation = np.eye(4)
        R = self.get_rotation_matrix(np.deg2rad(5), [0, 1, 0])
        self.new_transformation[:3, :3] = R @ self.new_transformation[:3, :3]
        self.update_mesh()

    def rotate_right(self, vis):
        self.new_transformation = np.eye(4)
        R = self.get_rotation_matrix(np.deg2rad(-5), [0, 1, 0])
        self.new_transformation[:3, :3] = R @ self.new_transformation[:3, :3]
        self.update_mesh()

    def rotate_up(self, vis):
        self.new_transformation = np.eye(4)
        R = self.get_rotation_matrix(np.deg2rad(5), [1, 0, 0])
        self.new_transformation[:3, :3] = R @ self.new_transformation[:3, :3]
        self.update_mesh()

    def rotate_down(self, vis):
        self.new_transformation = np.eye(4)
        R = self.get_rotation_matrix(np.deg2rad(-5), [1, 0, 0])
        self.new_transformation[:3, :3] = R @ self.new_transformation[:3, :3]
        self.update_mesh()

    def rotate_clockwise(self, vis):
        self.new_transformation = np.eye(4)
        R = self.get_rotation_matrix(np.deg2rad(5), [0, 0, 1])
        self.new_transformation[:3, :3] = R @ self.new_transformation[:3, :3]
        self.update_mesh()

    def rotate_counterclockwise(self, vis):
        self.new_transformation = np.eye(4)
        R = self.get_rotation_matrix(np.deg2rad(-5), [0, 0, 1])
        self.new_transformation[:3, :3] = R @ self.new_transformation[:3, :3]
        self.update_mesh()

    def get_rotation_matrix(self, angle, axis):
        axis = np.array(axis)
        axis = axis / np.linalg.norm(axis)
        cos_theta = np.cos(angle)
        sin_theta = np.sin(angle)
        one_minus_cos_theta = 1 - cos_theta
        ux, uy, uz = axis

        R = np.array([
            [cos_theta + ux**2 * one_minus_cos_theta, ux*uy * one_minus_cos_theta - uz*sin_theta, ux*uz * one_minus_cos_theta + uy*sin_theta],
            [uy*ux * one_minus_cos_theta + uz*sin_theta, cos_theta + uy**2 * one_minus_cos_theta, uy*uz * one_minus_cos_theta - ux*sin_theta],
            [uz*ux * one_minus_cos_theta - uy*sin_theta, uz*uy * one_minus_cos_theta + ux*sin_theta, cos_theta + uz**2 * one_minus_cos_theta]
        ])
        return R

    def update_mesh(self):
        self.mesh.transform(self.new_transformation)
        self.vis.update_geometry(self.mesh)
        self.vis.poll_events()
        self.vis.update_renderer()

        self.total_transformation = self.new_transformation @ \
            self.total_transformation
        self.new_transformation = np.eye(4)  # Reset new transformation after applying

    def run(self):
        self.vis.run()
        self.vis.destroy_window()

    def check_total_transformation(self, mesh_again):
        mesh_again = copy.deepcopy(mesh_again)

        # Visualize before.
        o3d.visualization.draw_geometries(
            [mesh_again], window_name="Before transformation")

        # Then apply the transform and visualize again.
        mesh_again.transform(self.total_transformation)
        o3d.visualization.draw_geometries(
            [mesh_again], window_name="After transformation")


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
            file_utils.object_scan_dir(), 'ICP_scaling')

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
            file_utils.object_scan_dir(), 'ICP_scaling',
            f'{self.object}.obj'
        )
        o3d.io.write_triangle_mesh(corrected_filepath, self.wrong_scaling_mesh,
                write_triangle_uvs=False, write_vertex_colors=False)
        print(f'Saved ground truth mesh transformed to align with ' + \
              f'BundleSDF mesh, as {self.object}.obj in ICP_scaling.')
        material_filepath = op.join(
            file_utils.object_scan_dir(), 'ICP_scaling',
            f'{self.object}.mtl'
        )
        if op.exists(material_filepath):
            os.system(f'rm {material_filepath}')


class MeshScalingProcessor(UnscaledMeshProcessor):
    def __init__(self, object: str):
        super().__init__(object)

    def scale_manually(self, show=True):
        scalings = SCALING[self.object]
        true_length_meters = scalings[REAL_INCHES] * 2.54 / 100
        current_length_meters = scalings[CAD_MM] * 0.001

        scale_factor = true_length_meters / current_length_meters

        inspection_dir = op.join(
            file_utils.object_scan_dir(), 'CAD_scaling')

        # Show initial scaling.
        bsdf_cloud = self.bundlesdf_mesh.sample_points_poisson_disk(2000)
        wrong_cloud = self.wrong_scaling_mesh.sample_points_poisson_disk(2000)
        if show:
            o3d.visualization.draw_geometries(
                [bsdf_cloud, wrong_cloud], window_name='Before Scaling')

        # Save the before image.
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

        # Apply the scale factor.
        self.wrong_scaling_mesh.scale(
            scale_factor, center=self.wrong_scaling_mesh.get_center())

        # Visualize the result.
        bsdf_cloud = self.bundlesdf_mesh.sample_points_poisson_disk(2000)
        wrong_cloud = self.wrong_scaling_mesh.sample_points_poisson_disk(2000)
        if show:
            o3d.visualization.draw_geometries(
                [bsdf_cloud, wrong_cloud], window_name='After Scaling')

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

        # Save the result.
        corrected_filepath = op.join(
            file_utils.object_scan_dir(), 'CAD_scaling',
            f'{self.object}.obj'
        )
        o3d.io.write_triangle_mesh(corrected_filepath, self.wrong_scaling_mesh,
                write_triangle_uvs=False, write_vertex_colors=False)
        print(f'Saved ground truth mesh scaled to match CAD as ' + \
              f'{self.object}.obj in CAD_scaling.')
        material_filepath = op.join(
            file_utils.object_scan_dir(), 'CAD_scaling',
            f'{self.object}.mtl'
        )
        if op.exists(material_filepath):
            os.system(f'rm {material_filepath}')


class MeshInspector:
    def __init__(self, object: str):
        self.object = object
        true_obj_file = file_utils.object_scan_filepath(self.object)
        self.true_mesh = icp.load_mesh_from_obj(true_obj_file)

    def load_bsdf_and_tagslam_pll_meshes(self):
        """Load PLL meshes:  one trained on BundleSDF poses and one trained on
        TagSLAM poses."""
        vision_asset = f'{self.object}_1'

        pb_results_dir = file_utils.contactnets_output_dir(
            vision_asset, 1, 'pll_id_04')
        pb_obj_file = op.join(pb_results_dir, 'urdfs', 'test.obj')
        if not op.exists(pb_obj_file):
            pb_obj_file = op.join(pb_results_dir, 'urdfs', 'test_best.obj')
        assert op.exists(pb_obj_file), f'Could not find {pb_obj_file}.'

        pt_results_dir = file_utils.contactnets_output_dir(
            vision_asset, 0, 'pll_id_00')
        pt_obj_file = op.join(pt_results_dir, 'urdfs', 'test.obj')
        if not op.exists(pt_obj_file):
            pt_obj_file = op.join(pt_results_dir, 'urdfs', 'test_best.obj')
        assert op.exists(pt_obj_file), f'Could not find {pt_obj_file}.'

        pb2_results_dir = file_utils.contactnets_output_dir(
            vision_asset, 1, 'pll_id_05')
        pb2_obj_file = op.join(pb2_results_dir, 'urdfs', 'test.obj')
        if not op.exists(pb2_obj_file):
            pb2_obj_file = op.join(pb2_results_dir, 'urdfs', 'test_best.obj')
        assert op.exists(pb2_obj_file), f'Could not find {pb2_obj_file}.'

        pb3_results_dir = file_utils.contactnets_output_dir(
            vision_asset, 1, 'pll_id_db00')
        # pb3_results_dir = file_utils.contactnets_output_dir(
        #     vision_asset, 1, 'pll_id_07')
        pb3_obj_file = op.join(pb3_results_dir, 'urdfs', 'test.obj')
        if not op.exists(pb3_obj_file):
            pb3_obj_file = op.join(pb3_results_dir, 'urdfs', 'test_best.obj')
        assert op.exists(pb3_obj_file), f'Could not find {pb3_obj_file}.'

        pt2_results_dir = file_utils.contactnets_output_dir(
            vision_asset, 1, 'pll_id_08')
        pt2_obj_file = op.join(pt2_results_dir, 'urdfs', 'test.obj')
        if not op.exists(pt2_obj_file):
            pt2_obj_file = op.join(pt2_results_dir, 'urdfs', 'test_best.obj')
        assert op.exists(pt2_obj_file), f'Could not find {pt2_obj_file}.'

        pt3_results_dir = file_utils.contactnets_output_dir(
            vision_asset, 0, 'pll_id_09')
        pt3_obj_file = op.join(pt3_results_dir, 'urdfs', 'test.obj')
        if not op.exists(pt3_obj_file):
            pt3_obj_file = op.join(pt3_results_dir, 'urdfs', 'test_best.obj')
        assert op.exists(pt3_obj_file), f'Could not find {pt3_obj_file}.'

        self.pb_mesh = icp.load_mesh_from_obj(pb_obj_file)
        self.pt_mesh = icp.load_mesh_from_obj(pt_obj_file)
        self.pb2_mesh = icp.load_mesh_from_obj(pb2_obj_file)
        self.pb3_mesh = icp.load_mesh_from_obj(pb3_obj_file)
        self.pt2_mesh = icp.load_mesh_from_obj(pt2_obj_file)
        self.pt3_mesh = icp.load_mesh_from_obj(pt3_obj_file)

    def view_true_bsdf_tagslam_pll_meshes(self):
        true_cloud = self.true_mesh.sample_points_poisson_disk(2000)
        pb_cloud = self.pb_mesh.sample_points_poisson_disk(2000)
        pt_cloud = self.pt_mesh.sample_points_poisson_disk(2000)
        pb2_cloud = self.pb2_mesh.sample_points_poisson_disk(2000)
        pb3_cloud = self.pb3_mesh.sample_points_poisson_disk(2000)
        pt2_cloud = self.pt2_mesh.sample_points_poisson_disk(2000)
        pt3_cloud = self.pt3_mesh.sample_points_poisson_disk(2000)
        o3d.visualization.draw_geometries(
            [true_cloud, pb_cloud, pt_cloud, pb2_cloud, pb3_cloud, pt2_cloud,
             pt3_cloud],
            window_name='Aligned'
        )

        # Now split them up a bit.
        slide_right = np.eye(4)
        slide_right[0, 3] = 0.1
        slide_right_more = np.eye(4)
        slide_right_more[0, 3] = 0.2
        slide_right_more_more = np.eye(4)
        slide_right_more_more[0, 3] = 0.3
        slide_right_more_more_more = np.eye(4)
        slide_right_more_more_more[0, 3] = 0.4
        slide_right_more_more_more_more = np.eye(4)
        slide_right_more_more_more_more[0, 3] = 0.5
        slide_right_more_more_more_more_more = np.eye(4)
        slide_right_more_more_more_more_more[0, 3] = 0.6
        pb_cloud.transform(slide_right)
        pt_cloud.transform(slide_right_more)
        pb2_cloud.transform(slide_right_more_more)
        pb3_cloud.transform(slide_right_more_more_more)
        pt2_cloud.transform(slide_right_more_more_more_more)
        pt3_cloud.transform(slide_right_more_more_more_more_more)
        o3d.visualization.draw_geometries(
            [true_cloud, pb_cloud, pt_cloud, pb2_cloud, pb3_cloud, pt2_cloud,
             pt3_cloud],
            window_name='True-BSDF-TagSLAM-BSDF2-TagSLAM2-TagSLAM3')



#######################################################################
@click.group()
def cli():
    pass


@cli.command('manual_icp')
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

def process_manual_icp_command(vision_asset: str, bundlesdf_id: str,
                               nerf_bundlesdf_id: str, cycle_iteration: int):
    # Decode the BundleSDF run ID.
    tracking_bundlesdf_id = bundlesdf_id
    if tracking_bundlesdf_id[:13] != 'bundlesdf_id_':
        tracking_bundlesdf_id = f'bundlesdf_id_{tracking_bundlesdf_id}'
    if nerf_bundlesdf_id is None:
        nerf_bundlesdf_id = tracking_bundlesdf_id
    elif nerf_bundlesdf_id[:13] != 'bundlesdf_id_':
        nerf_bundlesdf_id = f'bundlesdf_id_{nerf_bundlesdf_id}'

    eval_dir = file_utils.evaluation_subdir(
        dataset=vision_asset, cycle_iteration=cycle_iteration,
        tracking_bundlesdf_id=tracking_bundlesdf_id,
        nerf_bundlesdf_id=nerf_bundlesdf_id
    )

    mesh_processor = MeshProcessor(
        vision_asset, tracking_bundlesdf_id, nerf_bundlesdf_id, cycle_iteration)

    mesh_processor.interactive_align_true_to_learned_mesh_with_icp2(
        save_dir=eval_dir)


@cli.command('distribute_alignments')
def process_distribute_alignments_command():
    """This command looks for every true_geom_aligned_assist.obj in any
    evaluation/{vision_asset}_02_02_2/ directories, and writes a corresponding
    assets/object_scans/true_aligned_to_experiments/{vision_asset}.obj file."""
    eval_dir = file_utils.evaluation_dir()
    aligned_scan_dir = file_utils.aligned_object_scan_dir()

    # First gather all of the existing aligned meshes from the BundleSDF ID 02,
    # cycle iteration 2 results.
    print(f'Copying aligned meshes to aligned scan directory:')
    for eval_subdir in os.listdir(eval_dir):
        if eval_subdir.endswith('02_02_2'):
            in_eval_path = op.join(
                eval_dir, eval_subdir, 'true_geom_aligned_assist.obj')
            if op.exists(in_eval_path):
                vision_asset = eval_subdir.split('_02_02_2')[0]
                in_aligned_path = op.join(
                    aligned_scan_dir, f'{vision_asset}.obj')
                shutil.copyfile(in_eval_path, in_aligned_path)
                print(f'  {vision_asset}')

    # Now copy all of the aligned meshes to every experiment's evaluation
    # folder.
    print(f'\nCopying aligned meshes to every experiment\'s evaluation folder:')
    for eval_subdir in os.listdir(eval_dir):
        if not op.isdir(op.join(eval_dir, eval_subdir)):
            continue
        vision_asset = eval_subdir.split('_02_02_2')[0]
        aligned_path = op.join(aligned_scan_dir, f'{vision_asset}.obj')
        if not op.exists(aligned_path):
            print(f'  Skipping {eval_subdir}:  could not find aligned for ' + \
                  f'{vision_asset}.')
            continue
        shutil.copyfile(
            aligned_path,
            op.join(eval_dir, eval_subdir, 'true_geom_aligned_assist.obj')
        )
        print(f'  {eval_subdir}')



if __name__ == '__main__':
    # for object in ['bakingbox', 'gallon', 'greencan', 'crushedcan', 'stapler',
    #                'styrofoam']:
    # for object in ['cardboard']:
    #     mesh_processor = UnscaledMeshProcessor(object=object)
    #     mesh_processor.scale_and_align_wrong_to_bundlesdf(show=True)

    # Looks good but maybe axis of symmetry misalignments:  bakingbox, greencan
    # Bad ones:  gallon (tried 00, no 01), crushedcan (01 better than 00)
    # Mostly ok:  stapler
    # Great:  styrofoam

    # for object in SCALING.keys():
    # for object in ['croc']:
    #     mesh_processor = MeshScalingProcessor(object=object)
    #     mesh_processor.scale_manually(show=True)

    # mi = MeshInspector(object='milk')
    # mi.load_bsdf_and_tagslam_pll_meshes()
    # mi.view_true_bsdf_tagslam_pll_meshes()


    cli()  # pylint: disable=no-value-for-parameter
