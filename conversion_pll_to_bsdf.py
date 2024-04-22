"""This file performs output conversions from PLL shapes to input formats
required by BundleSDF."""

import click
import numpy as np
import os
import os.path as op
import torch

import file_utils
import math_utils


# In transforming from PLL geometry outputs to BundleSDF geometry inputs, some
# of the files represent scalar fields instead of point clouds.  These files
# are excluded from the transformation process.
TRANSFORM_EXCLUDE_FILENAMES = [
    'sdfs.pt',
    'sdf_bounds.pt',
    'support_point_normal_forces.pt'
]


def batch_no_transform_function(points_wrt_T: np.ndarray,
                                _bsdf_output_pose_dir: str,
                                _annotated_poses_dir: str) -> np.ndarray:
    """Function that does not transform the points at all.  This is used when
    the PLL run used BundleSDF poses and no transformation is needed."""
    return points_wrt_T


class ConverterPLLToBundleSDF:
    """Class for processing shape data from PLL.  All of the data in a PLL run's
    geometry output directory is processed.

    The only change that might be needed is if the PLL run used TagSLAM poses
    instead of BundleSDF poses.  In that case, PLL's body origin matches
    TagSLAM's and needs to be converted to BundleSDF's origin.  TODO"""
    def __init__(self, pll_geom_output_dir: str,
                 bundlesdf_geom_input_dir: str,
                 bundlesdf_pose_output_dir: str,
                 annotated_poses_dir: str):
        # Do some checks on the input directories.
        assert op.basename(pll_geom_output_dir) == 'geom_for_bsdf', \
            f'Unexpected PLL geometry output folder {pll_geom_output_dir}--' + \
            f' was looking for geom_for_bsdf folder.'
        assert op.basename(op.dirname(pll_geom_output_dir)) == \
            op.basename(bundlesdf_geom_input_dir), f'Expecting to find ' + \
            f'consistent PLL run IDs in {pll_geom_output_dir=} and ' + \
            f'{bundlesdf_geom_input_dir=}.'

        # If the BundleSDF pose directory and annotated poses directory are not
        # provided, then it is assumed that the PLL run used BundleSDF poses
        # and no transformation is needed.
        self.do_tagslam_to_bsdf_transform = True
        if bundlesdf_pose_output_dir is None:
            assert annotated_poses_dir is None, f'Expecting both ' + \
                f'{bundlesdf_pose_output_dir=} and {annotated_poses_dir=} ' + \
                f'to be None or both to be provided.'
            self.do_tagslam_to_bsdf_transform = False
            print('No transformation needed--PLL run used BundleSDF poses; ' + \
                  'conversion will copy files without modification.')

        # Store the directories.
        self.pll_geom_output_dir = pll_geom_output_dir
        self.bundlesdf_geom_input_dir = bundlesdf_geom_input_dir
        self.annotated_poses_dir = annotated_poses_dir
        self.bundlesdf_pose_output_dir = bundlesdf_pose_output_dir

    def process_and_save(self):
        """Process the data by recursively iterating through all the contents of
        the PLL run's output geometry directory, transforming them so they are
        represented in the BundleSDF origin's frame, and copying those
        transformed results to the BundleSDF's input geometry directory."""
        self._recursive_copy_and_transform(
            source_directory=self.pll_geom_output_dir,
            destination_directory=self.bundlesdf_geom_input_dir
        )
        print(f'\nFinished processing {self.pll_geom_output_dir} and put' + \
              f' the results in {self.bundlesdf_geom_input_dir}.')

    def _recursive_copy_and_transform(self, source_directory: str,
                                      destination_directory: str,
                                      prefix: str = ''):
        """Copy the file structure from the PLL geometry output directory to the
        BundleSDF geometry input directory."""
        for content in os.listdir(source_directory):
            # Recursively go down directory structure.
            if op.isdir(op.join(source_directory, content)):
                print(f'{prefix} Making {content} directory...')
                os.system(f'mkdir -p {op.join(destination_directory, content)}')
                self._recursive_copy_and_transform(
                    op.join(source_directory, content),
                    op.join(destination_directory, content),
                    prefix=prefix + '  '
                )
                continue
            
            # Copy files without modification that are not point clouds.
            if content in TRANSFORM_EXCLUDE_FILENAMES:
                print(f'{prefix} Copying unmodified {content}...')
                os.system(f'cp {op.join(source_directory, content)} ' + \
                    f'{op.join(destination_directory, content)}')
                continue

            # Transform the point cloud files.
            assert content.split('.')[-1] == 'pt', f'Expecting only ' + \
                f'.pt files in {source_directory=} but found {content}.'
            print(f'{prefix} Transforming {content}... ', end='')
            self._transform_and_save(op.join(source_directory, content),
                                     destination_directory)
        
    def _transform_and_save(self, source_file: str,
                            destination_directory: str) -> None:
        """Transform the points from TagSLAM origin to BundleSDF origin and save
        the result."""
        points_wrt_tagslam = torch.load(source_file).detach().numpy()
        assert points_wrt_tagslam.shape[1] == 3, f'Expecting (N, 3) shape ' + \
            f'in {source_file=}, but found {points_wrt_tagslam.shape=}.'

        n_points = points_wrt_tagslam.shape[0]
        print(f' {n_points} points.')

        # Create a batched transformation matrix (n_points, 4, 4).
        trans_mat_t = np.eye(4).reshape(1, 4, 4).repeat(n_points, axis=0)
        trans_mat_t[:, :3, 3] = points_wrt_tagslam

        # Perform the batched conversion.
        batch_transform_t_to_b_function = batch_no_transform_function if not \
            self.do_tagslam_to_bsdf_transform else \
            math_utils.transform_points_wrt_tagslam_origin_to_bundletrack_origin
        trans_mat_b = batch_transform_t_to_b_function(
            points_wrt_T=trans_mat_t,
            bsdf_output_pose_dir=self.bundlesdf_pose_output_dir,
            annotated_poses_dir=self.annotated_poses_dir
        )
        points_wrt_bundletrack = trans_mat_b[:, :3, 3]
        
        # Save the transformed points as a torch .pt file.
        filename = op.basename(source_file)
        torch.save(torch.tensor(points_wrt_bundletrack),
                   op.join(destination_directory, filename))


#######################################################################
@click.command()
@click.option('--vision-asset',
              type=str,
              default=None,
              help="directory of the asset folder e.g. cube_2; encodes " + \
                "system and tosses.")
@click.option('--pll-id',
              type=str,
              default=None,
              help="what PLL run ID associated with geometry estimates to use.")
@click.option('--cycle-iteration',
              type=int,
              default=1,
              help="BundleSDF-PLL cycle iteration number that the PLL " + \
                "geometry estimates come from (-1 means PLL used TagSLAM " + \
                "poses, 0 is invalid).")

def main_command(vision_asset: str, pll_id: str, cycle_iteration: int):
    # First decode the system and start/end tosses from the provided asset
    # directory.
    assert '_' in vision_asset, f'Invalid asset directory: {vision_asset}.'
    assert cycle_iteration != 0, f'cycle_iteration must be positive (PLL ' + \
        f'trained with BundleSDF tracking) or -1 (PLL trained with TagSLAM ' + \
        f'tracking), but got {cycle_iteration=}.'

    start_toss = int(vision_asset.split('_')[1].split('-')[0])
    end_toss = start_toss if '-' not in vision_asset else \
        int(vision_asset.split('-')[1])
    assert start_toss <= end_toss, f'Invalid toss range: {start_toss} ' + \
        f'-{end_toss} inferred from {vision_asset=}.'
    
    # Ensure the PLL run ID starts with prefix.
    if pll_id[:7] != 'pll_id_':
        pll_id = f'pll_id_{pll_id}'

    # Load the PLL run results folder.
    pll_output_dir = file_utils.contactnets_output_dir(
        vision_asset, cycle_iteration=cycle_iteration, pll_id=pll_id)
    pll_geometry_output_dir = file_utils.contactnets_output_geometry_dir(
        vision_asset, cycle_iteration=cycle_iteration, pll_id=pll_id)

    # No origin conversion is needed if the PLL run used BundleSDF poses.  In
    # that case, the geometry arrays will just be copied over exactly.
    # Otherwise, the BundleSDF/TagSLAM poses are needed for the conversion.
    if cycle_iteration <= 0:
        bundlesdf_pose_output_dir = None
        annotated_poses_dir = None
    else:
        # Load the BundleSDF run results folder from the pose data that was
        # provided the PLL run.
        bundlesdf_id = file_utils.load_bundlesdf_id_from_pll_json(
            pll_output_dir)
        bundlesdf_pose_output_dir = file_utils.bundlesdf_pose_dir(
            vision_asset, cycle_iteration=cycle_iteration,
            bundlesdf_id=bundlesdf_id)

        # Load the annotated poses directory, which contains 0000.txt with the
        # first frame's TagSLAM pose.
        annotated_poses_dir = file_utils.bundlesdf_annotated_poses_dir(
            vision_asset)

    bundlesdf_geometry_input_dir = file_utils.bundlesdf_geometry_dir(
        vision_asset, cycle_iteration=cycle_iteration, pll_run_id=pll_id,
        create=True
    )

    converter = ConverterPLLToBundleSDF(
        pll_geom_output_dir=pll_geometry_output_dir,
        bundlesdf_geom_input_dir=bundlesdf_geometry_input_dir,
        bundlesdf_pose_output_dir=bundlesdf_pose_output_dir,
        annotated_poses_dir=annotated_poses_dir
    )

    converter.process_and_save()


if __name__ == "__main__":
    main_command()  # pylint: disable=no-value-for-parameter
