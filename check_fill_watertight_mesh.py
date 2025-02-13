"""
This script is used to check if the ground truth mesh is watertight, 
and to calculate the volumetric error and IoU between the input grid and the ground truth mesh.
The input grid is in the normalized unit cube scale, and the ground truth mesh is in the real world scale.
The parameters needed for the scale conversion is stored in the input grid file.
"""

import trimesh
import numpy as np
import os
from numpy import ndarray

def transform_points(points: ndarray, transformation_matrix: ndarray):
    ones = np.ones((points.shape[0], 1))
    points_homogeneous = np.hstack([points, ones])
    transformed_points = points_homogeneous @ transformation_matrix.T
    return transformed_points[:, :3]

def inverse_homogeneous_transformation(transform: ndarray) -> ndarray:
    """Produce the inverse of a homogeneous transform.  If a homogeneous
    transformation matrix is of the form:
    
        T = [ R  d ]
            [ 0  1 ]

    for T (4,4), R (3,3), and d (3,1), then the inverse is:
    
        inv(T) = [ R^T  -R^T*d ]
                 [  0      1   ]
    
    """
    assert transform.shape == (4, 4)

    # Split into rotation and translation components.
    rot_mat = transform[:3, :3]
    translation = transform[:3, 3]

    # Build the inverse.
    inv_transform = np.zeros((4, 4))
    inv_transform[:3, :3] = rot_mat.T
    inv_transform[:3, 3] = -rot_mat.T @ translation
    inv_transform[3, 3] = 1

    return inv_transform
def transform_pts_to_normalized_space(points, translation, sc_factor, offset):
    """From Utils.py's mesh_to_real_world function, we've uncovered that the
    conversion from the SDF function's space to real world space is:
    
        # The basic structure from Utils.py's mesh_to_real_world implements:
        geom_origin_pts = sdf_pts/sc_factor - translation
        track_origin_pts = geom_origin_pts.apply(offset)

    Thus, this function needs to do the opposite.

        # Reverse order yields:
        geom_origin_pts = track_origin_pts.apply(inv(offset))
        sdf_pts = (geom_origin_pts + translation) * sc_factor
    """
    # The provided points are wrt the tracking body origin.
    track_origin_pts = points

    # Convert to wrt the geometry body origin, determined via the offset.
    inv_offset = inverse_homogeneous_transformation(offset)
    geom_origin_pts = transform_points(track_origin_pts, inv_offset)

    # Convert to the scaled space of the SDF inputs.
    sdf_pts = (geom_origin_pts + translation.reshape(1,3)) * sc_factor
    return sdf_pts

# Load the input grid
grid_path = '/mnt/data0/minghz/repos/bundlenets/results/bakingbox_1/bundlesdf_iteration_1/bundlesdf_id_00-cvwo-occ-grid/nerf_runs/bundlesdf_id_00-cvwo-occ-grid/occ_grid.npz'
input_grid = np.load(grid_path)

occ_bits = input_grid['occ_bits']
bounds = input_grid['bounds']
voxel_size = input_grid['voxel_size']
Nx = input_grid['Nx']
sc_factor = input_grid['sc_factor']
translation = input_grid['translation']
offset = input_grid['offset']
occ_grid = np.unpackbits(occ_bits, count=Nx*Nx*Nx).reshape(Nx, Nx, Nx)
print("Input grid bounds: ", bounds)


# # Load the ground truth mesh
### All meshes under the object_scans folder without 'old' in the fuke name should be watertight.
# ground_truth_folder = '/mnt/data0/minghz/repos/bundlenets/cnets-data-generation/assets/object_scans/'
# for file in os.listdir(ground_truth_folder):
#     if file.endswith("old.obj"):
#         continue
#     if file.endswith(".obj"):
#         print(f"Processing {file}")
#         ground_truth_mesh = trimesh.load_mesh(os.path.join(ground_truth_folder, file))
#         if isinstance(ground_truth_mesh, trimesh.Scene):
#             print(f"Scene {file}")
#             mesh = trimesh.util.concatenate([
#                 trimesh.Trimesh(vertices=m.vertices, faces=m.faces)
#                 for m in ground_truth_mesh.geometry.values()])
#             ground_truth_mesh = mesh
#         is_ground_truth_watertight = ground_truth_mesh.is_watertight
#         print(f"Is {file} gt mesh watertight? {is_ground_truth_watertight}")
#         # break
# ground_truth_mesh = trimesh.load_mesh('/mnt/data0/minghz/repos/bundlenets/cnets-data-generation/assets/object_scans/bakingbox.obj')

### Load the ground truth mesh that is aligned with the estimation from the results folder (or the evaluation folder)
ground_truth_mesh = trimesh.load_mesh('/mnt/data0/minghz/repos/bundlenets/results/bakingbox_1/bundlesdf_iteration_1/bundlesdf_id_00-cvwo-occ/nerf_runs/bundlesdf_id_00-cvwo-occ2x2-grid/bakingbox_blender_true_scale_aligned.obj')

# Check if the ground truth mesh is watertight
is_ground_truth_watertight = ground_truth_mesh.is_watertight
print(f"Is the ground truth mesh watertight? {is_ground_truth_watertight}")


### Ground truth mesh is in real world scale. Input grid is in normalized unit cube scale.
### Unify them to the normalized unit cube scale,
### so that the resolution of different objects are similar. 

### The ground truth mesh may go out of the unit cube bound. 
### We first convert the ground truth mesh to the normalized unit cube scale. 
vertices_gt = ground_truth_mesh.vertices
vertices_gt_normalized = transform_pts_to_normalized_space(
    vertices_gt, translation, sc_factor, offset)
ground_truth_mesh.vertices = vertices_gt_normalized
### print bound
print(f"Ground truth mesh bounds: {ground_truth_mesh.bounds}")

### Calculate the bounding box of the ground truth mesh, 
### then we determine the overall bounding box of the input mesh and the ground truth mesh, 
### and initialize the voxel grid with the common bounding box.
### The grid points of input grid should be a subset of the new grid.
min_bounds = np.minimum(bounds[0], ground_truth_mesh.bounds[0])
max_bounds = np.maximum(bounds[1], ground_truth_mesh.bounds[1])

### Round the min and max bounds to the outer nearest grid point.
grid_zero_in_cube = bounds[0]
min_bounds_in_grid = np.floor((min_bounds - bounds[0]) / voxel_size)
min_bounds = min_bounds_in_grid * voxel_size + bounds[0]
max_bounds_in_grid = np.ceil((max_bounds - bounds[0]) / voxel_size)
max_bounds = max_bounds_in_grid * voxel_size + bounds[0]
grid_size = max_bounds_in_grid - min_bounds_in_grid + np.ones(3)
### Make sure the grid_size is odd in each dimension.
for i in range(3):
    if grid_size[i] % 2 == 0:
        grid_size[i] += 1
        max_bounds[i] += voxel_size
        max_bounds_in_grid[i] += 1

### Make the grid_size the same in each dimension.
max_grid_size = np.max(grid_size)
for i in range(3):
    if grid_size[i] < max_grid_size:
        diff = max_grid_size - grid_size[i]
        min_bounds[i] -= diff / 2 * voxel_size
        max_bounds[i] += diff / 2 * voxel_size
        grid_size[i] = max_grid_size
        min_bounds_in_grid[i] -= diff / 2
        max_bounds_in_grid[i] += diff / 2

### Find the cube coord of the center grid point.
grid_half_n = (max_grid_size - 1) / 2
print(f"{grid_half_n=}")
print(f"{min_bounds_in_grid=}, {max_bounds_in_grid=}")
print(f"{min_bounds=}, {max_bounds=}")
min_bounds_in_grid = min_bounds_in_grid.astype(int)
max_bounds_in_grid = max_bounds_in_grid.astype(int)
grid_half_n = int(grid_half_n)

center_in_grid = grid_half_n + min_bounds_in_grid
center_in_cube = center_in_grid * voxel_size + bounds[0]
gt_voxel_grid = trimesh.voxel.creation.local_voxelize(
        ground_truth_mesh, center_in_cube, voxel_size, grid_half_n)

### We copy the input grid to the corresponding position in the common voxel grid. 
### The input grid should be a subset of the new grid.
input_grid = np.zeros((int(max_grid_size), int(max_grid_size), int(max_grid_size)), dtype=bool)
input_grid[
    -min_bounds_in_grid[0]:Nx-min_bounds_in_grid[0],
    -min_bounds_in_grid[1]:Nx-min_bounds_in_grid[1],
    -min_bounds_in_grid[2]:Nx-min_bounds_in_grid[2]] = occ_grid

### Calculate IoU (Intersection over Union)
intersection = np.logical_and(input_grid, gt_voxel_grid.matrix).sum()
union = np.logical_or(input_grid, gt_voxel_grid.matrix).sum()
iou = intersection / union

### Calculate volumetric error
volumetric_error = (1 - iou) * union / gt_voxel_grid.volume
print(f"Volumetric Error: {volumetric_error}")
print(f"IoU: {iou}")

# ### Visualize the input grid and the ground truth mesh and grid
# import plotly.graph_objects as go
# grid_x, grid_y, grid_z = np.mgrid[
#     min_bounds[0]:max_bounds[0]:1j*max_grid_size,
#     min_bounds[1]:max_bounds[1]:1j*max_grid_size,
#     min_bounds[2]:max_bounds[2]:1j*max_grid_size]
# fig = go.Figure(data=[
#     go.Volume(
#         x=grid_x.flatten(),
#         y=grid_y.flatten(),
#         z=grid_z.flatten(),
#         value=input_grid.flatten(),
#         isomin=0,
#         isomax=1,
#         opacity=0.1, # needs to be small to see through all surfaces
#         surface_count=1, # needs to be a large number for good volume rendering
#         ),
# ])
# # Save the figure
# fig.write_image("voxel_grid_plot.png")
# # fig.write_html('/mnt/data0/minghz/repos/bundlenets/cnets-data-generation/input_grid.html')


from mayavi import mlab
from tvtk.util.ctf import ColorTransferFunction, PiecewiseFunction
input_grid = input_grid.astype(float) #* 1
gt_grid = gt_voxel_grid.matrix.astype(float) #* 2
# combined_grid = input_grid + gt_grid
print(f"{input_grid.shape=}, {gt_grid.shape=}, {input_grid.min()=}, {input_grid.max()=}, {gt_grid.min()=}, {gt_grid.max()=}")
input_field = mlab.pipeline.scalar_field(input_grid)
gt_field = mlab.pipeline.scalar_field(gt_grid)
# combined_field = mlab.pipeline.scalar_field(combined_grid)

input_volume = mlab.pipeline.volume(input_field) #, vmin=0, vmax=1
gt_volume = mlab.pipeline.volume(gt_field) #, vmin=0, vmax=1

# Create color transfer functions
ctf_1 = ColorTransferFunction()
ctf_1.add_rgb_point(0, 0, 0, 1)  # Map scalar value 0 to blue
ctf_1.add_rgb_point(1, 0, 0, 1)  # Map scalar value 1 to blue

ctf_2 = ColorTransferFunction()
ctf_2.add_rgb_point(0, 1, 0, 0)  # Map scalar value 0 to red
ctf_2.add_rgb_point(1, 1, 0, 0)  # Map scalar value 1 to red
input_volume._volume_property.set_color(ctf_1)
gt_volume._volume_property.set_color(ctf_2)

# Create opacity transfer functions
otf_1 = PiecewiseFunction()
otf_1.add_point(0, 0.0)  # Fully transparent at scalar value 0
otf_1.add_point(1, 0.01)  # 80% opacity at scalar value 1

otf_2 = PiecewiseFunction()
otf_2.add_point(0, 0.0)  # Fully transparent at scalar value 0
otf_2.add_point(1, 0.01)  # 60% opacity at scalar value 1
input_volume._volume_property.set_scalar_opacity(otf_1)
gt_volume._volume_property.set_scalar_opacity(otf_2)
mlab.show()