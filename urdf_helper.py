import numpy as np
import matplotlib.pyplot as plt
import trimesh
from scipy.spatial import ConvexHull
from itertools import combinations
import open3d as o3d
import argparse

from experiments import toOpen3dCloud

def ensure_vertex_normals(mesh):
    # Accessing vertex_normals will compute them if they aren't already present
    mesh.vertex_normals = (mesh.vertex_normals.T / np.linalg.norm(mesh.vertex_normals, axis=1)).T
    return mesh

def simplify_mesh(input_path, output_path, fraction):
    # Load the mesh from the given .obj file
    mesh = trimesh.load_mesh(input_path)

    # Simplify the mesh
    mesh_simplified = mesh.simplify_quadratic_decimation(int(fraction * len(mesh.faces)))
    # mesh_simplified = mesh.simplify_quadratic_decimation(10)
    # Export the simplified mesh to an .obj file
    mesh_simplified.export(output_path)
    print(f'Similified to {len(mesh_simplified.faces)} faces')

def simplify_to_cube(input_path, output_path):
    # 1. Load the dense mesh
    mesh = trimesh.load_mesh(input_path)
    
    # 2. Compute the axis-aligned bounding box of the mesh
    min_bound = np.min(mesh.vertices, axis=0)
    max_bound = np.max(mesh.vertices, axis=0)
    
    # 3. Determine the side length of the cube using the average dimension of the AABB
    dimensions = max_bound - min_bound
    side_length = np.mean(dimensions)
    half_length = side_length / 2
    
    # 4. Create cube vertices with equal side lengths centered around the mesh's centroid
    centroid = mesh.centroid
    cube_vertices = [
        [centroid[0] - half_length, centroid[1] - half_length, centroid[2] - half_length],
        [centroid[0] + half_length, centroid[1] - half_length, centroid[2] - half_length],
        [centroid[0] - half_length, centroid[1] + half_length, centroid[2] - half_length],
        [centroid[0] + half_length, centroid[1] + half_length, centroid[2] - half_length],
        [centroid[0] - half_length, centroid[1] + half_length, centroid[2] + half_length],
        [centroid[0] + half_length, centroid[1] + half_length, centroid[2] + half_length],
        [centroid[0] - half_length, centroid[1] - half_length, centroid[2] + half_length],
        [centroid[0] + half_length, centroid[1] - half_length, centroid[2] + half_length]
    ]

    # Construct faces for a cube
    cube_faces = [
        [0, 1, 2], [2, 1, 3],
        [4, 5, 6], [6, 5, 7],
        [0, 1, 6], [6, 1, 7],
        [2, 3, 4], [4, 3, 5],
        [1, 3, 5], [1, 5, 7],
        [0, 2, 4], [0, 4, 6]
    ]

    # 5. Create a mesh with the cube vertices and faces
    cube = trimesh.Trimesh(vertices=cube_vertices, faces=cube_faces)

    # 6. Save the cube mesh
    cube.export(output_path)

def add_normals_to_obj(input_path, output_path):
    # Load the mesh from the given file
    mesh = trimesh.load(input_path)
    # Compute the vertex normals
    mesh_with_normals = ensure_vertex_normals(mesh)
    # Save the mesh with normals back to .obj format
    print(mesh_with_normals.vertices.dtype)
    mesh_with_normals.export(output_path, file_type="obj")
    print(f'mesh exported to {output_path}')

def maximal_volume_subset(mesh_path, vertex_count=20):
    mesh = trimesh.load_mesh(mesh_path)
    vertices = mesh.vertices

    max_volume = -np.inf
    best_subset = None

    # Iterate over all combinations of vertices
    for subset in combinations(vertices, vertex_count):
        hull = ConvexHull(np.array(subset))
        if hull.volume > max_volume:
            max_volume = hull.volume
            best_subset = subset

    return np.array(best_subset)

def create_max_volume_obj(input_path, output_path, vertex_count=20, target_vertex_count=None, target_face_count=None):
    # Get the best subset of vertices
    subset = maximal_volume_subset(input_path, vertex_count=vertex_count)
    
    # Create a convex hull from the subset
    hull = ConvexHull(subset)

    # Convert the convex hull to a Trimesh object
    mesh = trimesh.Trimesh(vertices=hull.points, faces=hull.simplices)
    while len(mesh.vertices) > target_vertex_count or len(mesh.faces) > target_face_count:
        mesh = mesh.simplify_quadratic_decimation(int(0.98 * len(mesh.faces)))

        # Safety check to avoid infinite loops
        if len(mesh.faces) <= 1:
            print("Cannot achieve target vertex and face count while preserving connectivity.")
            break
    # Export the new mesh to an .obj file
    mesh.export(output_path)
    print(f'max-vol mesh exported to {output_path}')

def mesh_to_cube(file_path, output_path):
    # Load the mesh using trimesh
    mesh = trimesh.load(file_path)

    # Determine the maximal magnitude among all vertices of the mesh
    max_magnitude = np.max(np.abs(mesh.vertices)) * 0.05

    # Define vertices for the cube
    cube_vertices = [
        [-max_magnitude, -max_magnitude, -max_magnitude],
        [ max_magnitude, -max_magnitude, -max_magnitude],
        [ max_magnitude,  max_magnitude, -max_magnitude],
        [-max_magnitude,  max_magnitude, -max_magnitude],
        [-max_magnitude, -max_magnitude,  max_magnitude],
        [ max_magnitude, -max_magnitude,  max_magnitude],
        [ max_magnitude,  max_magnitude,  max_magnitude],
        [-max_magnitude,  max_magnitude,  max_magnitude]
    ]

    # Faces for the cube
    cube_faces = [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [0, 1, 5, 4],
        [1, 2, 6, 5],
        [2, 3, 7, 6],
        [3, 0, 4, 7]
    ]
    cube = trimesh.Trimesh(vertices=cube_vertices, faces=cube_faces)
    cube.export(output_path)

def shrink_mesh_file(mesh_file_path, divisor, output_file_path=None):
    # Read the input mesh file
    with open(mesh_file_path, 'r') as f:
        lines = f.readlines()

    # Placeholder for the shrunken lines
    shrunken_lines = []

    for line in lines:
        # Split the line into parts
        parts = line.split()

        # Check if this line represents a vertex
        if len(parts) > 0 and parts[0] == "v":
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            x, y, z = x / divisor, y / divisor, z / divisor
            shrunken_lines.append(f"v {x} {y} {z}\n")
        else:
            shrunken_lines.append(line)

    # Write the modified mesh to the output file
    with open(output_file_path, 'w') as f:
        f.writelines(shrunken_lines)

    print(f"Shrunken mesh saved to {output_file_path}")

def add_noise_to_mesh(input, output, noise_factor=0.1):
    mesh = trimesh.load_mesh(input)
    noise = (np.random.rand(*mesh.vertices.shape) - 0.5) * noise_factor
    mesh.vertices += noise
    mesh.export(output)
    print(f'Noisy mesh exported to {output}')

def sample_points_from_mesh(input_obj_path, output_obj_path, num_points=1000, triangle_size=0.005):
    mesh = trimesh.load_mesh(input_obj_path)
    points = mesh.sample(num_points)
    vertices = []
    faces = []
    for point in points:
        v0 = point
        v1 = point + [triangle_size, 0, 0]
        v2 = point + [0, triangle_size, 0]
        vertices.extend([v0, v1, v2])
        n = len(vertices) - 3
        faces.append([n, n+1, n+2])
    point_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    point_mesh.export(output_obj_path, file_type='obj')

def sample_pcd_from_mesh(input_obj_path, output_ply_path, num_points=1000):
    mesh = trimesh.load(input_obj_path, force='mesh')
    if np.asarray(mesh.triangles).shape[0] == 0:
        raise ValueError("The mesh doesn't contain any triangles.")
    pts = mesh.sample(99999)
    pcd = toOpen3dCloud(np.array(pts))
    o3d.io.write_point_cloud(output_ply_path,pcd,write_ascii=True)
    print(f'pcd exported to {output_ply_path}')
    
def pcd_to_mesh(input_ply_path, output_obj_path):
    pcd = o3d.io.read_point_cloud(input_ply_path)
    pcd.estimate_normals()
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = 1.5 * avg_dist   
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd,
            o3d.utility.DoubleVector([radius, radius * 2]))

    # create the triangular mesh with the vertices and faces from open3d
    tri_mesh = trimesh.Trimesh(np.asarray(mesh.vertices), np.asarray(mesh.triangles),
                            vertex_normals=np.asarray(mesh.vertex_normals))

    # trimesh.convex.is_convex(tri_mesh)
    tri_mesh.export(output_obj_path, file_type='obj')

'''Process the 3D scanned napkin mesh
'''
def scale_and_center(input_file, output_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()

    vertices = [list(map(float, line.split()[1:])) for line in lines if line.startswith('v ')]
    vts = [line for line in lines if line.startswith('vt ')]
    faces = [line for line in lines if line.startswith('f ')]

    min_x = min(v[0] for v in vertices)
    max_x = max(v[0] for v in vertices)
    min_y = min(v[1] for v in vertices)
    max_y = max(v[1] for v in vertices)
    min_z = min(v[2] for v in vertices)
    max_z = max(v[2] for v in vertices)
    scale_x = 0.2286 / (max_x - min_x)
    scale_y = 0.089 / (max_y - min_y)
    scale_z = 0.12 / (max_z - min_z)
    uniform_scale = min(scale_x, scale_y, scale_z)
    centroid_x = (max_x + min_x) / 2
    centroid_y = (max_y + min_y) / 2
    centroid_z = (max_z + min_z) / 2
    scaled_and_centered = []
    for v in vertices:
        x = (v[0] - centroid_x) * uniform_scale
        y = (v[1] - centroid_y) * uniform_scale
        z = (v[2] - centroid_z) * uniform_scale
        scaled_and_centered.append([x, y, z])

    with open(output_file, 'w') as f:
        for v in scaled_and_centered:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for vt in vts:
            f.write(vt)
        for face in faces:
            f.write(face)
    print('File exported')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--filename",
        type=str,
        required=True,
    )
    args = parser.parse_args()
    filename = args.filename
    
    # get inertia 
    # mesh_file = './mesh_convex_hull.txt'
    # print(get_inertia(mesh_file))
    
    original_mesh = f'./assets/{filename}.obj'
    simplified_mesh = f'./assets/{filename}_rescale_simplified.obj'
    output_path = f'./assets/{filename}_optimized.obj'
    normal_mesh = f'./assets/{filename}_with_normals.obj'
    rescale_mesh = f'./assets/{filename}_rescale.obj'
    alt_simplified_mesh = f'./assets/{filename}_rescale_simplified_alt.obj'
    sampled_mesh = f'./assets/{filename}_sampled.obj'
    sampled_pcd = f'./assets/{filename}_sampled.ply'
    # simplify_mesh(original_mesh, simplified_mesh, 0.01)
    # create_max_volume_obj(simplified_mesh, output_path, vertex_count=10, target_vertex_count=8, target_face_count=6)
    # add_normals_to_obj(output_path, normal_mesh)
    # mesh_to_cube(original_mesh, output_path)
    
    # shrink_mesh_file(original_mesh, 8.7, rescale_mesh)
    # simplify_mesh(rescale_mesh, simplified_mesh, 0.5)
    # add_normals_to_obj(simplified_mesh, normal_mesh)
    
    # noisy_mesh = f'./assets/{filename}_noise.obj'
    # add_noise_to_mesh(original_mesh, noisy_mesh)
    
    # sample_points_from_mesh(original_mesh, sampled_mesh, num_points=5000, triangle_size=0.005)
    # sample_pcd_from_mesh(original_mesh, sampled_pcd)
    # pcd_to_mesh(sampled_pcd, sampled_mesh)
    scan_file = f'./assets/gt_napkin.obj'
    output = f'./assets/gt_napkin_scale.obj'
    scale_and_center(scan_file, output)