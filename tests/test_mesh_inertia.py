"""Some tests to get inertia tensors out of meshes."""

import numpy as np
import os
import os.path as op
import pdb
import sys
import trimesh

CNETS_DATA_GEN_DIR = op.dirname(op.dirname(op.abspath(__file__)))
if CNETS_DATA_GEN_DIR not in sys.path:
    sys.path.append(CNETS_DATA_GEN_DIR)

import file_utils


EARLY_TESTS = False
COMPUTE_AVERAGE_VYSICS_PARAMETERS = True

OBJECT_MASS = 0.37

FRICTION_SLIP_ANGLE_BOTTLE = np.deg2rad(26)
FRICTION_SLIP_ANGLE_HALF = np.deg2rad(15)
FRICTION_SLIP_ANGLE_MILK = np.deg2rad(9)
FRICTION_SLIP_ANGLE_CUBE = np.deg2rad(13)
FRICTION_SLIP_ANGLES = {'cube': FRICTION_SLIP_ANGLE_CUBE,
                        'bottle': FRICTION_SLIP_ANGLE_BOTTLE,
                        'half': FRICTION_SLIP_ANGLE_HALF,
                        'milk': FRICTION_SLIP_ANGLE_MILK}

# Since the surface was slightly different from the toss surface, can use the
# cube's "known" friction coefficient on the toss surface to scale the measured
# coefficients for all objects.
expected_cube_friction = 0.15
measured_cube_friction = np.tan(FRICTION_SLIP_ANGLE_CUBE)
friction_scaling = expected_cube_friction / measured_cube_friction


### Functions to compute inertia information from meshes assuming they are
### hollow shells with uniform thickness.
def point_mass_inertia_tensor(location, mass):
    return mass * (np.dot(location, location) * np.eye(3) - \
                   np.outer(location, location))

def compute_hollow_shell_inertia_tensor_about_origin(mesh, total_mass):
    inertia_tensor = np.zeros((3, 3))
    total_area = mesh.area

    # Iterate over each face of the mesh.
    for face, area in zip(mesh.faces, mesh.area_faces):
        centroid = np.average(mesh.vertices[face], axis=0)
        triangle_mass = (area/total_area) * total_mass

        # Incorporate this face's contribution to inertia and center of mass.
        inertia_tensor += point_mass_inertia_tensor(centroid, triangle_mass)

    return inertia_tensor

def compute_hollow_shell_com(mesh):
    area_times_location = np.zeros(3)
    total_area = mesh.area

    # Iterate over each face of the mesh.
    for face, area in zip(mesh.faces, mesh.area_faces):
        centroid = np.average(mesh.vertices[face], axis=0)
        area_times_location += area * centroid

    com = area_times_location / total_area

    return com


### Function to read a URDF file and extract out the inertia and friction.
def get_parameters_from_urdf_string(urdf_file, parameter_list_dict):
    with open(urdf_file, 'r') as f:
        urdf_string = f.read()

    ixx = float(urdf_string.split('ixx="')[-1].split('"')[0])
    ixy = float(urdf_string.split('ixy="')[-1].split('"')[0])
    ixz = float(urdf_string.split('ixz="')[-1].split('"')[0])
    iyy = float(urdf_string.split('iyy="')[-1].split('"')[0])
    iyz = float(urdf_string.split('iyz="')[-1].split('"')[0])
    izz = float(urdf_string.split('izz="')[-1].split('"')[0])

    com = urdf_string.split('inertial>')[1].split('origin xyz="')[-1].split(
        '"')[0]
    com_x = float(com.split(' ')[0])
    com_y = float(com.split(' ')[1])
    com_z = float(com.split(' ')[2])

    mu = float(urdf_string.split('drake:mu_static value="')[-1].split('"')[0])

    parameter_list_dict['ixx'].append(ixx)
    parameter_list_dict['ixy'].append(ixy)
    parameter_list_dict['ixz'].append(ixz)
    parameter_list_dict['iyy'].append(iyy)
    parameter_list_dict['iyz'].append(iyz)
    parameter_list_dict['izz'].append(izz)
    parameter_list_dict['com_x'].append(com_x)
    parameter_list_dict['com_y'].append(com_y)
    parameter_list_dict['com_z'].append(com_z)
    parameter_list_dict['mu'].append(mu)


################################################################################

# Compute the average Vysics-estimated inertial properties and friction
# coefficients.
if COMPUTE_AVERAGE_VYSICS_PARAMETERS:
    parameter_list_dict = {
        'ixx': [], 'ixy': [], 'ixz': [], 'iyy': [], 'iyz': [], 'izz': [],
        'com_x': [], 'com_y': [], 'com_z': [], 'mu': []
    }

    eval_dir = file_utils.evaluation_dir()
    for subdir in os.listdir(eval_dir):
        if not subdir.endswith('02_02_2'):
            continue

        print(f'Processing {subdir}...', end=' ')
        urdf_file = op.join(eval_dir, subdir, 'bsdf_mesh_pll_params.urdf')
        if not op.exists(urdf_file):
            print('no URDF file found.')
            continue

        get_parameters_from_urdf_string(urdf_file, parameter_list_dict)
        print('done.')

    # Compute the average values.
    print(f'\n')
    for key in parameter_list_dict:
        print(f'Average {key}:  {np.mean(parameter_list_dict[key])}')

    pdb.set_trace()
    exit()


# Compute the inertial properties of some objects from their meshes.  Also
# compute the coefficient of friction based on the slip angle.
for object in ['bottle', 'half', 'milk', 'cube']:
    _urdf_path, obj_path = file_utils.ground_truth_object_urdf_obj_filepaths(
        object, body_t = object=='cube')

    # Load the mesh and ensure it's high-resolution (the inertia estimate is
    # more accurate with a finer mesh).
    mesh = trimesh.load_mesh(obj_path)
    finer_mesh = mesh.subdivide_to_size(0.005).process(validate=True)

    # First compute the center of mass.
    com = compute_hollow_shell_com(finer_mesh)

    # Want the inertia tensor about the center of mass, not the origin.  Thus,
    # translate the mesh by the opposite of the center of mass before computing
    # the inertia tensor.
    transform = np.eye(4)
    transform[:3, 3] = -com
    finer_mesh.apply_transform(transform)
    assert np.allclose(compute_hollow_shell_com(finer_mesh), np.zeros(3))

    inertia = compute_hollow_shell_inertia_tensor_about_origin(
        finer_mesh, OBJECT_MASS)
    
    # Compute the coefficient of friction based on the slip angle.
    mu = np.tan(FRICTION_SLIP_ANGLES[object]) * friction_scaling
    
    # Print summary.
    print(f'Object:  {object}')
    print(f'Coefficient of friction:  {mu}')
    print(f'Center of mass:  {com}')
    print(f'Inertia tensor about center of mass:\n{inertia}\n')
    


# Hide some early tests to sanity check the correctness of the above
# inertia tensor approximations.
if EARLY_TESTS:
    _urdf_path, obj_path = file_utils.ground_truth_object_urdf_obj_filepaths(
        'cube')

    # Load the mesh
    mesh = trimesh.load_mesh(obj_path)
    mesh_inertia_properties = mesh.mass_properties
    mesh_mass = mesh_inertia_properties.mass
    mesh_inertia = mesh_inertia_properties.inertia

    mass = 0.37
    cube_length = 0.1048

    # Use true object mass, and get the mesh's inertia tensor to reflect that.
    inertia = mesh_inertia / mesh_mass * mass

    # Test compute the inertia of a solid cube.
    solid_cube_inertia_Ixx = (1/6) * mass * cube_length**2

    print(f'mesh inertia == solid cube inertia:  ' + \
        f'{np.isclose(inertia[0,0], solid_cube_inertia_Ixx)}')



    finer_mesh = mesh.subdivide_to_size(0.01).process(validate=True)
    mesh_hollow_inertia = compute_hollow_shell_inertia_tensor_about_origin(
        mesh, mass)
    better_mesh_hollow_inertia = compute_hollow_shell_inertia_tensor_about_origin(
        finer_mesh, mass)

    print(f'mesh_hollow_inertia:\n{mesh_hollow_inertia}\n')
    print(f'better_mesh_hollow_inertia:\n{better_mesh_hollow_inertia}\n')
    print(f'expected diagonal values:  0.0013545749333333335\n')


    # Test with a sphere.  This seems to work.
    radius = 0.1
    sphere = trimesh.creation.icosphere(radius=radius, subdivisions=2)
    finer_sphere = sphere.subdivide_to_size(0.005).process(validate=True)

    sphere_hollow_inertia = compute_hollow_shell_inertia_tensor_about_origin(
        sphere, mass)
    better_sphere_hollow_inertia = \
        compute_hollow_shell_inertia_tensor_about_origin(finer_sphere, mass)

    print(f'sphere_hollow_inertia:\n{sphere_hollow_inertia}\n')
    print(f'better_sphere_hollow_inertia:\n{better_sphere_hollow_inertia}\n')

    expected_sphere_diagonal = (2/3) * mass * radius**2
    print(f'expected_sphere_diagonals: {expected_sphere_diagonal}')


    # Test with a thin rectangle.
    half_length = 0.1   # x-direction
    half_width = 0.2    # y-direction

    xs = np.linspace(-half_length, half_length, 1000)
    ys = np.linspace(-half_width, half_width, 1000)

    rect_inertia_tensor = np.zeros((3, 3))
    partial_mass = mass / (len(xs) * len(ys))
    for x in xs:
        for y in ys:
            location = np.array([x, y, 0])
            rect_inertia_tensor += point_mass_inertia_tensor(
                location, partial_mass)

    print(f'rect_inertia_tensor:\n{rect_inertia_tensor}\n')

    expected_rectangle_Ixx = (1/12) * mass * (2*half_width)**2
    expected_rectangle_Iyy = (1/12) * mass * (2*half_length)**2
    expected_rectangle_Izz = (1/12) * mass * ((2*half_length)**2 + \
                                              (2*half_width)**2)
    expected_rectangle_inertia = np.diag(
        [expected_rectangle_Ixx, expected_rectangle_Iyy, expected_rectangle_Izz
         ])
    print(f'expected_rectangle_inertia:\n{expected_rectangle_inertia}\n')



pdb.set_trace()


