import numpy as np

from pydrake.all import MultibodyPlant, Parser, DiagramBuilder, RigidTransform

import file_utils


"""Convert Franka joint angles to the 3D world position of the end effector's
center.  Assumes the Franka base is at the world origin, and uses the Franka and
end effector geometry defined in assets/franka_with_ee.urdf.  This uses Drake to
build a plant with the Franka, set the plant's joint angles, then query the
position of the end effector tip's origin."""
def convert_franka_joints_to_ee_positions(
        joint_angles: np.ndarray, joint_names: list,
        as_homogeneous_transform_with_rotation: bool = False):
    # Build a Drake plant with the Franka at the world origin.
    builder = DiagramBuilder()
    plant = MultibodyPlant(time_step=0.0)
    parser = Parser(plant)
    parser.AddModels(file_utils.franka_filepath())
    plant.WeldFrames(
        plant.world_frame(), plant.GetFrameByName("panda_link0"),
        RigidTransform()
    )
    plant.Finalize()
    builder.AddSystem(plant)
    builder.Build()
    context = plant.CreateDefaultContext()

    # Get the indices of the joint names that correspond to the Franka joints.
    sim_joint_names = plant.GetPositionNames()
    sim_joint_indices = -1 * np.ones((len(sim_joint_names),), dtype=int)
    for sim_joint_index, sim_joint_name in enumerate(sim_joint_names):
        for recorded_joint_index, recorded_joint_name in enumerate(joint_names):
            if recorded_joint_name in sim_joint_name:
                sim_joint_indices[sim_joint_index] = recorded_joint_index
                break
    assert np.all(sim_joint_indices >= 0), 'Failed to find all joint ' + \
        f'indices: {sim_joint_names=} vs. {joint_names=}.'

    # Prepare to store the end effector positions (or homogeneous
    # transformation).
    if as_homogeneous_transform_with_rotation:
        T_WEs = np.eye(4).reshape(1, 4, 4).repeat(joint_angles.shape[0], axis=0)
        to_return = T_WEs
    else:
        ee_positions = np.zeros((joint_angles.shape[0], 3))
        to_return = ee_positions

    # Iterate over all the joint angles and read the corresponding EE position.
    for i, joint_angle in enumerate(joint_angles):
        plant.SetPositions(context, joint_angle[sim_joint_indices])
        ee_pose = plant.EvalBodyPoseInWorld(
            context, plant.GetBodyByName("end_effector_tip"))

        if as_homogeneous_transform_with_rotation:
            T_WEs[i, :3, :3] = ee_pose.rotation().matrix()
            T_WEs[i, :3, 3] = ee_pose.translation()
        else:
            ee_positions[i] = ee_pose.translation()

    return to_return
