"""Script to gather evaluation results.  Writes a few yaml files:

Should get BundleSDF ID 02 for BSDF-PLL.  BundleSDF ID 00 for BSDF-only. PLL ID
00 for PLL-only.  BundleSDF ID 03 for BSDF-PLL with online NeRF, though these
only have results for tosses 1-3.

    bsdf_pll:
        tagged_objects:
            tagged_object_1:
                trained_on_toss_1:
                    geometry_metrics:
                        convex_hull:
                            volume_error:
                            chamfer_distance:
                            f_score:
                        full_geometry:
                            volume_error:
                            iou:
                            chamfer_distance:
                            f_score:
                        hull_to_full:
                            chamfer_distance:
                            f_score:
                    tracking_metrics:
                        toss:               <-- averaged over all tosses
                            against_bundlesdf:
                                metric_1:
                                    mean: XXX
                                    mean_auc: XXX
                                metric_2: ...
                                metric_3: ...
                                ...
                            against_tagslam:
                                metric_1:
                                    mean: XXX
                                    mean_auc: XXX
                                metric_2: ...
                                metric_3: ...
                                ...
                        full: ...
                    dynamics_rollout_metrics:
                        training_tosses:
                            against_bundlesdf:
                                metric_1:  <-- all of these from training tosses
                                    mean: XXX
                                    mean_auc: XXX
                                metric_2: ...
                                metric_3: ...
                                ...
                            against_tagslam:
                                metric_1:
                                    mean: XXX
                                    mean_auc: XXX
                                metric_2: ...
                                metric_3: ...
                                ...
                        unseen_tosses:
                            against_tagslam: <-- no against_bundlesdf for unseen
                                metric_1:
                                    mean: XXX
                                    mean_auc: XXX
                                metric_2: ...
                                metric_3: ...
                                ...
                        all_tosses: ...
                    dynamics_single_step_metrics:
                        against_bundlesdf: ...
                        against_tagslam: ...
                trained_on_toss_1-2: ...
                trained_on_toss_1-3: ...
                ...
            tagged_object_2: ...
            tagged_object_3: ...
            ...
            all_tagged_objects: ...
        tagless_objects:
            tagless_object_1:
                trained_on_toss_1:
                    geometry_metrics: ...
                    tracking_metrics:
                        toss:               <-- averaged over all tosses
                            against_bundlesdf:
                                metric_1:
                                    mean: XXX
                                    mean_auc: XXX
                                metric_2: ...
                                metric_3: ...
                                ...
                        full: ...
                    dynamics_rollout_metrics:
                        training_tosses:   <-- no other keys here
                            metric_1:      <-- all of these from training tosses
                                mean: XXX
                                mean_auc: XXX
                            metric_2: ...
                            metric_3: ...
                            ...
                    dynamics_single_step_metrics: ...
                trained_on_toss_1-2: ...
                trained_on_toss_1-3: ...
            tagless_object_2: ...
            tagless_object_3: ...
            ...
            all_tagless_objects: ...
        all_objects: ...    <-- same contents as e.g. tagged_object_1
    nerf_on: ...            <-- same as bsdf_pll but with online NeRF
    bsdf_only: ...          <-- same as bsdf_pll but without dynamics sections
    pll_only: ...           <-- same as bsdf_pll
    pll_tagslam: ...        <-- same as bsdf_pll
    pll_blind: ...          <-- same as bsdf_pll

Outdated experiments:
    - PLL with TagSLAM and no vision supervision:  PLL 00_0 (or PLL 06_0)

Experiments to collect:
    - bsdf_pll:  bsdf 02_2
    - nerf_on:  bsdf 03_2
    - bsdf_only:  bsdf 00_1
    - PLL with vision supervision:  PLL 00_1
    - PLL with size vision supervision only:  PLL 04_1 (or PLL 05_1)
    - PLL with BundleSDF tracking and no vision supervision:  PLL 07_1
    - PLL with TagSLAM and no vision supervision:  PLL 09_0
"""

import click
import numpy as np
import os
import os.path as op
import pdb
from scipy import stats

from matplotlib import rc, rcParams
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, NullFormatter, ScalarFormatter

import eval_utils, file_utils, robot_dynamics_predictions

from file_utils import OBJECTS_WITH_GT_URDF


# Some settings on the plot generation.
rc('legend', fontsize=30)
plt.rc('axes', titlesize=40)    # fontsize of the axes title
plt.rc('axes', labelsize=40)    # fontsize of the x and y labels



CM_LENGTH_SCALES_BY_OBJ = {
    'croc': 100 * 0.1632026,
    'half': 100 * 0.23557212,
    'napkin': 100 * 0.262057693,
    'styrofoam': 100 * 0.25444198,
    'crushedcan': 100 * 0.122251,
    'cardboard': 100 * 0.2252869,
    'bakingbox': 100 * 0.291327,
    'oatly': 100 * 0.24767799999999998,
    'pinkcan': 100 * 0.13025200000000003,
    'milk': 100 * 0.25997706,
    'gallon': 100 * 0.2555541,
    'greencan': 100 * 0.13025200000000003,
    'egg': 100 * 0.31548529999999997,
    'stapler': 100 * 0.1417093,
    'bottle': 100 * 0.27827598600000003,
    'cube': 100 * 0.1048,
    'toblerone': 100 * 0.2118,
}


ERROR_LABELS = {
    'position_error': 'Position Error [cm]',
    'rotation_error': 'Rotation Error [deg]',
    'add_error': 'ADD Error [cm]',
    'adds_error': 'ADD-S Error [cm]',
    'penetration_learned_geom_tagslam_traj': \
        'Learned Geometry Penetration [cm]',
    'penetration_true_geom_estimated_traj': \
        'True Geometry Penetration [cm]',
    'penetration_learned_geom_estimated_traj': \
        'Learned Geometry Penetration [cm]',
    'penetration_true_geom_predicted_traj': \
        'Predicted Penetration [cm]',
    'volume_error': 'Relative Volume Error',
    'chamfer_distance': 'Chamfer Distance [cm]',
    'iou': 'IoU',
    #'f_score': TODO this isn't implemented so exclude from dictionary
    'pos_rollout_error_mean': 'Position Error [cm]',
    'rot_rollout_error_mean': 'Rotation Error [deg]',
    'time_before_bad_pos': 'Time [s]',
    'time_before_bad_rot': 'Time [s]',
    'contact_activation_iou': '',
}
AUC_LABELS = {
    'position_error': \
        f'Position Error AUC (<{eval_utils.POSITION_AUC_THRESHOLD}m)',
    'rotation_error': \
        f'Rotation Error AUC (<{eval_utils.ORIENTATION_AUC_THRESHOLD/np.pi*180} deg)',
    'add_error': f'ADD AUC (<{eval_utils.POSITION_AUC_THRESHOLD})',
    'adds_error': f'ADD-S AUC (<{eval_utils.POSITION_AUC_THRESHOLD})',
    'penetration_learned_geom_tagslam_traj': \
        f'Learned Geometry Penetration (<{eval_utils.POSITION_AUC_THRESHOLD}m)',
    'penetration_true_geom_estimated_traj': \
        f'True Geometry Penetration (<{eval_utils.POSITION_AUC_THRESHOLD}m)',
    'penetration_learned_geom_estimated_traj': \
        f'Learned Geometry Penetration (<{eval_utils.POSITION_AUC_THRESHOLD}m)',
    'penetration_true_geom_predicted_traj': \
        f'Predicted Penetration (<{eval_utils.POSITION_AUC_THRESHOLD}m)',
}
NUM_TOSSES_LABEL = 'Number of Training Tosses'

METRIC_SCALING = {
    'auc': 100,                                     # [%]
    'position_error': 100,                          # [cm]
    'rotation_error': 180/np.pi,                    # [deg]
    'add_error': 100,                               # [cm]
    'adds_error': 100,                              # [cm]
    'penetration_learned_geom_tagslam_traj': 100,   # [cm]
    'penetration_true_geom_estimated_traj': 100,    # [cm]
    'penetration_learned_geom_estimated_traj': 100, # [cm]
    'penetration_true_geom_predicted_traj': 100,    # [cm]
    'volume_error': 1,                              # weird fractional units
    'chamfer_distance': 100,                        # [cm]
    'iou': 1,                                       # [%]
    #'f_score': TODO this isn't implemented so exclude from dictionary
    'pos_rollout_error_mean': 100,
    'rot_rollout_error_mean': 180/np.pi,
    'time_before_bad_pos': 1,
    'time_before_bad_rot': 1,
    'contact_activation_iou': 1,
}

# The following are t values for 95% confidence interval.
T_SCORE_PER_DOF = {1: 12.71, 2: 4.303, 3: 3.182, 4: 2.776,
                   5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
                   9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
                   13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120,
                   17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
                   21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064,
                   25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048,
                   29: 2.045, 30: 2.042}
for i in range(31, 200):
    T_SCORE_PER_DOF[i] = 1.960

BSDF_PLL = 'bsdf_pll'
NERF_ON = 'nerf_on'
BSDF_ONLY = 'bsdf_only'
BSDF_OCTREE = 'bsdf_octree'
PLL_VISION = 'pll_vision'
PLL_SIZE = 'pll_size'
PLL_BLIND_B = 'pll_blind_b'
PLL_BLIND_T = 'pll_blind_t'
BSDF_CONVEX = 'bsdf_convex'
BSDF_CONVEX_HULL = 'bsdf_convex_hull'
BSDF_PLL_CONVEX = 'bsdf_pll_convex'
GT = 'gt'
BSDF_CONVEX_OCC = 'bsdf_convex_occ'
BSDF_PLL_CONVEX_OCC = 'bsdf_pll_convex_occ'
BSDF_PLL_ROBOTOCC = 'bsdf_pll_robotocc'
BSDF_ROBOTOCC = 'bsdf_robotocc'

COLORS = {
    BSDF_PLL: '#7030a0',  #'#398537',  #'#8c59b3',
    NERF_ON: '#ff0000',  #'#92668d',
    BSDF_ONLY: '#f9a602',  #'#a1b8e1',  #'#8eaadb',  #'#4472c4',  #'cd5b45',
    BSDF_OCTREE: '#0fff00',  #'#ff0000',
    PLL_VISION: '#00ff00',  #'#833785',
    PLL_SIZE: '#0000ff',  #'#4a0042',
    PLL_BLIND_B: '#ffff00',  #'#92668d',
    PLL_BLIND_T: '#ff00ff',  #'#95001a',
    BSDF_CONVEX: '#00ffff',  #'#ff0000',
    BSDF_CONVEX_HULL: '#8eaadb',
    BSDF_PLL_CONVEX: '#f000ff',
    BSDF_CONVEX_OCC: '#00ffff',
    BSDF_PLL_CONVEX_OCC: '#f000ff',
    BSDF_PLL_ROBOTOCC: '#7030a0',
    BSDF_ROBOTOCC: '#0f9ed5',
    GT: '#000000',
}

LABELS = {
    BSDF_PLL: 'Vysics',
    NERF_ON: 'BundleSDF-PLL NeRF Online',
    BSDF_ONLY: 'BundleSDF',
    BSDF_OCTREE: 'BundleSDF Octree Init',
    PLL_VISION: 'PLL with Vision Supervision',
    PLL_SIZE: 'PLL with Vision-Supervised Size Only',
    PLL_BLIND_B: 'Blind PLL on Vision-Based Tracking',
    PLL_BLIND_T: 'Blind PLL on Fiducial-Based Tracking',
    BSDF_CONVEX: 'BundleSDF Convex Loss',
    BSDF_CONVEX_HULL: 'BundleSDF Convex Hull',
    BSDF_PLL_CONVEX: 'Vysics Convex Loss',
    GT: 'Using Ground Truth URDF',
    BSDF_CONVEX_OCC: 'BundleSDF Convex Loss (Occluded)',
    BSDF_PLL_CONVEX_OCC: 'Vysics Convex Loss (Occluded)',
    BSDF_PLL_ROBOTOCC: 'Vysics',
    BSDF_ROBOTOCC: 'BundleSDF',
}

BSDF_PLL_COLOR = '#7030a0'  #'#398537'  #'#8c59b3'
NERF_ON_COLOR = '#ff0000'  #'#92668d'
BSDF_ONLY_COLOR = '#f9a602'  #'#a1b8e1'  #'#8eaadb'  #'#4472c4'  #'cd5b45'
PLL_VISION_COLOR = '#00ff00'  #'#833785'
PLL_SIZE_COLOR = '#0000ff'  #'#4a0042'
PLL_BLIND_B_COLOR = '#ffff00'  #'#92668d'
PLL_BLIND_T_COLOR = '#ff00ff'  #'#95001a'
BSDF_CONVEX_COLOR = '#00ffff'  #'#ff0000'
BSDF_CONVEX_HULL_COLOR = '#8eaadb'
BSDF_PLL_CONVEX_COLOR = '#f000ff'
BSDF_PLL_ROBOTOCC_COLOR = '#f000ff'
BSDF_ROBOTOCC_COLOR = '#00ffff'

BSDF_PLL_LABEL = 'Vysics'
NERF_ON_LABEL = 'BundleSDF-PLL NeRF Online'
BSDF_ONLY_LABEL = 'BundleSDF'  # [1]'
PLL_VISION_LABEL = 'PLL with Vision Supervision'
PLL_SIZE_LABEL = 'PLL with Vision-Supervised Size Only'
PLL_BLIND_B_LABEL = 'Blind PLL on Vision-Based Tracking'
PLL_BLIND_T_LABEL = 'Blind PLL on Fiducial-Based Tracking'
BSDF_CONVEX_LABEL = 'BundleSDF Convex Loss'
BSDF_CONVEX_HULL_LABEL = 'BundleSDF Convex Hull'
BSDF_PLL_CONVEX_LABEL = 'Vysics Convex Loss'
GT_LABEL = 'Using Ground Truth URDF'
BSDF_PLL_ROBOTOCC_LABEL = 'Vysics'
BSDF_ROBOTOCC_LABEL = 'BundleSDF'

LINEWIDTH = 5
MARKERSIZE = 100


PLOTS_TO_PRINT = [
    'all_tagged_objs_penetration_true_geom_predicted_traj_tagslam_v_data_unseen_tosses_dynamics_rollout_metrics',
    'all_tagged_objs_adds_error_tagslam_v_data_unseen_tosses_dynamics_rollout_metrics',
    'all_tagged_objs_add_error_tagslam_v_data_unseen_tosses_dynamics_rollout_metrics',
    'all_tagged_objs_add_error_v_data_full',
    'all_tagged_objs_adds_error_v_data_full',
]
PLOTS_TO_PRINT += [
    f'{obj}_add_error_for_gt_comp_unseen_tosses_dynamics_rollout_metrics' for \
        obj in OBJECTS_WITH_GT_URDF]
PLOTS_TO_PRINT += [
    f'{obj}_penetration_true_geom_predicted_traj_for_gt_comp_unseen_tosses_dynamics_rollout_metrics' \
        for obj in OBJECTS_WITH_GT_URDF]


def welchs_t_test(samples_1: list, samples_2: list):
    # Scipy's stats.ttest_ind is Welch's t-test, so no need to manually compute.
    t_stat, p_value = stats.ttest_ind(samples_1, samples_2, equal_var = False)

    return t_stat, p_value

def xs_and_ys_to_x_mean_lower_upper(mixed_xs, mixed_ys):
    mixed_xs = np.array(mixed_xs)
    mixed_ys = np.array(mixed_ys)

    # Get the unique xs.  This also puts them in increasing order.
    xs = np.unique(mixed_xs).tolist()

    # Combine the ys.
    ys, lowers, uppers = [], [], []
    for x in xs:
        idx = np.where(mixed_xs == x)[0]
        mean, lower, upper = set_of_vals_to_t_confidence_interval(mixed_ys[idx])

        ys.append(mean)
        lowers.append(lower)
        uppers.append(upper)

    return xs, ys, lowers, uppers

def set_of_vals_to_t_confidence_interval(ys):
    if len(ys) <= 0:
        return None, None, None
    elif len(ys) == 1:
        return ys[0], None, None

    dof = len(ys) - 1

    mean = np.mean(ys)
    lower = mean - T_SCORE_PER_DOF[dof]*np.std(ys)/np.sqrt(dof+1)
    upper = mean + T_SCORE_PER_DOF[dof]*np.std(ys)/np.sqrt(dof+1)

    return mean, lower, upper

def compute_auc_by_metric(errors_over_traj, metric: str):
    if metric in ['position_error', 'add_error', 'adds_error']:
        return eval_utils.compute_position_auc_error(errors_over_traj)
    elif metric == 'rotation_error':
        return eval_utils.compute_orientation_auc_error(errors_over_traj)
    elif 'penetration' in metric:
        return eval_utils.compute_penetration_auc_error(errors_over_traj)
    raise ValueError(f'Unknown metric: {metric}.')

def add_auc_to_results(results):
    """Add AUC to the results."""
    # Iterate over every metric category, e.g. dynamics_rollout_metrics.
    for _category, subresults in results.items():
        # Skip if the category doesn't have tracking-related metrics.
        if ('against_bundlesdf' not in subresults.keys()) and \
            ('against_tagslam' not in subresults.keys()):
            continue

        # Iterate over every comparison against, e.g. against_bundlesdf.
        for _against, subsubresults in subresults.items():

            # Iterate through every metric, e.g. position_error.
            for metric, trajectory_results in subsubresults.items():

                # Iterate through every trajectory, e.g. toss_1.
                for _traj_name, reported_errors in trajectory_results.items():

                    # Compute AUC for the metric and trajectory.
                    errors_over_traj = reported_errors['traj']
                    reported_errors['auc'] = None if errors_over_traj is None \
                        else compute_auc_by_metric(errors_over_traj, metric)

def recursive_dict_add(d, val, keys: list):
    """Add a value to a nested dictionary, adding keys as necessary."""
    if len(keys) == 1:
        d[keys[0]] = val
        return d

    if keys[0] not in d.keys():
        d[keys[0]] = {}
    d[keys[0]] = recursive_dict_add(d[keys[0]], val, keys[1:])

    return d

def recursive_dict_create(d, keys: list):
    """Create a nested dictionary.  This is the same as recursive_dict_add but
    with the desired value of an empty dictionary."""
    return recursive_dict_add(d, {}, keys)

def add_experiment_to_overall_results(experiment_results, add_to_results):
    # First interpret the experiment's object and training data so can narrow
    # down to the place in the overall results.
    vision_asset = experiment_results['_overview']['vision_asset']
    if 'robotocc_' in vision_asset or 'robot_' in vision_asset:
        object = '_'.join(vision_asset.split('_')[1:-1])
        print('robotocc_ or robot_ in vision asset')
    else:
        object = '_'.join(vision_asset.split('_')[:-1])
    tag_key = 'tagless_objects' if object in file_utils.TAGLESS_OBJECTS else \
        'tagged_objects'

    toss_key = vision_asset.split('_')[-1]
    start_toss = int(toss_key.split('-')[0])
    end_toss = start_toss if '-' not in toss_key else \
        int(toss_key.split('-')[1])
    trained_on_key = f'trained_on_toss_{start_toss}'
    trained_on_key += f'-{end_toss}' if start_toss != end_toss else ''

    add_to_results = recursive_dict_create(
        d=add_to_results, keys=[tag_key, object, trained_on_key])
    subresults = add_to_results[tag_key][object][trained_on_key]

    # Iterate over every metric category, e.g. dynamics_rollout_metrics.
    for category, exp_subresults in experiment_results.items():
        if category == '_overview':
            continue

        # Geometry metrics don't need to be conglomerated; copy over fully.
        elif category == 'geometry_metrics':
            recursive_dict_add(
                d=subresults, val=exp_subresults, keys=[category])
            continue

        # TODO:  these were commented out for robotocc experiments but would be
        # better to programmatically determine if these should be run or not
        # based on the experiment.

        # elif category == 'tracking_metrics':
        #     # Tracking and dynamics metrics need to be conglomerated by toss.
        #     # Iterate over every comparison against, e.g. against_bundlesdf.
        #     for against, subsubresults in exp_subresults.items():
        #         # Iterate over every metric, e.g. position_error.
        #         for metric, traj_results in subsubresults.items():

        #             # Iterate over every trajectory, e.g. toss_1.
        #             toss_means = []
        #             toss_aucs = []
        #             for traj_name, reported_errors in traj_results.items():
        #                 # If full trajectory, can use the mean and AUC directly.
        #                 if traj_name == 'full':
        #                     subresults = recursive_dict_add(
        #                         d=subresults,
        #                         val=reported_errors['mean'],
        #                         keys=[category, 'full', against, metric, 'mean']
        #                     )
        #                     subresults = recursive_dict_add(
        #                         d=subresults,
        #                         val=reported_errors['auc'],
        #                         keys=[category, 'full', against, metric,
        #                               'mean_auc']
        #                     )
        #                     continue

        #                 # Otherwise, need to conglomerate by toss.
        #                 toss_means.append(reported_errors['mean'])
        #                 toss_aucs.append(reported_errors['auc'])

        #             # Add the mean and AUC for the tosses.
        #             try:
        #                 subresults = recursive_dict_add(
        #                     d=subresults,
        #                     val=np.mean(toss_means).item(),
        #                     keys=[category, 'toss', against, metric, 'mean']
        #                 )
        #             except Exception as e:
        #                 print('Error in adding mean to results.')
        #                 print(e)
        #                 pdb.set_trace()
        #             subresults = recursive_dict_add(
        #                 d=subresults,
        #                 val=np.mean(toss_aucs).item(),
        #                 keys=[category, 'toss', against, metric, 'mean_auc']
        #             )
        #     continue

        # # Dynamics metrics need to be split by training and evaluation tosses.
        # # Tracking and dynamics metrics need to be conglomerated by toss.
        # # Iterate over every comparison against, e.g. against_bundlesdf.
        # for against, subsubresults in exp_subresults.items():
        #     # Iterate over every metric, e.g. position_error.
        #     for metric, traj_results in subsubresults.items():

        #         # Iterate over every trajectory, e.g. toss_1.
        #         training_toss_means = []
        #         training_toss_aucs = []
        #         test_toss_means = []
        #         test_toss_aucs = []
        #         for traj_name, reported_errors in traj_results.items():
        #             toss_num = int(traj_name.split('toss_')[-1])
        #             if toss_num in range(start_toss, end_toss + 1):
        #                 training_toss_means.append(reported_errors['mean'])
        #                 training_toss_aucs.append(reported_errors['auc'])
        #             else:
        #                 test_toss_means.append(reported_errors['mean'])
        #                 test_toss_aucs.append(reported_errors['auc'])

        #         # Add the mean and AUC for the tosses.
        #         subresults = recursive_dict_add(
        #             d=subresults,
        #             val=None if None in training_toss_means else \
        #                 np.mean(training_toss_means).item(),
        #             keys=[category, 'training_tosses', against, metric,
        #             'mean']
        #         )
        #         subresults = recursive_dict_add(
        #             d=subresults,
        #             val=None if None in training_toss_aucs else \
        #                 np.mean(training_toss_aucs).item(),
        #             keys=[category, 'training_tosses', against, metric,
        #             'mean_auc']
        #         )
        #         if against == 'against_tagslam':
        #             subresults = recursive_dict_add(
        #                 d=subresults,
        #                 val=None if None in test_toss_means else \
        #                     np.mean(test_toss_means).item(),
        #                 keys=[category, 'unseen_tosses', against, metric,
        #                 'mean']
        #             )
        #             subresults = recursive_dict_add(
        #                 d=subresults,
        #                 val=None if None in test_toss_aucs else \
        #                     np.mean(test_toss_aucs).item(),
        #                 keys=[category, 'unseen_tosses', against, metric,
        #                 'mean_auc']
        #             )
        #             subresults = recursive_dict_add(
        #                 d=subresults,
        #                 val=None if None in training_toss_means+test_toss_means\
        #                     else np.mean(
        #                         training_toss_means + test_toss_means).item(),
        #                 keys=[category, 'all_tosses', against, metric,
        #                 'mean']
        #             )
        #             subresults = recursive_dict_add(
        #                 d=subresults,
        #                 val=None if None in training_toss_aucs+test_toss_aucs \
        #                     else np.mean(
        #                         training_toss_aucs + test_toss_aucs).item(),
        #                 keys=[category, 'all_tosses', against, metric,
        #                 'mean_auc']
        #             )

def add_experiment_to_gt_results(experiment_results, gt_results):
    # First interpret the experiment's object.
    vision_asset = experiment_results['_overview']['vision_asset']
    object = '_'.join(vision_asset.split('_')[:-1])

    gt_results = recursive_dict_create(d=gt_results, keys=[object])
    subresults = gt_results[object]

    # The only metric category implemented for GT is dynamics_rollout_metrics.
    for category, exp_subresults in experiment_results.items():
        if category != 'dynamics_rollout_metrics':
            continue

        # Dynamics metrics need to be split by training and evaluation tosses.
        # Tracking and dynamics metrics need to be conglomerated by toss.
        # Iterate over every comparison against, e.g. against_bundlesdf.
        subsubresults = exp_subresults['against_tagslam']

        # Iterate over every metric, e.g. position_error.
        for metric, traj_results in subsubresults.items():

            # Iterate over every trajectory, e.g. toss_1.
            means = []
            aucs = []
            for traj_name, reported_errors in traj_results.items():
                toss_num = int(traj_name.split('toss_')[-1])
                means.append(reported_errors['mean'])
                aucs.append(reported_errors['auc'])

                subresults = recursive_dict_add(
                    d=subresults,
                    val=reported_errors['mean'],
                    keys=[category, 'all_tosses', 'against_tagslam', metric,
                    f'toss_{toss_num}']
                )
                subresults = recursive_dict_add(
                    d=subresults,
                    val=reported_errors['auc'],
                    keys=[category, 'all_tosses', 'against_tagslam', metric,
                    f'toss_{toss_num}_auc']
                )

            subresults = recursive_dict_add(
                d=subresults,
                val=None if None in means else np.mean(means).item(),
                keys=[category, 'all_tosses', 'against_tagslam', metric,
                'mean']
            )
            subresults = recursive_dict_add(
                d=subresults,
                val=None if None in aucs else np.mean(aucs).item(),
                keys=[category, 'all_tosses', 'against_tagslam', metric,
                'mean_auc']
            )

def add_all_robot_predictions_to_overall_results(vysics_results, bsdf_results,
                                                 pll_results, gt_results):
    # Load the robot prediction results from the robot dynamics directory.
    vysics_dynamics_results = file_utils.load_robot_dynamics_yaml('vysics')
    bsdf_dynamics_results = file_utils.load_robot_dynamics_yaml('bsdf')
    pll_dynamics_results = file_utils.load_robot_dynamics_yaml('pll')
    gt_dynamics_results = file_utils.load_robot_dynamics_yaml('gt')

    add_robot_predictions_to_overall_results(
        vysics_dynamics_results, vysics_results)
    add_robot_predictions_to_overall_results(
        bsdf_dynamics_results, bsdf_results)
    add_robot_predictions_to_overall_results(pll_dynamics_results, pll_results)
    add_robot_predictions_to_overall_results(gt_dynamics_results, gt_results)

def add_robot_predictions_to_overall_results(dynamics_results, add_to_results):
    """Copy robot dynamics prediction results (which can be found under the 
    `dynamics_prediction_metrics` key in the 
    file_utils.robot_dynamics_dir()/{method}.yaml file, see: 
    robot_dynamics_predictions.ConglomeratedDynamicsMetrics.export_statistics() )
    to the overall results under the `robot_dynamics_rollout_metrics` key."""
    # Iterate over every object.
    for robotocc_obj in dynamics_results['tagless_objects'].keys():
        obj = robotocc_obj.replace('robotocc_', '')

        # First, find the sub-results where the dynamics results should go.
        tag_key = 'tagless_objects' if obj in file_utils.TAGLESS_OBJECTS else \
            'tagged_objects'

        # Iterate over every training toss.
        for trained_on in dynamics_results['tagless_objects'][robotocc_obj
                                                              ].keys():
            dyn_pred_dict = dynamics_results[
                'tagless_objects'][robotocc_obj][trained_on][
                    'dynamics_prediction_metrics']['full_geometry']
            recursive_dict_create(
                d=add_to_results,
                keys=[tag_key, obj, trained_on,
                      'robot_dynamics_rollout_metrics']
            )
            add_to_subresults = add_to_results[tag_key][obj][trained_on]
            recursive_dict_add(
                d=add_to_subresults,
                val=dyn_pred_dict,
                keys=['robot_dynamics_rollout_metrics']
            )


class ResultsPlotter:
    """Generate plots of results stored in result dictionaries."""
    def __init__(self, bsdf_pll_results: dict, nerf_on_results: dict,
                 bsdf_only_results: dict, #pll_only_results: dict,
                 bsdf_octree_results: dict,
                 #pll_tagslam_results: dict, pll_blind_results: dict,
                 pll_vision_results: dict, pll_size_results: dict,
                 pll_blind_b_results: dict, pll_blind_t_results: dict,
                 bsdf_convex_results: dict, bsdf_pll_convex_results: dict,
                 bsdf_convex_occ_results: dict, bsdf_pll_convex_occ_results: dict,
                 bsdf_pll_robotocc_results: dict, bsdf_robotocc_results: dict,
                 gt_results: dict, do_objects: bool):
        # Store the results dictionaries.
        self.results = {
            BSDF_PLL: bsdf_pll_results,
            NERF_ON: nerf_on_results,
            BSDF_ONLY: bsdf_only_results,
            BSDF_OCTREE: bsdf_octree_results,
            BSDF_PLL_CONVEX: bsdf_pll_convex_results,
            PLL_VISION: pll_vision_results,
            PLL_SIZE: pll_size_results,
            PLL_BLIND_B: pll_blind_b_results,
            PLL_BLIND_T: pll_blind_t_results,
            BSDF_CONVEX: bsdf_convex_results,
            BSDF_CONVEX_OCC: bsdf_convex_occ_results,
            BSDF_PLL_CONVEX_OCC: bsdf_pll_convex_occ_results,
            BSDF_PLL_ROBOTOCC: bsdf_pll_robotocc_results,
            BSDF_ROBOTOCC: bsdf_robotocc_results,
            GT: gt_results,
        }

        self.bsdf_pll_results = bsdf_pll_results
        self.nerf_on_results = nerf_on_results
        self.bsdf_only_results = bsdf_only_results
        self.bsdf_octree_results = bsdf_octree_results
        self.bsdf_pll_convex_results = bsdf_pll_convex_results
        # self.pll_only_results = pll_only_results
        # self.pll_tagslam_results = pll_tagslam_results
        # self.pll_blind_results = pll_blind_results
        self.pll_vision_results = pll_vision_results
        self.pll_size_results = pll_size_results
        self.pll_blind_b_results = pll_blind_b_results
        self.pll_blind_t_results = pll_blind_t_results
        self.bsdf_convex_results = bsdf_convex_results
        self.bsdf_pll_robotocc_results = bsdf_pll_robotocc_results
        self.bsdf_robotocc_results = bsdf_robotocc_results
        self.gt_results = gt_results

        # Get a plotting directory.
        self.plot_dir = file_utils.plot_dir()
        print(f'Preparing to store to {self.plot_dir}')

        self.do_objects = do_objects

        ### Extract all the objects.
        self.tagged_objects = []
        if 'tagged_objects' in bsdf_pll_robotocc_results.keys():
            self.tagged_objects = [
                key for key in bsdf_pll_robotocc_results['tagged_objects'].keys()]
        self.tagless_objects = []
        if 'tagless_objects' in bsdf_pll_robotocc_results.keys():
            self.tagless_objects = [
                key for key in bsdf_pll_robotocc_results['tagless_objects'].keys()]

    def plot_tagslam_tracking_error_vs_data(
            self, full_or_toss: str, tracking_metric: str):
        """Tracking metrics against TagSLAM.  Only doable for tagged objects and
        not for PLL-only."""
        # Keep track of all objects.
        all_objects_bsdf_pll_mean = [[], []]
        all_objects_nerf_on_mean = [[], []]
        all_objects_bsdf_only_mean = [[], []]

        all_objects_bsdf_pll_auc = [[], []]
        all_objects_nerf_on_auc = [[], []]
        all_objects_bsdf_only_auc = [[], []]

        # Prepare to zip in consistent order:  BSDF-PLL, NeRF Online, BundleSDF
        # Only.
        means = [all_objects_bsdf_pll_mean, all_objects_nerf_on_mean,
                 all_objects_bsdf_only_mean]
        aucs = [all_objects_bsdf_pll_auc, all_objects_nerf_on_auc,
                all_objects_bsdf_only_auc]
        result_dicts = [self.bsdf_pll_results, self.nerf_on_results,
                        self.bsdf_only_results]

        for obj in self.tagged_objects:
            scale = METRIC_SCALING[tracking_metric]
            auc_scale = METRIC_SCALING['auc']

            bsdf_pll_mean = [[], []]
            nerf_on_mean = [[], []]
            bsdf_only_mean = [[], []]

            bsdf_pll_auc = [[], []]
            nerf_on_auc = [[], []]
            bsdf_only_auc = [[], []]

            obj_means = [bsdf_pll_mean, nerf_on_mean, bsdf_only_mean]
            obj_aucs = [bsdf_pll_auc, nerf_on_auc, bsdf_only_auc]

            for all_mean, all_auc, obj_mean, obj_auc, result_dict in zip(
                    means, aucs, obj_means, obj_aucs, result_dicts):
                if 'tagged_objects' not in result_dict.keys():
                    obj_mean = None
                    obj_auc = None
                    continue
                for tosses, results in result_dict['tagged_objects'][
                    obj].items():
                    tosses = tosses.split('trained_on_toss_')[-1]
                    start_toss = int(tosses.split('-')[0])
                    end_toss = int(tosses.split('-')[-1])
                    n_tosses = end_toss - start_toss + 1

                    error = results['tracking_metrics'][full_or_toss][
                        'against_tagslam'][tracking_metric]['mean']
                    auc = results['tracking_metrics'][full_or_toss][
                        'against_tagslam'][tracking_metric]['mean_auc']

                    if error is None or auc is None:
                        continue
                    error *= scale
                    auc *= auc_scale

                    obj_mean[0].append(n_tosses)
                    obj_mean[1].append(error)
                    obj_auc[0].append(n_tosses)
                    obj_auc[1].append(auc)
                    all_mean[0].append(n_tosses)
                    all_mean[1].append(error)
                    all_auc[0].append(n_tosses)
                    all_auc[1].append(auc)

            # Generate the plots.
            self._do_plot(
                bp_data=bsdf_pll_mean, n_data=nerf_on_mean,
                bo_data=bsdf_only_mean,
                ylabel=ERROR_LABELS[tracking_metric], xlabel=NUM_TOSSES_LABEL,
                title=f'{obj} {full_or_toss} Trajectory'.title(),
                filename=f'{obj}_{tracking_metric}_v_data_{full_or_toss}',
                subdir='tracking')
            self._do_plot(
                bp_data=bsdf_pll_auc, n_data=nerf_on_auc,
                bo_data=bsdf_only_auc,
                ylabel=AUC_LABELS[tracking_metric], xlabel=NUM_TOSSES_LABEL,
                title=f'{obj} {full_or_toss} Trajectory'.title(),
                filename=f'{obj}_{tracking_metric}_v_data_{full_or_toss}' + \
                    f'_auc', subdir='tracking')

        # Do a confidence interval plot that aggregates all the objects.
        save_to_txt = (tracking_metric in ['add_error', 'adds_error']) and \
            (full_or_toss == 'full')
        self._do_confidence_interval_plot(
            bp_data=all_objects_bsdf_pll_mean,
            n_data=all_objects_nerf_on_mean,
            bo_data=all_objects_bsdf_only_mean,
            ylabel=ERROR_LABELS[tracking_metric], xlabel=NUM_TOSSES_LABEL,
            title=f'All Tagged Objects {full_or_toss.capitalize()} Trajectory',
            filename=f'all_tagged_objs_{tracking_metric}_v_data_' + \
                f'{full_or_toss}', subdir='tracking',
            save_to_txt=save_to_txt)
        self._do_confidence_interval_plot(
            bp_data=all_objects_bsdf_pll_auc,
            n_data=all_objects_nerf_on_auc,
            bo_data=all_objects_bsdf_only_auc,
            ylabel=AUC_LABELS[tracking_metric], xlabel=NUM_TOSSES_LABEL,
            title=f'All Tagged Objects {full_or_toss.capitalize()} Trajectory',
            filename=f'all_tagged_objs_{tracking_metric}_v_data_' + \
                f'{full_or_toss}_auc', subdir='tracking',
            save_to_txt=save_to_txt)

    def plot_bundlesdf_tracking_error_vs_data(
            self, full_or_toss: str, tracking_metric: str):
        """Tracking metrics against BundleSDF.  Doable for all objects but not
        for PLL-only."""
        # Keep track of all objects.
        all_objects_bsdf_pll_mean = [[], []]
        all_objects_nerf_on_mean = [[], []]
        all_objects_bsdf_only_mean = [[], []]

        all_objects_bsdf_pll_auc = [[], []]
        all_objects_nerf_on_auc = [[], []]
        all_objects_bsdf_only_auc = [[], []]

        # Prepare to zip in consistent order:  BSDF-PLL, NeRF Online, BundleSDF
        # Only.
        means = [all_objects_bsdf_pll_mean, all_objects_nerf_on_mean,
                 all_objects_bsdf_only_mean]
        aucs = [all_objects_bsdf_pll_auc, all_objects_nerf_on_auc,
                all_objects_bsdf_only_auc]
        result_dicts = [self.bsdf_pll_results, self.nerf_on_results,
                        self.bsdf_only_results]

        # Do both tagged and tagless objects.
        all_objects = self.tagless_objects + self.tagged_objects
        object_labels = ['tagless_objects'] * len(self.tagless_objects) + \
            ['tagged_objects'] * len(self.tagged_objects)
        for obj, tag_label in zip(all_objects, object_labels):
            scale = METRIC_SCALING[tracking_metric]
            auc_scale = METRIC_SCALING['auc']

            bsdf_pll_mean = [[], []]
            nerf_on_mean = [[], []]
            bsdf_only_mean = [[], []]

            bsdf_pll_auc = [[], []]
            nerf_on_auc = [[], []]
            bsdf_only_auc = [[], []]

            obj_means = [bsdf_pll_mean, nerf_on_mean, bsdf_only_mean]
            obj_aucs = [bsdf_pll_auc, nerf_on_auc, bsdf_only_auc]

            for all_mean, all_auc, obj_mean, obj_auc, result_dict in zip(
                    means, aucs, obj_means, obj_aucs, result_dicts):
                if 'tagged_objects' not in result_dict.keys():
                    obj_mean = None
                    obj_auc = None
                    continue
                for tosses, results in result_dict[tag_label][obj].items():
                    tosses = tosses.split('trained_on_toss_')[-1]
                    start_toss = int(tosses.split('-')[0])
                    end_toss = int(tosses.split('-')[-1])
                    n_tosses = end_toss - start_toss + 1

                    error = results['tracking_metrics'][full_or_toss][
                        'against_bundlesdf'][tracking_metric]['mean'] * scale
                    auc = results['tracking_metrics'][full_or_toss][
                        'against_bundlesdf'][tracking_metric]['mean_auc'] * \
                            auc_scale

                    obj_mean[0].append(n_tosses)
                    obj_mean[1].append(error)
                    obj_auc[0].append(n_tosses)
                    obj_auc[1].append(auc)
                    all_mean[0].append(n_tosses)
                    all_mean[1].append(error)
                    all_auc[0].append(n_tosses)
                    all_auc[1].append(auc)

            # Generate the plots.
            self._do_plot(
                bp_data=bsdf_pll_mean, n_data=nerf_on_mean,
                bo_data=bsdf_only_mean,
                ylabel=ERROR_LABELS[tracking_metric], xlabel=NUM_TOSSES_LABEL,
                title=f'{obj} {full_or_toss} Trajectory'.title(),
                filename=f'{obj}_{tracking_metric}_v_data_{full_or_toss}',
                subdir='tracking')
            self._do_plot(
                bp_data=bsdf_pll_auc, n_data=nerf_on_auc,
                bo_data=bsdf_only_auc,
                ylabel=AUC_LABELS[tracking_metric], xlabel=NUM_TOSSES_LABEL,
                title=f'{obj} {full_or_toss} Trajectory'.title(),
                filename=f'{obj}_{tracking_metric}_v_data_{full_or_toss}_auc',
                subdir='tracking')

        # Do a confidence interval plot that aggregates all the objects.
        self._do_confidence_interval_plot(
            bp_data=all_objects_bsdf_pll_mean,
            n_data=all_objects_nerf_on_mean,
            bo_data=all_objects_bsdf_only_mean,
            ylabel=ERROR_LABELS[tracking_metric], xlabel=NUM_TOSSES_LABEL,
            title=f'All Objects {full_or_toss.capitalize()} Trajectory',
            filename=f'all_objs_{tracking_metric}_v_data_' + \
                f'{full_or_toss}', subdir='tracking')
        self._do_confidence_interval_plot(
            bp_data=all_objects_bsdf_pll_auc,
            n_data=all_objects_nerf_on_auc,
            bo_data=all_objects_bsdf_only_auc,
            ylabel=AUC_LABELS[tracking_metric], xlabel=NUM_TOSSES_LABEL,
            title=f'All Objects {full_or_toss.capitalize()} Trajectory',
            filename=f'all_objs_{tracking_metric}_v_data_' + \
                f'{full_or_toss}_auc', subdir='tracking')

    def plot_object_tagslam_tracking_scatter(
            self, full_or_toss: str, tracking_metric: str,
            exps: list = [BSDF_ONLY, BSDF_PLL], do_sort: bool = True):
        """Scatter plot of tracking metrics against TagSLAM.  Only doable for
        tagged objects and not for PLL-only."""
        # Get the scale of the metric.
        scale = METRIC_SCALING[tracking_metric]

        by_objects = {key: dict() for key in exps}
        sort_key = exps[0] if do_sort else ''

        for exp_key in exps:
            by_object = by_objects[exp_key]
            result_dict = self.results[exp_key]

            # Iterate over tagged objects.
            for obj in self.tagged_objects:
                if 'tagged_objects' not in result_dict.keys():
                    continue
                if obj not in result_dict['tagged_objects'].keys():
                    continue

                by_object[obj] = {}

                # Iterate over all the toss strings.
                for trained_on, result in result_dict['tagged_objects'][obj].items():
                    toss_str = trained_on.split('trained_on_toss_')[-1]

                    by_object[obj][toss_str] = result['tracking_metrics'][
                        full_or_toss]['against_tagslam'][tracking_metric]['mean'] * scale

        # Generate the plots.
        title = f'Scatter {full_or_toss.replace("_", " ")} Tracking'.title()
        ylabel = ERROR_LABELS[tracking_metric]
        self._do_scatter_plot(
            data_dict=by_objects, exp_key_list=exps, sort_key=sort_key, ylabel=ylabel,
            xlabel_txt=self.tagged_objects, title=title,
            filename=f'scatter_tagslam_{tracking_metric}_mean_v_data_{full_or_toss}',
            subdir='object_tracking', save_to_txt=True, normalize=False, show_toss_id=True)

    def plot_geometry_error_vs_data(
            self, hull_or_full: str, geometry_metric: str):
        """Geometry metrics.  Doable for all objects and approaches."""
        # Keep track of all objects.
        all_objects_bsdf_pll_mean = [[], []]
        all_objects_nerf_on_mean = [[], []]
        all_objects_bsdf_only_mean = [[], []]
        all_objects_pll_vision_mean = [[], []]
        all_objects_pll_size_mean = [[], []]
        all_objects_pll_blind_b_mean = [[], []]
        all_objects_pll_blind_t_mean = [[], []]
        all_objects_bsdf_convex_mean = [[], []]
        all_objects_bsdf_pll_convex_mean = [[], []]

        # Prepare to zip in consistent order:  BSDF-PLL, NeRF Online, BundleSDF
        # Only, PLL Vision, PLL Size, PLL Blind B, PLL Blind T.
        means = [all_objects_bsdf_pll_mean, all_objects_nerf_on_mean,
                 all_objects_bsdf_only_mean, all_objects_pll_vision_mean,
                 all_objects_pll_size_mean, all_objects_pll_blind_b_mean,
                 all_objects_pll_blind_t_mean, all_objects_bsdf_convex_mean,
                 all_objects_bsdf_pll_convex_mean]
        result_dicts = [self.bsdf_pll_results, self.nerf_on_results,
                        self.bsdf_only_results, self.pll_vision_results,
                        self.pll_size_results, self.pll_blind_b_results,
                        self.pll_blind_t_results, self.bsdf_convex_results,
                        self.bsdf_pll_convex_results]

        # Do both tagged and tagless objects.
        all_objects = self.tagless_objects + self.tagged_objects
        object_labels = ['tagless_objects'] * len(self.tagless_objects) + \
            ['tagged_objects'] * len(self.tagged_objects)
        for obj, tag_label in zip(all_objects, object_labels):
            scale = METRIC_SCALING[geometry_metric]

            bsdf_pll_mean = [[], []]
            nerf_on_mean = [[], []]
            bsdf_only_mean = [[], []]
            pll_vision_mean = [[], []]
            pll_size_mean = [[], []]
            pll_blind_b_mean = [[], []]
            pll_blind_t_mean = [[], []]
            bsdf_convex_mean = [[], []]
            bsdf_pll_convex_mean = [[], []]

            obj_means = [bsdf_pll_mean, nerf_on_mean, bsdf_only_mean,
                         pll_vision_mean, pll_size_mean, pll_blind_b_mean,
                         pll_blind_t_mean, bsdf_convex_mean, bsdf_pll_convex_mean]

            for all_mean, obj_mean, result_dict in zip(
                    means, obj_means, result_dicts):
                if tag_label not in result_dict.keys():
                    obj_mean = None
                    continue
                for tosses, results in result_dict[tag_label][obj
                    ].items():
                    tosses = tosses.split('trained_on_toss_')[-1]
                    start_toss = int(tosses.split('-')[0])
                    end_toss = int(tosses.split('-')[-1])
                    n_tosses = end_toss - start_toss + 1

                    error = results['geometry_metrics'][hull_or_full][
                        geometry_metric] * scale

                    obj_mean[0].append(n_tosses)
                    obj_mean[1].append(error)
                    all_mean[0].append(n_tosses)
                    all_mean[1].append(error)

            # Generate the plots.
            self._do_plot(
                bp_data=bsdf_pll_mean, n_data=nerf_on_mean,
                bo_data=bsdf_only_mean, pv_data=pll_vision_mean,
                ps_data=pll_size_mean, pbb_data=pll_blind_b_mean,
                pbt_data=pll_blind_t_mean, bc_data=bsdf_convex_mean,
                bpc_data=bsdf_pll_convex_mean,
                ylabel=ERROR_LABELS[geometry_metric], xlabel=NUM_TOSSES_LABEL,
                title=f'{obj} {hull_or_full.replace("_", " ")} Geometry'.title(),
                filename=f'{obj}_{geometry_metric}_v_data_{hull_or_full}',
                subdir='geometry')

        # Do a confidence interval plot that aggregates all the objects.
        self._do_confidence_interval_plot(
            bp_data=all_objects_bsdf_pll_mean,
            n_data=all_objects_nerf_on_mean,
            bo_data=all_objects_bsdf_only_mean,
            pv_data=all_objects_pll_vision_mean,
            ps_data=all_objects_pll_size_mean,
            pbb_data=all_objects_pll_blind_b_mean,
            pbt_data=all_objects_pll_blind_t_mean,
            bc_data=all_objects_bsdf_convex_mean,
            bpc_data=all_objects_bsdf_pll_convex_mean,
            ylabel=ERROR_LABELS[geometry_metric], xlabel=NUM_TOSSES_LABEL,
            title=f'All Objects {hull_or_full.replace("_", " ")} Geometry'.title(),
            filename=f'all_objs_{geometry_metric}_v_data_{hull_or_full}',
            subdir='geometry',
            save_to_txt=True)

    def plot_tagslam_dynamics_error_vs_data(
            self, toss_subset: str, dynamics_category: str,
            dynamics_metric: str):
        """Dynamics metrics against TagSLAM.  Only doable for tagged objects.
        BundleSDF-only results use the BundleSDF mesh and average dynamics
        parameters for inertia and friction."""
        title_add = 'Rollout' if dynamics_category=='dynamics_rollout_metrics' \
            else 'Single-Step'

        # Keep track of all objects.
        all_objects_bsdf_pll_mean = [[], []]
        all_objects_nerf_on_mean = [[], []]
        all_objects_bsdf_only_mean = [[], []]
        all_objects_pll_vision_mean = [[], []]
        all_objects_pll_size_mean = [[], []]
        all_objects_pll_blind_b_mean = [[], []]
        all_objects_pll_blind_t_mean = [[], []]

        all_objects_bsdf_pll_auc = [[], []]
        all_objects_nerf_on_auc = [[], []]
        all_objects_bsdf_only_auc = [[], []]
        all_objects_pll_vision_auc = [[], []]
        all_objects_pll_size_auc = [[], []]
        all_objects_pll_blind_b_auc = [[], []]
        all_objects_pll_blind_t_auc = [[], []]

        # Prepare to zip in consistent order:  BSDF-PLL, NeRF Online, BSDF-only,
        # PLL Vision, PLL Size, PLL Blind B, PLL Blind T.
        means = [all_objects_bsdf_pll_mean, all_objects_nerf_on_mean,
                 all_objects_bsdf_only_mean, all_objects_pll_vision_mean,
                 all_objects_pll_size_mean, all_objects_pll_blind_b_mean,
                 all_objects_pll_blind_t_mean]
        aucs = [all_objects_bsdf_pll_auc, all_objects_nerf_on_auc,
                all_objects_bsdf_only_auc, all_objects_pll_vision_auc,
                all_objects_pll_size_auc, all_objects_pll_blind_b_auc,
                all_objects_pll_blind_t_auc]
        result_dicts = [self.bsdf_pll_results, self.nerf_on_results,
                        self.bsdf_only_results, self.pll_vision_results,
                        self.pll_size_results, self.pll_blind_b_results,
                        self.pll_blind_t_results]

        for obj in self.tagged_objects:
            scale = METRIC_SCALING[dynamics_metric]
            auc_scale = METRIC_SCALING['auc']

            bsdf_pll_mean = [[], []]
            nerf_on_mean = [[], []]
            bsdf_only_mean = [[], []]
            pll_vision_mean = [[], []]
            pll_size_mean = [[], []]
            pll_blind_b_mean = [[], []]
            pll_blind_t_mean = [[], []]

            bsdf_pll_auc = [[], []]
            nerf_on_auc = [[], []]
            bsdf_only_auc = [[], []]
            pll_vision_auc = [[], []]
            pll_size_auc = [[], []]
            pll_blind_b_auc = [[], []]
            pll_blind_t_auc = [[], []]

            obj_means = [bsdf_pll_mean, nerf_on_mean, bsdf_only_mean,
                         pll_vision_mean, pll_size_mean, pll_blind_b_mean,
                         pll_blind_t_mean]
            obj_aucs = [bsdf_pll_auc, nerf_on_auc, bsdf_only_auc,
                        pll_vision_auc, pll_size_auc, pll_blind_b_auc,
                        pll_blind_t_auc]

            for all_mean, all_auc, obj_mean, obj_auc, result_dict in zip(
                    means, aucs, obj_means, obj_aucs, result_dicts):
                if 'tagged_objects' not in result_dict.keys():
                    obj_mean = None
                    obj_auc = None
                    continue

                for tosses, results in result_dict['tagged_objects'][obj].items():
                    tosses = tosses.split('trained_on_toss_')[-1]
                    start_toss = int(tosses.split('-')[0])
                    end_toss = int(tosses.split('-')[-1])
                    n_tosses = end_toss - start_toss + 1

                    # Ensure there are even numbers for all training set sizes.
                    if start_toss != 1:
                        continue

                    error = results[dynamics_category][toss_subset][
                        'against_tagslam'][dynamics_metric]['mean'] * scale
                    auc = results[dynamics_category][toss_subset][
                        'against_tagslam'][dynamics_metric]['mean_auc'] * \
                            auc_scale

                    obj_mean[0].append(n_tosses)
                    obj_mean[1].append(error)
                    obj_auc[0].append(n_tosses)
                    obj_auc[1].append(auc)
                    all_mean[0].append(n_tosses)
                    all_mean[1].append(error)
                    all_auc[0].append(n_tosses)
                    all_auc[1].append(auc)

            # Generate the plots.
            self._do_plot(
                bp_data=bsdf_pll_mean, n_data=nerf_on_mean,
                bo_data=bsdf_only_mean, pv_data=pll_vision_mean,
                ps_data=pll_size_mean, pbb_data=pll_blind_b_mean,
                pbt_data=pll_blind_t_mean,
                ylabel=ERROR_LABELS[dynamics_metric], xlabel=NUM_TOSSES_LABEL,
                title=f'{obj} {toss_subset.replace("_", " ")} Dynamics '.title() + \
                    f'{title_add} Prediction'.title(),
                filename=f'{obj}_{dynamics_metric}_tagslam_v_data_' + \
                    f'{toss_subset}_{dynamics_category}', subdir='dynamics')
            self._do_plot(
                bp_data=bsdf_pll_auc, n_data=nerf_on_auc, bo_data=bsdf_only_auc,
                pv_data=pll_vision_auc, ps_data=pll_size_auc,
                pbb_data=pll_blind_b_auc, pbt_data=pll_blind_t_auc,
                ylabel=AUC_LABELS[dynamics_metric], xlabel=NUM_TOSSES_LABEL,
                title=f'{obj} {toss_subset.replace("_", " ")} Dynamics '.title() + \
                    f'{title_add} Prediction'.title(),
                filename=f'{obj}_{dynamics_metric}_tagslam_v_data_{toss_subset}' + \
                    f'_{dynamics_category}_auc', subdir='dynamics')

        # Do a confidence interval plot that aggregates all the objects.
        self._do_confidence_interval_plot(
            bp_data=all_objects_bsdf_pll_mean,
            n_data=all_objects_nerf_on_mean,
            bo_data=all_objects_bsdf_only_mean,
            pv_data=all_objects_pll_vision_mean,
            ps_data=all_objects_pll_size_mean,
            pbb_data=all_objects_pll_blind_b_mean,
            pbt_data=all_objects_pll_blind_t_mean,
            ylabel=ERROR_LABELS[dynamics_metric], xlabel=NUM_TOSSES_LABEL,
            title=f'All Tagged Objects {toss_subset.replace("_", " ")} '.title() + \
                f'Dynamics {title_add} Prediction'.title(),
            filename=f'all_tagged_objs_{dynamics_metric}_tagslam_v_data_' + \
                f'{toss_subset}_{dynamics_category}', subdir='dynamics')
        self._do_confidence_interval_plot(
            bp_data=all_objects_bsdf_pll_auc,
            n_data=all_objects_nerf_on_auc,
            bo_data=all_objects_bsdf_only_auc,
            pv_data=all_objects_pll_vision_auc,
            ps_data=all_objects_pll_size_auc,
            pbb_data=all_objects_pll_blind_b_auc,
            pbt_data=all_objects_pll_blind_t_auc,
            ylabel=AUC_LABELS[dynamics_metric], xlabel=NUM_TOSSES_LABEL,
            title=f'All Tagged Objects {toss_subset.replace("_", " ")} '.title() + \
                f'Dynamics {title_add} Prediction'.title(),
            filename=f'all_tagged_objs_{dynamics_metric}_tagslam_v_data_' + \
                f'{toss_subset}_{dynamics_category}_auc', subdir='dynamics')

    def plot_bundlesdf_dynamics_error_vs_data(
            self, toss_subset: str, dynamics_category: str,
            dynamics_metric: str):
        """Dynamics metrics against BundleSDF.  Doable for all objects, not for
        PLL trained on TagSLAM, and since against BundleSDF this only works for
        all training tosses since predicting beyond the dataset requires
        TagSLAM.  BundleSDF-only uses the BundleSDF mesh and average parameters
        for inertia and friction."""
        if toss_subset != 'training_tosses':
            print(f'Cannot compute dynamics predictions w.r.t. BundleSDF ' + \
                  f'beyond the training set (told to do {toss_subset}; skip.')
            return

        title_add = 'Rollout' if dynamics_category=='dynamics_rollout_metrics' \
            else 'Single-Step'

        # Keep track of all objects.
        tagged_objects_bsdf_pll_mean = [[], []]
        tagged_objects_nerf_on_mean = [[], []]
        tagged_objects_bsdf_only_mean = [[], []]
        tagged_objects_pll_vision_mean = [[], []]
        tagged_objects_pll_size_mean = [[], []]
        tagged_objects_pll_blind_b_mean = [[], []]

        tagged_objects_bsdf_pll_auc = [[], []]
        tagged_objects_nerf_on_auc = [[], []]
        tagged_objects_bsdf_only_auc = [[], []]
        tagged_objects_pll_vision_auc = [[], []]
        tagged_objects_pll_size_auc = [[], []]
        tagged_objects_pll_blind_b_auc = [[], []]

        tagless_objects_bsdf_pll_mean = [[], []]
        tagless_objects_nerf_on_mean = [[], []]
        tagless_objects_bsdf_only_mean = [[], []]
        tagless_objects_pll_vision_mean = [[], []]
        tagless_objects_pll_size_mean = [[], []]
        tagless_objects_pll_blind_b_mean = [[], []]

        tagless_objects_bsdf_pll_auc = [[], []]
        tagless_objects_nerf_on_auc = [[], []]
        tagless_objects_bsdf_only_auc = [[], []]
        tagless_objects_pll_vision_auc = [[], []]
        tagless_objects_pll_size_auc = [[], []]
        tagless_objects_pll_blind_b_auc = [[], []]

        # Prepare to zip in consistent order:  BSDF-PLL, NeRF Online, BSDF-only,
        # PLL Vision, PLL Size, PLL Blind B.
        tagged_means = [
            tagged_objects_bsdf_pll_mean, tagged_objects_nerf_on_mean,
            tagged_objects_bsdf_only_mean, tagged_objects_pll_vision_mean,
            tagged_objects_pll_size_mean, tagged_objects_pll_blind_b_mean]
        tagged_aucs = [
            tagged_objects_bsdf_pll_auc, tagged_objects_nerf_on_auc,
            tagged_objects_bsdf_only_auc, tagged_objects_pll_vision_auc,
            tagged_objects_pll_size_auc, tagged_objects_pll_blind_b_auc]
        tagless_means = [
            tagless_objects_bsdf_pll_mean, tagless_objects_nerf_on_mean,
            tagless_objects_bsdf_only_mean, tagless_objects_pll_vision_mean,
            tagless_objects_pll_size_mean, tagless_objects_pll_blind_b_mean]
        tagless_aucs = [
            tagless_objects_bsdf_pll_auc, tagless_objects_nerf_on_auc,
            tagless_objects_bsdf_only_auc, tagless_objects_pll_vision_auc,
            tagless_objects_pll_size_auc, tagless_objects_pll_blind_b_auc]
        result_dicts = [self.bsdf_pll_results, self.nerf_on_results,
                        self.bsdf_only_results, self.pll_vision_results,
                        self.pll_size_results, self.pll_blind_b_results]

        # Do both tagged and tagless objects.
        all_objects = self.tagless_objects + self.tagged_objects
        object_labels = ['tagless_objects'] * len(self.tagless_objects) + \
            ['tagged_objects'] * len(self.tagged_objects)
        for obj, tag_label in zip(all_objects, object_labels):
            # Cannot do out-of-training data dynamics predictions for tagless
            # objects.
            if obj in self.tagless_objects and toss_subset != 'training_tosses':
                continue
            # Also cannot do out-of-training data dynamics predictions w.r.t.
            # BundleSDF, even for tagged objects.
            # if toss_subset != 'training_tosses':
            #     continue

            scale = METRIC_SCALING[dynamics_metric]
            auc_scale = METRIC_SCALING['auc']

            bsdf_pll_mean = [[], []]
            nerf_on_mean = [[], []]
            bsdf_only_mean = [[], []]
            pll_vision_mean = [[], []]
            pll_size_mean = [[], []]
            pll_blind_b_mean = [[], []]

            bsdf_pll_auc = [[], []]
            nerf_on_auc = [[], []]
            bsdf_only_auc = [[], []]
            pll_vision_auc = [[], []]
            pll_size_auc = [[], []]
            pll_blind_b_auc = [[], []]

            obj_means = [bsdf_pll_mean, nerf_on_mean, bsdf_only_mean,
                        pll_vision_mean,  pll_size_mean, pll_blind_b_mean]
            obj_aucs = [bsdf_pll_auc, nerf_on_auc, bsdf_only_auc,
                        pll_vision_auc, pll_size_auc, pll_blind_b_auc]

            for tagged_mean, tagged_auc, tagless_mean, tagless_auc, obj_mean, \
                obj_auc, result_dict in zip(
                    tagged_means, tagged_aucs, tagless_means, tagless_aucs, \
                    obj_means, obj_aucs, result_dicts):
                if tag_label not in result_dict.keys():
                    obj_mean = None
                    obj_auc = None
                    continue
                for tosses, results in result_dict[tag_label][obj].items():
                    tosses = tosses.split('trained_on_toss_')[-1]
                    start_toss = int(tosses.split('-')[0])
                    end_toss = int(tosses.split('-')[-1])
                    n_tosses = end_toss - start_toss + 1

                    # Ensure there are even numbers for all training set sizes.
                    if start_toss != 1:
                        continue

                    error = results[dynamics_category][toss_subset][
                        'against_bundlesdf'][dynamics_metric]['mean'] * scale
                    auc = results[dynamics_category][toss_subset][
                        'against_bundlesdf'][dynamics_metric]['mean_auc'] * \
                            auc_scale

                    obj_mean[0].append(n_tosses)
                    obj_auc[0].append(n_tosses)
                    obj_mean[1].append(error)
                    obj_auc[1].append(auc)

                    if obj in self.tagged_objects:
                        tagged_mean[0].append(n_tosses)
                        tagged_auc[0].append(n_tosses)
                        tagged_mean[1].append(error)
                        tagged_auc[1].append(auc)
                    else:
                        tagless_mean[0].append(n_tosses)
                        tagless_auc[0].append(n_tosses)
                        tagless_mean[1].append(error)
                        tagless_auc[1].append(auc)

            # Generate the plots.
            self._do_plot(
                bp_data=bsdf_pll_mean, n_data=nerf_on_mean,
                bo_data=bsdf_only_mean, pv_data=pll_vision_mean,
                ps_data=pll_size_mean, pbb_data=pll_blind_b_mean,
                ylabel=ERROR_LABELS[dynamics_metric],
                xlabel=NUM_TOSSES_LABEL,
                title=f'{obj} {toss_subset.replace("_", " ")} Dynamics '.title() + \
                    f'{title_add} Prediction'.title(),
                filename=f'{obj}_{dynamics_metric}_bsdf_v_data_{toss_subset}' +\
                    f'_{dynamics_category}', subdir='dynamics')
            self._do_plot(
                bp_data=bsdf_pll_auc, n_data=nerf_on_auc,
                bo_data=bsdf_only_auc, pv_data=pll_vision_auc,
                ps_data=pll_size_auc, pbb_data=pll_blind_b_auc,
                ylabel=AUC_LABELS[dynamics_metric],
                xlabel=NUM_TOSSES_LABEL,
                title=f'{obj} {toss_subset.replace("_", " ")} Dynamics '.title() + \
                    f'{title_add} Prediction'.title(),
                filename=f'{obj}_{dynamics_metric}_bsdf_v_data_{toss_subset}' + \
                    f'_{dynamics_category}_auc', subdir='dynamics')

        # Do a confidence interval plot that aggregates all the objects.
        all_objects_bsdf_pll_mean = \
            tagged_objects_bsdf_pll_mean + tagless_objects_bsdf_pll_mean
        all_objects_nerf_on_mean = \
            tagged_objects_nerf_on_mean + tagless_objects_nerf_on_mean
        all_objects_bsdf_only_mean = \
            tagged_objects_bsdf_only_mean + tagless_objects_bsdf_only_mean
        all_objects_pll_vision_mean = \
            tagged_objects_pll_vision_mean + tagless_objects_pll_vision_mean
        all_objects_pll_size_mean = \
            tagged_objects_pll_size_mean + tagless_objects_pll_size_mean
        all_objects_pll_blind_b_mean = \
            tagged_objects_pll_blind_b_mean + tagless_objects_pll_blind_b_mean
        all_objects_bsdf_pll_auc = \
            tagged_objects_bsdf_pll_auc + tagless_objects_bsdf_pll_auc
        all_objects_nerf_on_auc = \
            tagged_objects_nerf_on_auc + tagless_objects_nerf_on_auc
        all_objects_bsdf_only_auc = \
            tagged_objects_bsdf_only_auc + tagless_objects_bsdf_only_auc
        all_objects_pll_vision_auc = \
            tagged_objects_pll_vision_auc + tagless_objects_pll_vision_auc
        all_objects_pll_size_auc = \
            tagged_objects_pll_size_auc + tagless_objects_pll_size_auc
        all_objects_pll_blind_b_auc = \
            tagged_objects_pll_blind_b_auc + tagless_objects_pll_blind_b_auc
        self._do_confidence_interval_plot(
            bp_data=tagged_objects_bsdf_pll_mean,
            n_data=tagged_objects_nerf_on_mean,
            bo_data=tagged_objects_bsdf_only_mean,
            pv_data=tagged_objects_pll_vision_mean,
            ps_data=tagged_objects_pll_size_mean,
            pbb_data=tagged_objects_pll_blind_b_mean,
            ylabel=ERROR_LABELS[dynamics_metric], xlabel=NUM_TOSSES_LABEL,
            title=f'Tagged Objects {toss_subset.replace("_", " ")} '.title() + \
                f'Dynamics {title_add} Prediction'.title(),
            filename=f'tagged_objs_{dynamics_metric}_bsdf_v_data_{toss_subset}' + \
                f'_{dynamics_category}', subdir='dynamics')
        self._do_confidence_interval_plot(
            bp_data=tagged_objects_bsdf_pll_auc,
            n_data=tagged_objects_nerf_on_auc,
            bo_data=tagged_objects_bsdf_only_auc,
            pv_data=tagged_objects_pll_vision_auc,
            ps_data=tagged_objects_pll_size_auc,
            pbb_data=tagged_objects_pll_blind_b_auc,
            ylabel=AUC_LABELS[dynamics_metric], xlabel=NUM_TOSSES_LABEL,
            title=f'Tagged Objects {toss_subset.replace("_", " ")} '.title() + \
                f'Dynamics {title_add} Prediction'.title(),
            filename=f'tagged_objs_{dynamics_metric}_bsdf_v_data_{toss_subset}' + \
                f'_{dynamics_category}_auc', subdir='dynamics')
        self._do_confidence_interval_plot(
            bp_data=tagless_objects_bsdf_pll_mean,
            n_data=tagless_objects_nerf_on_mean,
            bo_data=tagless_objects_bsdf_only_mean,
            pv_data=tagless_objects_pll_vision_mean,
            ps_data=tagless_objects_pll_size_mean,
            pbb_data=tagless_objects_pll_blind_b_mean,
            ylabel=ERROR_LABELS[dynamics_metric], xlabel=NUM_TOSSES_LABEL,
            title=f'Tagless Objects {toss_subset.replace("_", " ")} '.title() + \
                f'Dynamics {title_add} Prediction'.title(),
            filename=f'tagless_objs_{dynamics_metric}_bsdf_v_data_{toss_subset}' + \
                f'_{dynamics_category}', subdir='dynamics')
        self._do_confidence_interval_plot(
            bp_data=tagless_objects_bsdf_pll_auc,
            n_data=tagless_objects_nerf_on_auc,
            bo_data=tagless_objects_bsdf_only_auc,
            pv_data=tagless_objects_pll_vision_auc,
            ps_data=tagless_objects_pll_size_auc,
            pbb_data=tagless_objects_pll_blind_b_auc,
            ylabel=AUC_LABELS[dynamics_metric], xlabel=NUM_TOSSES_LABEL,
            title=f'Tagless Objects {toss_subset.replace("_", " ")} '.title() + \
                f'Dynamics {title_add} Prediction'.title(),
            filename=f'tagless_objs_{dynamics_metric}_bsdf_v_data_{toss_subset}' + \
                f'_{dynamics_category}_auc', subdir='dynamics')
        self._do_confidence_interval_plot(
            bp_data=all_objects_bsdf_pll_mean,
            n_data=all_objects_nerf_on_mean,
            bo_data=all_objects_bsdf_only_mean,
            pv_data=all_objects_pll_vision_mean,
            ps_data=all_objects_pll_size_mean,
            pbb_data=all_objects_pll_blind_b_mean,
            ylabel=ERROR_LABELS[dynamics_metric], xlabel=NUM_TOSSES_LABEL,
            title=f'All Objects {toss_subset.replace("_", " ")} '.title() + \
                f'Dynamics {title_add} Prediction'.title(),
            filename=f'all_objs_{dynamics_metric}_bsdf_v_data_{toss_subset}' + \
                f'_{dynamics_category}', subdir='dynamics')
        self._do_confidence_interval_plot(
            bp_data=all_objects_bsdf_pll_auc,
            n_data=all_objects_nerf_on_auc,
            bo_data=all_objects_bsdf_only_auc,
            pv_data=all_objects_pll_vision_auc,
            ps_data=all_objects_pll_size_auc,
            pbb_data=all_objects_pll_blind_b_auc,
            ylabel=AUC_LABELS[dynamics_metric], xlabel=NUM_TOSSES_LABEL,
            title=f'All Objects {toss_subset.replace("_", " ")} '.title() + \
                f'Dynamics {title_add} Prediction'.title(),
            filename=f'all_objs_{dynamics_metric}_bsdf_v_data_{toss_subset}' + \
                f'_{dynamics_category}_auc', subdir='dynamics')

    def plot_object_scatter(
            self, hull_or_full: str, geometry_metric: str,
            exps: list = [BSDF_ONLY, BSDF_PLL], do_sort: bool = True):

        # Get the scale of the metric.
        scale = METRIC_SCALING[geometry_metric]

        # Iterate over the evaluation scopes.
        for obj_scope in ['tagged', 'all']:
            if obj_scope == 'tagged':
                if len(self.tagged_objects) == 0:
                    continue
            # by_objects: nested dict (exp_key -> obj -> toss_str -> metric)
            by_objects = {key: {} for key in exps}
            sort_key = exps[0] if do_sort else ''

            # Iterate over all the approaches.
            for exp_key in exps:
                by_object = by_objects[exp_key]
                result_dict = self.results[exp_key]

                if obj_scope == 'tagged':
                    all_objects = self.tagged_objects
                    object_labels = ['tagged_objects']*len(self.tagged_objects)
                elif obj_scope == 'all':
                    all_objects = self.tagless_objects + self.tagged_objects
                    object_labels = \
                        ['tagless_objects']*len(self.tagless_objects) + \
                        ['tagged_objects']*len(self.tagged_objects)
                else:
                    raise ValueError(f'Unknown object scope {obj_scope}')

                # Iterate over all the objects.
                for obj, tag_label in zip(all_objects, object_labels):
                    if tag_label not in result_dict.keys():
                        continue
                    if obj not in result_dict[tag_label].keys():
                        continue

                    by_object[obj] = {}

                    # Iterate over all the toss strings.
                    for trained_on, result in \
                        result_dict[tag_label][obj].items():
                        toss_str = trained_on.split('trained_on_toss_')[-1]

                        if geometry_metric in \
                            result['robot_dynamics_rollout_metrics'].keys():

                            plot_subdir = 'robot_dynamics'
                            by_object[obj][toss_str] = result[
                                'robot_dynamics_rollout_metrics'][
                                    geometry_metric] * scale
                        else:
                            assert geometry_metric in result[
                                'geometry_metrics'][hull_or_full].keys()
                            
                            plot_subdir = 'object_geometry'
                            by_object[obj][toss_str] = result[
                                'geometry_metrics'][hull_or_full][
                                    geometry_metric] * scale

            # Generate the plots.
            title = 'Chamfer Distance \u2193' if \
                geometry_metric == 'chamfer_distance' and 'full' in hull_or_full \
                else f'Scatter {hull_or_full.replace("_", " ")} Geometry'.title()
            title = 'Volumetric IoU \u2191' if geometry_metric == 'iou' and 'full_geometry' in hull_or_full \
                else title
            title = 'Time Before Divergence in Position \u2191' if geometry_metric == 'time_before_bad_pos' \
                and 'full_geometry' in hull_or_full else title
            title = 'Time Before Divergence in Rotation \u2191' if geometry_metric == 'time_before_bad_rot' \
                and 'full_geometry' in hull_or_full else title
            title = 'Average Rollout Pose Error \u2193' if geometry_metric == 'pos_rollout_error_mean' \
                and 'full_geometry' in hull_or_full else title
            title = 'Average Rollout Pose Error \u2193' if geometry_metric == 'rot_rollout_error_mean' \
                and 'full_geometry' in hull_or_full else title
            title = 'Temporal IoU of Contact Activation \u2191' if geometry_metric == 'contact_activation_iou' \
                and 'full_geometry' in hull_or_full else title

            ylabel = '% object length' if geometry_metric == 'chamfer_distance' \
                and 'full' in hull_or_full else ERROR_LABELS[geometry_metric]
            if obj_scope == 'tagged':
                filename = f'scatter_{geometry_metric}_v_data_{hull_or_full}_normalized'
            elif obj_scope == 'all':
                filename = f'scatter_{geometry_metric}_v_data_allobj_{hull_or_full}_normalized'
            else:
                raise ValueError(f'Unknown object scope {obj_scope}')
            self._do_scatter_plot(
                data_dict=by_objects, exp_key_list=exps, sort_key=sort_key, ylabel=ylabel,
                xlabel_txt=all_objects, title=title,
                # filename=f'scatter_{geometry_metric}_v_data_allobj_{hull_or_full}_normalized',
                filename=filename,
                subdir=plot_subdir, save_to_txt=True, normalize=True,
                show_toss_id=True, )
                # toss_id_filter={'all': ['1','2','3','4','5']})

            ylabel = 'Centimeters' if geometry_metric == 'chamfer_distance' \
                and 'full' in hull_or_full else ERROR_LABELS[geometry_metric]
            ylabel = '' if geometry_metric == 'iou' and 'full' in hull_or_full else ylabel
            if obj_scope == 'tagged':
                filename = f'scatter_{geometry_metric}_v_data_{hull_or_full}_true_units'
            elif obj_scope == 'all':
                filename = f'scatter_{geometry_metric}_v_data_allobj_{hull_or_full}_true_units'
            else:
                raise ValueError(f'Unknown object scope {obj_scope}')
            self._do_scatter_plot(
                data_dict=by_objects, exp_key_list=exps, sort_key=sort_key, ylabel=ylabel,
                xlabel_txt=all_objects, title=title,
                filename=filename,
                subdir=plot_subdir, save_to_txt=True, normalize=False, show_toss_id=True,)
                # toss_id_filter={'all': ['1','2','3','4','5']})

    def plot_gt_dynamics_comparison(
            self, toss_subset: str, dynamics_metric: str):
        """Dynamics metrics against TagSLAM.  Only doable for objects with
        ground truth URDFs."""
        DYNAMICS_CATEGORY = 'dynamics_rollout_metrics'

        # Keep track of all objects.
        objects_w_gt_bsdf_pll_mean = [[], []]
        objects_w_gt_nerf_on_mean = [[], []]
        objects_w_gt_bsdf_only_mean = [[], []]
        objects_w_gt_pll_vision_mean = [[], []]
        objects_w_gt_pll_size_mean = [[], []]
        objects_w_gt_pll_blind_b_mean = [[], []]
        objects_w_gt_pll_blind_t_mean = [[], []]
        objects_w_gt_gt_mean = []

        objects_w_gt_bsdf_pll_auc = [[], []]
        objects_w_gt_nerf_on_auc = [[], []]
        objects_w_gt_bsdf_only_auc = [[], []]
        objects_w_gt_pll_vision_auc = [[], []]
        objects_w_gt_pll_size_auc = [[], []]
        objects_w_gt_pll_blind_b_auc = [[], []]
        objects_w_gt_pll_blind_t_auc = [[], []]
        objects_w_gt_gt_auc = []

        # Prepare to zip in consistent order:  BSDF-PLL, NeRF Online, BSDF-Only,
        # PLL Vision, PLL Size, PLL Blind B, PLL Blind T.
        means = [objects_w_gt_bsdf_pll_mean, objects_w_gt_nerf_on_mean,
                 objects_w_gt_bsdf_only_mean, objects_w_gt_pll_vision_mean,
                 objects_w_gt_pll_size_mean, objects_w_gt_pll_blind_b_mean,
                 objects_w_gt_pll_blind_t_mean]
        aucs = [objects_w_gt_bsdf_pll_auc, objects_w_gt_nerf_on_auc,
                objects_w_gt_bsdf_only_auc, objects_w_gt_pll_vision_auc,
                objects_w_gt_pll_size_auc, objects_w_gt_pll_blind_b_auc,
                objects_w_gt_pll_blind_t_auc]
        result_dicts = [self.bsdf_pll_results, self.nerf_on_results,
                        self.bsdf_only_results, self.pll_vision_results,
                        self.pll_size_results, self.pll_blind_b_results,
                        self.pll_blind_t_results]

        for obj in OBJECTS_WITH_GT_URDF:
            scale = METRIC_SCALING[dynamics_metric]
            auc_scale = METRIC_SCALING['auc']

            # Handle the ground truth first.
            gt_means, gt_aucs = [], []
            # Iterate over the individual tosses.
            for toss_key, gt_val in self.gt_results[obj][DYNAMICS_CATEGORY][
                'all_tosses']['against_tagslam'][dynamics_metric].items():
                if 'auc' in toss_key:
                    gt_aucs.append(gt_val * auc_scale)
                elif 'toss' in toss_key:
                    gt_means.append(gt_val * scale)

            gt_mean = self.gt_results[obj][DYNAMICS_CATEGORY]['all_tosses'][
                'against_tagslam'][dynamics_metric]['mean'] * scale
            gt_auc = self.gt_results[obj][DYNAMICS_CATEGORY]['all_tosses'][
                'against_tagslam'][dynamics_metric]['mean_auc'] * auc_scale
            objects_w_gt_gt_mean.append(gt_mean)
            objects_w_gt_gt_auc.append(gt_auc)

            # Handle every approach next.
            bsdf_pll_mean = [[], []]
            nerf_on_mean = [[], []]
            bsdf_only_mean = [[], []]
            pll_vision_mean = [[], []]
            pll_size_mean = [[], []]
            pll_blind_b_mean = [[], []]
            pll_blind_t_mean = [[], []]

            bsdf_pll_auc = [[], []]
            nerf_on_auc = [[], []]
            bsdf_only_auc = [[], []]
            pll_vision_auc = [[], []]
            pll_size_auc = [[], []]
            pll_blind_b_auc = [[], []]
            pll_blind_t_auc = [[], []]

            obj_means = [bsdf_pll_mean, nerf_on_mean, bsdf_only_mean,
                         pll_vision_mean, pll_size_mean, pll_blind_b_mean,
                         pll_blind_t_mean]
            obj_aucs = [bsdf_pll_auc, nerf_on_auc, bsdf_only_auc,
                        pll_vision_auc, pll_size_auc, pll_blind_b_auc,
                        pll_blind_t_auc]

            for all_mean, all_auc, obj_mean, obj_auc, result_dict in zip(
                    means, aucs, obj_means, obj_aucs, result_dicts):
                if 'tagged_objects' not in result_dict.keys():
                    obj_mean = None
                    obj_auc = None
                    continue
                for tosses, results in result_dict['tagged_objects'][obj].items():
                    tosses = tosses.split('trained_on_toss_')[-1]
                    start_toss = int(tosses.split('-')[0])
                    end_toss = int(tosses.split('-')[-1])
                    n_tosses = end_toss - start_toss + 1

                    # NOTE:  Currently commented out so we can get more data.
                    # if start_toss != 1:
                    #     continue

                    error = results[DYNAMICS_CATEGORY][toss_subset][
                        'against_tagslam'][dynamics_metric]['mean'] * scale
                    auc = results[DYNAMICS_CATEGORY][toss_subset][
                        'against_tagslam'][dynamics_metric]['mean_auc'] * \
                            auc_scale

                    obj_mean[0].append(n_tosses)
                    obj_mean[1].append(error)
                    obj_auc[0].append(n_tosses)
                    obj_auc[1].append(auc)
                    all_mean[0].append(n_tosses)
                    all_mean[1].append(error)
                    all_auc[0].append(n_tosses)
                    all_auc[1].append(auc)

            # Generate the plots.
            self._do_plot(
                bp_data=bsdf_pll_mean, n_data=nerf_on_mean,
                bo_data=bsdf_only_mean, pv_data=pll_vision_mean,
                ps_data=pll_size_mean, pbb_data=pll_blind_b_mean,
                pbt_data=pll_blind_t_mean, gt=gt_means,
                ylabel=ERROR_LABELS[dynamics_metric], xlabel=NUM_TOSSES_LABEL,
                title=f'{obj} {toss_subset.replace("_", " ")} Dynamics '.title() + \
                    f'Rollout Prediction'.title(),
                filename=f'{obj}_{dynamics_metric}_for_gt_comp_' + \
                    f'{toss_subset}_{DYNAMICS_CATEGORY}', subdir='gt_dynamics')
            self._do_plot(
                bp_data=bsdf_pll_auc, n_data=nerf_on_auc,
                bo_data=bsdf_only_auc, pv_data=pll_vision_auc,
                ps_data=pll_size_auc, pbb_data=pll_blind_b_auc,
                pbt_data=pll_blind_t_auc, gt=gt_aucs,
                ylabel=AUC_LABELS[dynamics_metric], xlabel=NUM_TOSSES_LABEL,
                title=f'{obj} {toss_subset.replace("_", " ")} Dynamics '.title() + \
                    f'Rollout Prediction'.title(),
                filename=f'{obj}_{dynamics_metric}_for_gt_comp_{toss_subset}' + \
                    f'_{DYNAMICS_CATEGORY}_auc', subdir='gt_dynamics')

        # Do a confidence interval plot that aggregates all the objects.
        if len(OBJECTS_WITH_GT_URDF) > 1:
            self._do_confidence_interval_plot(
                bp_data=objects_w_gt_bsdf_pll_mean,
                n_data=objects_w_gt_nerf_on_mean,
                bo_data=objects_w_gt_bsdf_only_mean,
                pv_data=objects_w_gt_pll_vision_mean,
                ps_data=objects_w_gt_pll_size_mean,
                pbb_data=objects_w_gt_pll_blind_b_mean,
                pbt_data=objects_w_gt_pll_blind_t_mean,
                gt=objects_w_gt_gt_auc,
                ylabel=ERROR_LABELS[dynamics_metric], xlabel=NUM_TOSSES_LABEL,
                title=f'All GT Objects {toss_subset.replace("_", " ")} '.title() + \
                    f'Dynamics Rollout Prediction'.title(),
                filename=f'all_gt_objs_{dynamics_metric}_for_gt_comp_' + \
                    f'{toss_subset}_{DYNAMICS_CATEGORY}', subdir='gt_dynamics')
            self._do_confidence_interval_plot(
                bp_data=objects_w_gt_bsdf_pll_auc,
                n_data=objects_w_gt_nerf_on_auc,
                bo_data=objects_w_gt_bsdf_only_auc,
                pv_data=objects_w_gt_pll_vision_auc,
                ps_data=objects_w_gt_pll_size_auc,
                pbb_data=objects_w_gt_pll_blind_b_auc,
                pbt_data=objects_w_gt_pll_blind_t_auc,
                gt=objects_w_gt_gt_auc,
                ylabel=AUC_LABELS[dynamics_metric], xlabel=NUM_TOSSES_LABEL,
                title=f'All GT Objects {toss_subset.replace("_", " ")} '.title() + \
                    f'Dynamics Rollout Prediction'.title(),
                filename=f'all_gt_objs_{dynamics_metric}_for_gt_comp_' + \
                    f'{toss_subset}_{DYNAMICS_CATEGORY}_auc',
                subdir='gt_dynamics')

    def plot_tracking_geometry_diff_correlation(self, exp_tracking: str,
                                                exp_geometry_diff: list):
        """
        Plotting the correlation between the tracking metrics of one experiment and the
        difference in geometry metrics between two experiments.
        """

        empty_results = file_utils.load_empty_results_yaml()

        # List of tracking metrics
        tracking_metrics_name = []
        tracking_metrics_data = []
        tracking_metrics_dict = {}
        for metric in ERROR_LABELS.keys():
            # Tracking.
            scale = METRIC_SCALING[metric]
            for trajectory in ['full', 'toss']:
                if metric in \
                    empty_results['tracking_metrics']['against_tagslam'].keys():
                    tracking_metric_name = f'{metric}_{trajectory}'
                    tracking_metrics_name.append(tracking_metric_name)
                    tracking_metrics_data.append([])
                    tracking_metrics_dict[tracking_metric_name] = {}
                    metric_dict = tracking_metrics_dict[tracking_metric_name]
                    # Load the tracking data.

                    # Iterate over all the objects.
                    all_objects = self.tagged_objects
                    object_labels = ['tagged_objects'] * len(self.tagged_objects)
                    for obj, tag_label in zip(all_objects, object_labels):
                        if tag_label not in self.results[exp_tracking].keys():
                            continue
                        if obj not in self.results[exp_tracking][tag_label].keys():
                            continue
                        metric_dict[obj] = {}
                        metric_obj_dict = metric_dict[obj]

                        # Iterate over all the toss strings.
                        toss_ids = ['1', '2', '3', '4', '5']
                        for toss_id in toss_ids:
                            if obj == 'napkin' and toss_id in ['1', '2', '5']:
                                continue
                            toss_str = f'trained_on_toss_{toss_id}'
                            assert toss_str in self.results[exp_tracking][
                                tag_label][obj].keys()
                            result = self.results[exp_tracking][
                                tag_label][obj][toss_str]

                            metric_obj_dict[toss_id] = result['tracking_metrics'][
                                trajectory]['against_tagslam'][metric]['mean'] * scale
                            tracking_metrics_data[-1].append(
                                metric_obj_dict[toss_id])

        # List of geometry metrics
        geometry_metrics_name = []
        geometry_metrics_data = []
        geometry_metrics_dict = {}
        for metric in ERROR_LABELS.keys():
            # Geometry.
            scale = METRIC_SCALING[metric]
            for hull_or_full in ['full_geometry', 'convex_hull']:
                for normalized in [True, False]:
                    if metric in \
                        empty_results['geometry_metrics'][hull_or_full].keys():
                        geometry_metric_name = f'{metric}_{hull_or_full}_' + \
                            f'{"normalized" if normalized else "true_units"}'
                        geometry_metrics_name.append(geometry_metric_name)
                        geometry_metrics_data.append([])
                        geometry_metrics_dict[geometry_metric_name] = {}
                        metric_dict = geometry_metrics_dict[geometry_metric_name]
                        # Load the geometry data.

                        # Iterate over all the objects.
                        all_objects = self.tagged_objects
                        object_labels = ['tagged_objects'] * len(self.tagged_objects)
                        for obj, tag_label in zip(all_objects, object_labels):
                            if tag_label not in self.results[
                                exp_geometry_diff[0]].keys():
                                continue
                            if obj not in self.results[
                                exp_geometry_diff[0]][tag_label].keys():
                                continue
                            metric_dict[obj] = {}
                            metric_obj_dict = metric_dict[obj]

                            # Iterate over all the toss strings.
                            toss_ids = ['1', '2', '3', '4', '5']
                            for toss_id in toss_ids:
                                if obj == 'napkin' and toss_id in ['1', '2', '5']:
                                    continue
                                toss_str = f'trained_on_toss_{toss_id}'
                                assert toss_str in self.results[exp_geometry_diff[0]][
                                    tag_label][obj].keys()
                                result_0 = self.results[exp_geometry_diff[0]][
                                    tag_label][obj][toss_str]
                                try:
                                    result_1 = self.results[exp_geometry_diff[1]][
                                        tag_label][obj][toss_str]
                                except:
                                    pdb.set_trace()

                                error_0 = result_0['geometry_metrics'][
                                    hull_or_full][metric] * scale
                                error_1 = result_1['geometry_metrics'][
                                    hull_or_full][metric] * scale
                                if normalized:
                                    error_0 /= CM_LENGTH_SCALES_BY_OBJ[obj]
                                    error_0 *= 100
                                    error_1 /= CM_LENGTH_SCALES_BY_OBJ[obj]
                                    error_1 *= 100
                                metric_obj_dict[toss_id] = error_1 - error_0
                                geometry_metrics_data[-1].append(error_1 - error_0)

        title = f'Spearman Correlation between {exp_tracking} Tracking \n and ' + \
                f'Geometry Metrics Diff {exp_geometry_diff[1]} - {exp_geometry_diff[0]}'
        self._do_correlation_matrix_plot(tracking_metrics_data, geometry_metrics_data,
                                         tracking_metrics_name, geometry_metrics_name,
                                         title=title, subdir='tracking_geometry_diff')

        title=f'{exp_tracking} Tracking vs. \n' + \
            f'{exp_geometry_diff[1]} - {exp_geometry_diff[0]} Geometry Metrics'
        self._do_correlation_scatter_plot(tracking_metrics_data, geometry_metrics_data,
                                     tracking_metrics_name, geometry_metrics_name,
                                     tracking_metrics_dict, geometry_metrics_dict,
                                     distinguish_exp = False,
                                     title = title, subdir = 'tracking_geometry_diff')

    def plot_tracking_geometry_correlation(
            self, exps: list = [BSDF_ONLY, BSDF_PLL]):
        """
        Plotting the correlation between tracking and geometry metrics.
        Given a list of experiments, go through each pair of tracking and geometric metrics,
        using log-transformed data if necessary, and apply standardization to the data.
        Then calculate the correlation between the two metrics, and plot the results.
        Use the Spearman correlation coefficient.
        """

        empty_results = file_utils.load_empty_results_yaml()
        # List of tracking metrics
        tracking_metrics_name = []
        tracking_metrics_data = []
        tracking_metrics_dict = {}
        for metric in ERROR_LABELS.keys():
            # Tracking.
            scale = METRIC_SCALING[metric]
            for trajectory in ['full', 'toss']:
                if metric in \
                    empty_results['tracking_metrics']['against_tagslam'].keys():
                    tracking_metric_name = f'{metric}_{trajectory}'
                    tracking_metrics_name.append(tracking_metric_name)
                    tracking_metrics_data.append([])
                    tracking_metrics_dict[tracking_metric_name] = {}
                    metric_dict = tracking_metrics_dict[tracking_metric_name]
                    # Load the tracking data.

                    # Iterate over all the approaches.
                    for exp_key in exps:
                        result_dict = self.results[exp_key]
                        metric_dict[exp_key] = {}
                        metric_exp_dict = metric_dict[exp_key]

                        # Iterate over all the objects.
                        all_objects = self.tagged_objects
                        object_labels = ['tagged_objects'] * len(self.tagged_objects)
                        for obj, tag_label in zip(all_objects, object_labels):
                            if tag_label not in result_dict.keys():
                                continue
                            if obj not in result_dict[tag_label].keys():
                                continue
                            metric_exp_dict[obj] = {}
                            metric_exp_obj_dict = metric_exp_dict[obj]

                            # Iterate over all the toss strings.
                            toss_ids = ['1', '2', '3', '4', '5']
                            for toss_id in toss_ids:
                                if obj == 'napkin' and toss_id in ['1', '2', '5']:
                                    continue
                                toss_str = f'trained_on_toss_{toss_id}'
                                assert toss_str in result_dict[tag_label][obj].keys()
                                result = result_dict[tag_label][obj][toss_str]

                                metric_exp_obj_dict[toss_id] = result['tracking_metrics'][
                                    trajectory]['against_tagslam'][metric]['mean'] * scale

                                tracking_metrics_data[-1].append(
                                    metric_exp_obj_dict[toss_id])

        # List of geometry metrics
        geometry_metrics_name = []
        geometry_metrics_data = []
        geometry_metrics_dict = {}
        geometry_metrics_dict_one_level = {}
        for metric in ERROR_LABELS.keys():
            # Geometry.
            scale = METRIC_SCALING[metric]
            for hull_or_full in ['full_geometry', 'convex_hull']:
                for normalized in [True, False]:
                    if metric in \
                        empty_results['geometry_metrics'][hull_or_full].keys():
                        geometry_metric_name = f'{metric}_{hull_or_full}_' + \
                            f'{"normalized" if normalized else "true_units"}'
                        geometry_metrics_name.append(geometry_metric_name)
                        geometry_metrics_data.append([])
                        geometry_metrics_dict_one_level[geometry_metric_name] = \
                            geometry_metrics_data[-1]
                        geometry_metrics_dict[geometry_metric_name] = {}
                        metric_dict = geometry_metrics_dict[geometry_metric_name]
                        # Load the geometry data.

                        # Iterate over all the approaches.
                        for exp_key in exps:
                            result_dict = self.results[exp_key]
                            metric_dict[exp_key] = {}
                            metric_exp_dict = metric_dict[exp_key]

                            # Iterate over all the objects.
                            all_objects = self.tagged_objects
                            object_labels = ['tagged_objects'] * len(self.tagged_objects)
                            for obj, tag_label in zip(all_objects, object_labels):
                                if tag_label not in result_dict.keys():
                                    continue
                                if obj not in result_dict[tag_label].keys():
                                    continue
                                metric_exp_dict[obj] = {}
                                metric_exp_obj_dict = metric_exp_dict[obj]

                                # Iterate over all the toss strings.
                                toss_ids = ['1', '2', '3', '4', '5']
                                for toss_id in toss_ids:
                                    if obj == 'napkin' and toss_id in ['1', '2', '5']:
                                        continue
                                    toss_str = f'trained_on_toss_{toss_id}'
                                    assert toss_str in result_dict[tag_label][obj].keys()
                                    result = result_dict[tag_label][obj][toss_str]

                                    error = result['geometry_metrics'][
                                        hull_or_full][metric] * scale
                                    if normalized:
                                        error /= CM_LENGTH_SCALES_BY_OBJ[obj]
                                        error *= 100
                                    metric_exp_obj_dict[toss_id] = error
                                    geometry_metrics_data[-1].append(error)

        title = 'Spearman Correlation between Tracking and Geometry Metrics \n' + \
                f'in {", ".join(exps)} Experiments'
        self._do_correlation_matrix_plot(tracking_metrics_data, geometry_metrics_data,
                                         tracking_metrics_name, geometry_metrics_name,
                                         title=title, subdir='tracking_geometry')

        title = f'Scatter Plot of Tracking vs. Geometry Metrics \n' + \
                f'in {", ".join(exps)} Experiments'
        self._do_correlation_scatter_plot(tracking_metrics_data, geometry_metrics_data,
                                        tracking_metrics_name, geometry_metrics_name,
                                        tracking_metrics_dict, geometry_metrics_dict,
                                        distinguish_exp = True, title = title,
                                        subdir = 'tracking_geometry')

    def _do_correlation_matrix_plot(self, data_1: list, data_2: list,
                                    data_1_labels: list, data_2_labels: list,
                                    title: str = 'Spearman Correlation',
                                    subdir: str = ''):
        data_1 = np.array(data_1)
        data_2 = np.array(data_2)

        # Calculate the spearmanr correlation between tracking and geometry metrics.
        spearmanr_correlation = stats.spearmanr(data_1, data_2, axis=1)
        spearmanr_correlation_mat = spearmanr_correlation.statistic
        pvalues = spearmanr_correlation.pvalue

        ### Plot the correlation matrix.
        fig, ax = plt.subplots()
        im = ax.imshow(spearmanr_correlation_mat)

        # We want to show all ticks...
        ax.set_xticks(np.arange(len(data_2_labels))+len(data_1_labels))
        ax.set_yticks(np.arange(len(data_1_labels)))
        # ... and label them with the respective list entries
        ax.set_xticklabels(data_2_labels)
        ax.set_yticklabels(data_1_labels)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                    rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        for i in range(len(data_1_labels)):
            for j in range(len(data_1_labels), len(data_1_labels) + len(data_2_labels)):
                text = ax.text(j, i, f'{spearmanr_correlation_mat[i, j]:.2f} \n' + \
                                f'({pvalues[i, j]:.2f})',
                            ha="center", va="center", color="w")

        ax.set_title(title, fontsize=20)
        fig.set_size_inches(15, 15)
        fig.tight_layout()
        # plt.show()
        filename = f'correlation_matrix.png'
        file_utils.assure_created(op.join(self.plot_dir, subdir))
        fig_path = op.join(self.plot_dir, subdir, filename)
        fig.savefig(fig_path, dpi=100)
        print(f'Saved {filename}')

    def _do_correlation_scatter_plot(self, data_1: list, data_2: list,
                                     data_1_labels: list, data_2_labels: list,
                                     data_1_dict: dict, data_2_dict: dict,
                                     distinguish_exp: bool = False, title: str = '',
                                     subdir: str = ''):
        """For each pair of metrics in data_dict_1 and data_dict_2,
        plot the scatter plot."""
        for i, data_1_label in enumerate(data_1_labels):
            data_1_data = data_1[i]
            for j, data_2_label in enumerate(data_2_labels):
                data_2_data = data_2[j]

                # Plot the scatter plot.
                fig, ax = plt.subplots()

                if not distinguish_exp:
                    # Draw a horizontal line at y=0.
                    ax.axhline(y=0, color='grey', linestyle='--')

                if distinguish_exp:
                    exp_legend_created = {exp_key: False for exp_key in \
                                          data_1_dict[data_1_label].keys()}
                # Add the text annotations.
                if distinguish_exp:
                    keys = sorted(list(data_1_dict[data_1_label].keys()))
                    for exp_key in keys:
                        assert exp_key in data_2_dict[data_2_label], \
                        f'{exp_key} not in {data_2_dict[data_2_label].keys()}'
                        for obj in data_1_dict[data_1_label][exp_key].keys():
                            assert obj in data_2_dict[data_2_label][exp_key], \
                            f'{obj} not in {data_2_dict[data_2_label][exp_key].keys()}'
                            for toss_id in data_1_dict[data_1_label][
                                exp_key][obj].keys():
                                assert toss_id in data_2_dict[data_2_label][exp_key][obj], \
                                f'{toss_id} not in {data_2_dict[data_2_label][exp_key][obj].keys()}'
                                x = data_1_dict[data_1_label][exp_key][obj][toss_id]
                                y = data_2_dict[data_2_label][exp_key][obj][toss_id]
                                if not exp_legend_created[exp_key]:
                                    exp_legend_created[exp_key] = True
                                    ax.scatter(x, y, color=COLORS[exp_key],
                                                label=LABELS[exp_key])
                                else:
                                    ax.scatter(x, y, color=COLORS[exp_key])
                                anno_txt = f'{obj[0]}{toss_id}'
                                ax.annotate(anno_txt, (x, y), #color=COLORS[exp_key],
                                            ha='left', va='bottom')
                else:
                    for obj in data_1_dict[data_1_label].keys():
                        for toss_id in data_1_dict[data_1_label][obj].keys():
                            x = data_1_dict[data_1_label][obj][toss_id]
                            y = data_2_dict[data_2_label][obj][toss_id]
                            anno_txt = f'{obj[0]}{toss_id}'
                            ax.annotate(anno_txt, (x, y), ha='left', va='bottom')

                if not distinguish_exp:
                    ax.scatter(data_1_data, data_2_data)
                    # ax.scatter(data_1_data, data_2_data, color='black')
                # else:
                #     ax.scatter(data_1_data, data_2_data)
                # Add the labels.
                ax.set_xlabel(data_1_label, fontsize=20)
                ax.set_ylabel(data_2_label, fontsize=20)
                ax.set_title(title, fontsize=20)

                if distinguish_exp:
                    ax.legend(fontsize=20)
                fig.set_size_inches(13, 10)
                # plt.show()
                filename = f'{data_1_label}_vs_{data_2_label}'
                filename += '_diff' if not distinguish_exp else ''
                filename += '.png' if not filename.endswith('.png') else ''
                file_utils.assure_created(op.join(self.plot_dir, subdir))
                fig_path = op.join(self.plot_dir, subdir, filename)
                fig.savefig(fig_path, dpi=100)
                print(f'Saved {filename}')
                plt.close()

    def _do_scatter_plot(self, data_dict: dict = None, exp_key_list: list = [],
                        sort_key: str = '', ylabel: str = '', xlabel_txt: str = '',
                        title: str = '', filename: str = '', subdir: str = '',
                        save_to_txt: bool = False, normalize: bool = False,
                        show_toss_id: bool = False, toss_id_filter: dict = None):
        """
        Modified scatter plot function supporting different toss IDs per object.

        Args:
            data_dict: Nested dictionary: {exp_key: {object_name: {toss_id: value}}}
            sort_key:  experiment key to sort the objects by average performance of tosses. 
            toss_id_filter: Dictionary specifying which toss IDs to include for each object.
                        Format: {'object_name': ['1', '2'], 'all': ['1', '2', '3']}
                        If None, use all available toss IDs for each object
                        If contains 'all' key, use those IDs for objects not specifically listed
        """
        assert data_dict is not None
        if sort_key == '':
            print('No sorting of the object list, plot will be in static order.')
        else:
            assert sort_key in data_dict.keys(), f'{sort_key=} not in {data_dict.keys()}'
            print(f'Sorting the object list by the performance of {sort_key}.')

        n_exp = len(data_dict)

        # Get all toss IDs for an object filtered by the toss_id_filter
        def get_toss_ids_for_object(obj_name, data):
            if toss_id_filter is None:
                # Use all available toss IDs for this object
                return sorted(list(data[obj_name].keys()))
            elif obj_name in toss_id_filter:
                # Use specified toss IDs for this object
                available_ids = set(data[obj_name].keys())
                filtered_ids = [tid for tid in toss_id_filter[obj_name] if tid in available_ids]
                return filtered_ids
            elif 'all' in toss_id_filter:
                # Use global filter if specified
                available_ids = set(data[obj_name].keys())
                filtered_ids = [tid for tid in toss_id_filter['all'] if tid in available_ids]
                return filtered_ids
            else:
                # No filter specified, use all available
                return sorted(list(data[obj_name].keys()))

        # Process data for each experiment
        ys = dict()
        ys_means = dict()
        all_toss_ids = dict()  # Store toss IDs for each object

        # First pass: collect all valid toss IDs for each object
        for exp_key in data_dict:
            data = data_dict[exp_key]
            for obj in xlabel_txt:
                if obj not in all_toss_ids:
                    all_toss_ids[obj] = get_toss_ids_for_object(obj, data)

        # Second pass: collect data points
        for exp_key in data_dict:
            data = data_dict[exp_key]
            ys[exp_key] = []
            ys_means[exp_key] = []

            # Process each object separately
            for obj in xlabel_txt:
                obj_toss_ids = all_toss_ids[obj]
                obj_data = [data[obj][toss_str] for toss_str in obj_toss_ids]

                # Store raw data
                ys[exp_key].extend(obj_data)

                # Calculate and store mean for this object
                obj_mean = sum(obj_data) / len(obj_data) if obj_data else 0
                ys_means[exp_key].append(obj_mean)

                # Normalize if requested
                if normalize:
                    for i in range(len(obj_data)):
                        ys[exp_key][-(i+1)] /= CM_LENGTH_SCALES_BY_OBJ[obj]
                        ys[exp_key][-(i+1)] *= 100
                    ys_means[exp_key][-1] /= CM_LENGTH_SCALES_BY_OBJ[obj]
                    ys_means[exp_key][-1] *= 100

        # Handle sorting if requested
        if sort_key != '':
            # Use ys_means[sort_key], which is a list of the mean performance of 
            # all tosses of each object in the sort_key experiment, to sort the objects.
            # All experiments' lists (ys_means[exp]) will be sorted in the same order.
            ys_list = []
            ys_means_list = []
            sort_key_i = None
            for i_key, exp_key in enumerate(exp_key_list):
                ys_list.append(ys[exp_key])
                ys_means_list.append(ys_means[exp_key])
                if exp_key == sort_key:
                    sort_key_i = i_key
            assert sort_key_i is not None, f'{sort_key=} not in {data_dict.keys()}'

            combined = list(zip(xlabel_txt, *ys_means_list))
            sorted_combined = sorted(combined, key=lambda x: x[sort_key_i+1], reverse=False)

            # Unzip the sorted data
            xlabel_txt_new = []
            ys_means_new = {exp_key: [] for exp_key in exp_key_list}
            ys_new = {exp_key: [] for exp_key in exp_key_list}

            # Reconstruct the sorted data
            for item in sorted_combined:
                obj_name = item[0]
                xlabel_txt_new.append(obj_name)

                for i, exp_key in enumerate(exp_key_list):
                    ys_means_new[exp_key].append(item[i+1])
                    # Find the corresponding data points for this object
                    start_idx = 0
                    for old_obj in xlabel_txt:
                        if old_obj == obj_name:
                            n_tosses = len(all_toss_ids[old_obj])
                            ys_new[exp_key].extend(ys[exp_key][start_idx:start_idx + n_tosses])
                            break
                        start_idx += len(all_toss_ids[old_obj])

            xlabel_txt = xlabel_txt_new
            ys = ys_new
            ys_means = ys_means_new

        # Create plot
        fig = plt.figure()
        ax = plt.gca()

        if save_to_txt:
            data_str = ''
            data_str += f'NORMALIZING? --> {normalize=}\n'
            data_str += f'{xlabel_txt=}\n'
            data_str += f'Toss IDs per object:\n'
            for obj in xlabel_txt:
                data_str += f'{obj}: {all_toss_ids[obj]}\n'

        # Calculate x-coordinates for scatter points
        width = 1 / n_exp
        x_offsets = np.linspace(-0.5+width/2, 0.5-width/2, n_exp)

        # Plot data points and means
        for i_key, exp_key in enumerate(exp_key_list):
            exp_color = COLORS[exp_key]
            exp_label = LABELS[exp_key]

            # Plot mean lines
            prefixes = [''] + ['_']*(len(xlabel_txt)-1)
            for i in range(len(xlabel_txt)):
                plt.plot([i-0.38, i+0.38],
                        [ys_means[exp_key][i], ys_means[exp_key][i]],
                        linestyle='-', linewidth=LINEWIDTH*2,
                        color=exp_color, label=prefixes[i]+exp_label)

            # Plot scatter points
            start_idx = 0
            for i, obj in enumerate(xlabel_txt):
                n_tosses = len(all_toss_ids[obj])
                obj_data = ys[exp_key][start_idx:start_idx + n_tosses]
                x_coords = [i + x_offsets[i_key]] * n_tosses
                plt.scatter(x_coords, obj_data, s=MARKERSIZE, color=exp_color,
                            label='_')

                # Add toss ID annotations if requested
                # if show_toss_id:
                #     for j, toss_id in enumerate(all_toss_ids[obj]):
                #         plt.annotate(toss_id,
                #                 (x_coords[j]+0.05, obj_data[j]),
                #                 fontsize=12,
                #                 ha='left',
                #                 va='center')
                start_idx += n_tosses

                # Draw connecting lines between experiments if showing toss IDs
                if show_toss_id and i_key < len(exp_key_list)-1:
                    next_exp_data = ys[exp_key_list[i_key+1]][start_idx-n_tosses:start_idx]
                    for j in range(n_tosses):
                        plt.plot([i-0.5+width*(i_key+0.5), i-0.5+width*(i_key+1.5)],
                                [obj_data[j], next_exp_data[j]],
                                linestyle='-', linewidth=0.5*LINEWIDTH,
                                color='grey', alpha=0.5)

            if save_to_txt:
                data_str += f'\n{exp_key}_data:\n{ys[exp_key]}'

        # Finalize plot
        plt.xticks(range(len(xlabel_txt)), xlabel_txt, rotation=45,
                ha='right', fontsize=20)

        if normalize:
            ylabel = ylabel.replace('[cm]', '[% length]')
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        # Leave some space for the legend at the top
        ymin, ymax = ax.get_ylim()
        ax.set_ylim([ymin, ymax*1.2])


        self._beautify_plot(fig, ax, range(len(xlabel_txt)), False,
                            scatter=True, normalized=normalize)

        # Save plot and data
        plot_filename = filename.split('.')[0] + '.png'
        file_utils.assure_created(op.join(self.plot_dir, subdir))
        fig_path = op.join(self.plot_dir, subdir, plot_filename)
        fig.savefig(fig_path, dpi=100)
        plt.close()

        if save_to_txt:
            str_filepath = fig_path.replace('.png', '.txt')
            with open(str_filepath, 'w') as txt_file:
                txt_file.write(data_str)
            print(f'Wrote to {str_filepath}')

    def _do_plot(self, bp_data: list = None, n_data: list = None,
                 bo_data: list = None, pv_data: list = None,
                 ps_data: list = None, pbb_data: list = None,
                 pbt_data: list = None, bc_data: list = None,
                 bpc_data: list = None, gt: list = None,
                 ylabel: str = '', xlabel: str = '', title: str = None,
                 filename: str = None, subdir: str = ''):
        """Plot the data with respect to the number of tosses. At each number
        of tosses, there could be multiple (but not too many) data points, so
        we scatter them to show the spread and plot the mean as a line. If
        there are a lot of points per number of tosses,
        _do_confidence_interval_plot might be better."""
        # Skip individual object plots, but not if comparing against GT.
        if not self.do_objects and gt is None:
            return

        string = None

        fig = plt.figure()
        ax = plt.gca()

        if bp_data is not None and len(bp_data[0]) > 0:
            x, y, _, _ = xs_and_ys_to_x_mean_lower_upper(bp_data[0], bp_data[1])
            ax.plot(x, y, linewidth=LINEWIDTH, color=BSDF_PLL_COLOR,
                    label=BSDF_PLL_LABEL)
            ax.scatter(bp_data[0], bp_data[1], s=MARKERSIZE,
                    color=BSDF_PLL_COLOR, label='_')
        if n_data is not None and len(n_data[0]) > 0:
            x, y, _, _ = xs_and_ys_to_x_mean_lower_upper(n_data[0], n_data[1])
            ax.plot(x, y, linewidth=LINEWIDTH, color=NERF_ON_COLOR,
                    label=NERF_ON_LABEL)
            ax.scatter(n_data[0], n_data[1], s=MARKERSIZE,
                    color=NERF_ON_COLOR, label='_')
        if bo_data is not None and len(bo_data[0]) > 0:
            x, y, _, _ = xs_and_ys_to_x_mean_lower_upper(bo_data[0], bo_data[1])
            ax.plot(x, y, linewidth=LINEWIDTH, color=BSDF_ONLY_COLOR,
                    label=BSDF_ONLY_LABEL)
            ax.scatter(bo_data[0], bo_data[1], s=MARKERSIZE,
                    color=BSDF_ONLY_COLOR, label='_')
        if pv_data is not None and len(pv_data[0]) > 0:
            x, y, _, _ = xs_and_ys_to_x_mean_lower_upper(pv_data[0], pv_data[1])
            ax.plot(x, y, linewidth=LINEWIDTH, color=PLL_VISION_COLOR,
                    label=PLL_VISION_LABEL)
            ax.scatter(pv_data[0], pv_data[1], s=MARKERSIZE,
                    color=PLL_VISION_COLOR, label='_')
        if ps_data is not None and len(ps_data[0]) > 0:
            x, y, _, _ = xs_and_ys_to_x_mean_lower_upper(ps_data[0], ps_data[1])
            ax.plot(x, y, linewidth=LINEWIDTH, color=PLL_SIZE_COLOR,
                    label=PLL_SIZE_LABEL)
            ax.scatter(ps_data[0], ps_data[1], s=MARKERSIZE,
                    color=PLL_SIZE_COLOR, label='_')
        if pbb_data is not None and len(pbb_data[0]) > 0:
            x, y, _, _ = xs_and_ys_to_x_mean_lower_upper(pbb_data[0], pbb_data[1])
            ax.plot(x, y, linewidth=LINEWIDTH, color=PLL_BLIND_B_COLOR,
                    label=PLL_BLIND_B_LABEL)
            ax.scatter(pbb_data[0], pbb_data[1], s=MARKERSIZE,
                    color=PLL_BLIND_B_COLOR, label='_')
        if pbt_data is not None and len(pbt_data[0]) > 0:
            x, y, _, _ = xs_and_ys_to_x_mean_lower_upper(pbt_data[0], pbt_data[1])
            ax.plot(x, y, linewidth=LINEWIDTH, color=PLL_BLIND_T_COLOR,
                    label=PLL_BLIND_T_LABEL)
            ax.scatter(pbt_data[0], pbt_data[1], s=MARKERSIZE,
                    color=PLL_BLIND_T_COLOR, label='_')
        if bc_data is not None and len(bc_data[0]) > 0:
            x, y, _, _ = xs_and_ys_to_x_mean_lower_upper(bc_data[0], bc_data[1])
            ax.plot(x, y, linewidth=LINEWIDTH, color=BSDF_CONVEX_COLOR,
                    label=BSDF_CONVEX_LABEL)
            ax.scatter(bc_data[0], bc_data[1], s=MARKERSIZE,
                    color=BSDF_CONVEX_COLOR, label='_')
        if bpc_data is not None and len(bpc_data[0]) > 0:
            ax.plot(x, y, linewidth=LINEWIDTH, color=BSDF_PLL_CONVEX_COLOR,
                    label=BSDF_PLL_CONVEX_LABEL)
            ax.scatter(bpc_data[0], bpc_data[1], s=MARKERSIZE,
                    color=BSDF_PLL_CONVEX_COLOR, label='_')
        if gt is not None:
            ax.hlines(np.mean(gt), xmin=0.5, xmax=np.max(bp_data[0])+0.5,
                      color='black', linestyle='--', linewidth=LINEWIDTH,
                      label=GT_LABEL)
            if filename in PLOTS_TO_PRINT:
                string = ''
                string = self._add_to_string(
                    string, BSDF_PLL_LABEL, compare_with=bp_data[1],
                    x=np.ones_like(bp_data[1]), y=bp_data[1],
                    l=np.zeros_like(bp_data[1]), u=np.zeros_like(bp_data[1]),
                    ys=bp_data[1])
                string = self._add_to_string(
                    string, BSDF_ONLY_LABEL, compare_with=bp_data[1],
                    x=np.ones_like(bo_data[1]), y=bo_data[1],
                    l=np.zeros_like(bo_data[1]), u=np.zeros_like(bo_data[1]),
                    ys=bo_data[1])
                string = self._add_to_string(
                    string, PLL_VISION_LABEL, compare_with=bp_data[1],
                    x=np.ones_like(pv_data[1]), y=pv_data[1],
                    l=np.zeros_like(pv_data[1]), u=np.zeros_like(pv_data[1]),
                    ys=pv_data[1])
                string = self._add_to_string(
                    string, GT_LABEL, compare_with=bp_data[1],
                    x=np.ones_like(gt), y=gt, l=np.zeros_like(gt),
                    u=np.zeros_like(gt), ys=gt)

        ax.set_xlim(0.5, np.max(bp_data[0])+0.5)
        x_markers = bp_data[0]
        ax.set_ylim(0, None)

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)

        self._beautify_plot(fig, ax, x_markers, auc='auc' in filename)

        filename += '.png' if not filename.endswith('.png') else ''
        file_utils.assure_created(op.join(self.plot_dir, subdir))
        fig_path = op.join(self.plot_dir, subdir, filename)
        fig.savefig(fig_path, dpi=100)
        plt.close()

        # Save the string to a text file as well.
        if string is not None:
            str_filepath = fig_path.replace('.png', '.txt')
            with open(str_filepath, 'w') as txt_file:
                txt_file.write(string)
            print(f'Wrote to {str_filepath}')

    def _add_to_string(self, string, data_name, x, y, l, u, ys, compare_with):
        pm = [(ui-li)/2 for ui, li in zip(u, l)]

        # Get a total mean and confidence interval.
        same_xs = [0] * len(ys)
        _, toty, totl, totu = xs_and_ys_to_x_mean_lower_upper(same_xs, ys)

        # Compare with the other data, using Welch's t-test.
        t, p = welchs_t_test(ys, compare_with)

        # Compute standard deviation.
        std_dev = np.std(y)

        string += f'{data_name}:\n'
        string += f'{x=}\n'
        string += f'y = '
        for yi, pmi in zip(y, pm):
            string += f' & ${yi:.1f} \pm {pmi:.1f}$'
        string += f'\nStandard deviation: {std_dev=}'
        string += f'\nCombined:  ${toty[0]:.1f} \pm {(totu[0]-totl[0])/2:.1f}$'
        string += f'\nWelchs t-test: {t=}, {p=}'
        string += f'\n\n'
        return string

    def _do_confidence_interval_plot(
            self, bp_data: list = None, n_data: list = None,
            bo_data: list = None, pv_data: list = None, ps_data: list = None,
            pbb_data: list = None, pbt_data: list = None, bc_data: list = None,
            bpc_data: list = None, gt: list = None, ylabel: str = '', xlabel: str = '',
            title: str = None, filename: str = None, subdir: str = '',
            save_to_txt: bool = False):
        """Plot the data with respect to the number of tosses. At each number
        of tosses, we assume there are more than one data points, so that the
        confidence intervals can be plotted."""

        if filename in PLOTS_TO_PRINT:
            save_to_txt = True
            print(f' -> Forcing {filename} to print to txt.')

        fig = plt.figure()
        ax = plt.gca()

        if save_to_txt:
            data_str = ''

        if bp_data is not None and len(bp_data[0]) > 0:
            # Convert the data to mean/lower/upper.
            x, y, l, u = xs_and_ys_to_x_mean_lower_upper(bp_data[0], bp_data[1])
            ax.plot(x, y, linewidth=LINEWIDTH,
                    color=BSDF_PLL_COLOR, label=BSDF_PLL_LABEL)
            ax.fill_between(x, l, u, alpha=0.3, color=BSDF_PLL_COLOR)
            if save_to_txt:
                data_str = self._add_to_string(
                    data_str, 'bp_data', x, y, l, u, bp_data[1],
                    compare_with=bp_data[1])
        if n_data is not None and len(n_data[0]) > 0:
            x, y, l, u = xs_and_ys_to_x_mean_lower_upper(n_data[0], n_data[1])
            ax.plot(x, y, linewidth=LINEWIDTH,
                    color=NERF_ON_COLOR, label=NERF_ON_LABEL)
            ax.fill_between(x, l, u, alpha=0.3, color=NERF_ON_COLOR)
            if save_to_txt:
                data_str = self._add_to_string(
                    data_str, 'n_data', x, y, l, u, n_data[1],
                    compare_with=bp_data[1])
        if bo_data is not None and len(bo_data[0]) > 0:
            x, y, l, u = xs_and_ys_to_x_mean_lower_upper(bo_data[0], bo_data[1])
            ax.plot(x, y, linewidth=LINEWIDTH,
                    color=BSDF_ONLY_COLOR, label=BSDF_ONLY_LABEL)
            ax.fill_between(x, l, u, alpha=0.3, color=BSDF_ONLY_COLOR)
            if save_to_txt:
                data_str = self._add_to_string(
                    data_str, 'bo_data', x, y, l, u, bo_data[1],
                    compare_with=bp_data[1])
        if pv_data is not None and len(pv_data[0]) > 0:
            x, y, l, u = xs_and_ys_to_x_mean_lower_upper(pv_data[0], pv_data[1])
            ax.plot(x, y, linewidth=LINEWIDTH,
                    color=PLL_VISION_COLOR, label=PLL_VISION_LABEL)
            ax.fill_between(x, l, u, alpha=0.3, color=PLL_VISION_COLOR)
            if save_to_txt:
                data_str = self._add_to_string(
                    data_str, 'pv_data', x, y, l, u, pv_data[1],
                    compare_with=bp_data[1])
        if ps_data is not None and len(ps_data[0]) > 0:
            x, y, l, u = xs_and_ys_to_x_mean_lower_upper(ps_data[0], ps_data[1])
            ax.plot(x, y, linewidth=LINEWIDTH,
                    color=PLL_SIZE_COLOR, label=PLL_SIZE_LABEL)
            ax.fill_between(x, l, u, alpha=0.3, color=PLL_SIZE_COLOR)
            if save_to_txt:
                data_str = self._add_to_string(
                    data_str, 'ps_data', x, y, l, u, ps_data[1],
                    compare_with=bp_data[1])
        if pbb_data is not None and len(pbb_data[0]) > 0:
            x, y, l, u = xs_and_ys_to_x_mean_lower_upper(pbb_data[0], pbb_data[1])
            ax.plot(x, y, linewidth=LINEWIDTH,
                    color=PLL_BLIND_B_COLOR, label=PLL_BLIND_B_LABEL)
            ax.fill_between(x, l, u, alpha=0.3, color=PLL_BLIND_B_COLOR)
            if save_to_txt:
                data_str = self._add_to_string(
                    data_str, 'pbb_data', x, y, l, u, pbb_data[1],
                    compare_with=bp_data[1])
        if pbt_data is not None and len(pbt_data[0]) > 0:
            x, y, l, u = xs_and_ys_to_x_mean_lower_upper(pbt_data[0], pbt_data[1])
            ax.plot(x, y, linewidth=LINEWIDTH,
                    color=PLL_BLIND_T_COLOR, label=PLL_BLIND_T_LABEL)
            ax.fill_between(x, l, u, alpha=0.3, color=PLL_BLIND_T_COLOR)
            if save_to_txt:
                data_str = self._add_to_string(
                    data_str, 'pbt_data', x, y, l, u, pbt_data[1],
                    compare_with=bp_data[1])
        if bc_data is not None and len(bc_data[0]) > 0:
            x, y, l, u = xs_and_ys_to_x_mean_lower_upper(bc_data[0], bc_data[1])
            ax.plot(x, y, linewidth=LINEWIDTH,
                    color=BSDF_CONVEX_COLOR, label=BSDF_CONVEX_LABEL)
            ax.fill_between(x, l, u, alpha=0.3, color=BSDF_CONVEX_COLOR)
            if save_to_txt:
                data_str = self._add_to_string(
                    data_str, 'bc_data', x, y, l, u, bc_data[1],
                    compare_with=bp_data[1])
        if bpc_data is not None and len(bpc_data[0]) > 0:
            x, y, l, u = xs_and_ys_to_x_mean_lower_upper(bpc_data[0], bpc_data[1])
            ax.plot(x, y, linewidth=LINEWIDTH,
                    color=BSDF_PLL_CONVEX_COLOR, label=BSDF_PLL_CONVEX_LABEL)
            ax.fill_between(x, l, u, alpha=0.3, color=BSDF_PLL_CONVEX_COLOR)
            if save_to_txt:
                data_str = self._add_to_string(
                    data_str, 'bpc_data', x, y, l, u, bpc_data[1],
                    compare_with=bp_data[1])
        if gt is not None:
            ax.hlines(np.mean(gt), xmin=0.5, xmax=np.max(bp_data[0])+0.5,
                      color='black', linestyle='--', linewidth=LINEWIDTH,
                      label=GT_LABEL)

        ax.set_xlim(0.5, np.max(bp_data[0])+0.5)
        x_markers = bp_data[0]
        ax.set_ylim(0, None)

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)

        self._beautify_plot(fig, ax, x_markers, auc='auc' in filename)

        if 'all_objs_chamfer_distance_v_data_convex_hull' in filename:
            ax.set_ylim(0, 0.1)
        if 'all_objs_chamfer_distance_v_data_full_geometry' in filename:
            ax.set_ylim(0, 0.1)
        if 'all_objs_chamfer_distance_v_data_hull_to_full' in filename:
            ax.set_ylim(0, 0.1)

        filename += '.png' if not filename.endswith('.png') else ''
        file_utils.assure_created(op.join(self.plot_dir, subdir))
        fig_path = op.join(self.plot_dir, subdir, filename)
        fig.savefig(fig_path, dpi=100)
        plt.close()

        # Save the string to a text file as well.
        if save_to_txt:
            str_filepath = fig_path.replace('.png', '.txt')
            with open(str_filepath, 'w') as txt_file:
                txt_file.write(data_str)
            print(f'Wrote to {str_filepath}')

    def _beautify_plot(
            self, fig, ax, x_markers: list, auc: bool = False,
            scatter: bool = False, normalized: bool = False):
        """Perform all the nice formatting on a plot."""
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(NullFormatter())

        ax.tick_params(axis='y', which='minor', labelsize=20)
        ax.tick_params(axis='y', which='major', labelsize=20)

        if auc:
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
            ax.yaxis.set_minor_formatter(FormatStrFormatter("%.0f"))
            ax.set_ylim(0, 110)
        else:
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
            ax.yaxis.set_minor_formatter(FormatStrFormatter("%.1f"))

        if not scatter:
            ax.xaxis.set_major_formatter(NullFormatter())
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.tick_params(axis='x', which='minor', bottom=False, labelsize=20)
            ax.tick_params(axis='x', which='major', bottom=False, labelsize=20)
            ax.set_xticks([])
            ax.set_xticklabels([])
            ax.set_xticks(x_markers)
            ax.set_xticklabels(x_markers)
            ax.xaxis.set_major_formatter(FormatStrFormatter("%.0f"))
            ax.xaxis.grid(True, which='major')
            ax.yaxis.grid(True, which='both')

            fig.set_size_inches(13, 13)

            handles, labels = plt.gca().get_legend_handles_labels()

            plt.legend(handles, labels)
            plt.legend(prop=dict(weight='bold', family='serif'))
            if scatter:
                ax.legend(loc='upper left')

        else:
            vert_xs = [xval + 0.5 for xval in x_markers[:-1]]
            for x in vert_xs:
                plt.axvline(x=x, color='#CCCCCC', linestyle='--', linewidth=2)
            ax.set_xlim(x_markers[0]-0.5, x_markers[-1]+0.5)
            # ax.set_yscale('log')
            if normalized:
                ax.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
                ax.yaxis.set_minor_formatter(FormatStrFormatter("%.0f"))
            else:
                ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
                ax.yaxis.set_minor_formatter(FormatStrFormatter("%.1f"))

            fig.set_size_inches(13, 9)
            plt.subplots_adjust(bottom=0.16)

            # Use fonts for the paper.
            ax.title.set_fontname('serif')
            ax.xaxis.label.set_fontname('serif')
            ax.yaxis.label.set_fontname('serif')
            ax.legend(loc='upper left')

            handles, labels = plt.gca().get_legend_handles_labels()
            plt.legend(handles, labels)
            plt.legend(prop=dict(weight='bold', family='serif'))

            for tick in ax.get_xticklabels():
                tick.set_fontname('serif')
            for tick in ax.get_yticklabels():
                tick.set_fontname('serif')


#######################################################################
@click.group()
def cli():
    pass

# Use 'auc' command to scrub through the results and calculate the AUC for
# tracking-related metrics.
@cli.command('auc')
def process_auc_command():
    # Iterate over every evaluation subdirectory.
    for subdir in os.listdir(file_utils.evaluation_dir()):
        eval_subdir = op.join(file_utils.evaluation_dir(), subdir)
        if not op.exists(op.join(eval_subdir, 'results.yaml')):
            print(f'  Skipping {subdir}.')
            continue

        print(f'Found {subdir}...', end='')

        # Load the results.
        results = file_utils.load_results_yaml_in_subdir(subdir)

        try:
            # Add AUC to the results.
            add_auc_to_results(results)

            # Save the results.
            file_utils.save_results_to_yaml(results, eval_subdir)
            print(f'done.')
        except Exception as e:
            print(f'ISSUE: {e}')


# Use 'gather' command to gather all the results into yaml files, one for
# BundleSDF-PLL, one for BundleSDF-only, and another for PLL-only.
@cli.command('gather')
@click.option('--single-toss', is_flag=True,
              help='Whether to only gather single-toss results.')
@click.option('--exclude-pll', is_flag=True,
                help='Whether to exclude PLL results.')
def process_gather_command(single_toss, exclude_pll):
    # Start dictionaries for each of the four result types.
    bsdf_pll_results = {}            # 02_2
    nerf_on_results = {}             # 03_2
    bsdf_only_results = {}           # bsdf 00_1 or 00-grid_1
    bsdf_octree_results = {}         # bsdf 00-o-grid_1
    bsdf_pll_convex_results = {}     # 02-cvwo_2 or 02-cvwo2_2

    pll_vision_results = {}          # pll 00_1
    pll_size_results = {}            # pll 04_1 (or pll 05_1)
    pll_blind_b_results = {}         # pll 07_1
    pll_blind_t_results = {}         # pll 09_0

    bsdf_convex_results = {}         # bsdf 00-cvwo_1 or bsdf 00-cvw_1
    bsdf_convex_occ_results = {}     # bsdf 00-cvwo-occ_1
    bsdf_pll_convex_occ_results = {} # bsdf 00-cvwo-occ2x2_1

    bsdf_pll_robotocc_results = {}   # bsdf 00-t11 (or 00-t09d_1)
    pll_robotocc_results = {}        # pll t11 (or t09d)
    bsdf_robotocc_results = {}       # bsdf 00_1
    gt_robotocc_results = {}

    gt_results = {}

    # Iterate over every evaluation subdirectory.
    for subdir in os.listdir(file_utils.evaluation_dir()):
        eval_subdir = op.join(file_utils.evaluation_dir(), subdir)
        if not op.exists(op.join(eval_subdir, 'results.yaml')):
            print(f'  No results in {subdir}.')
            continue

        toss_id = subdir.split('_')[1]

        if single_toss and len(toss_id) > 1:
            print(f'  Only gathering single-toss, skipping {subdir}')
            continue
        if exclude_pll and 'pll' in subdir:
            print(f'  Excluding PLL, skipping {subdir}')
            continue

        if 'bsdf' in subdir and subdir.endswith('_1') and \
            '00-o-grid_' in subdir:
            add_to_results = bsdf_octree_results
        elif 'bsdf' in subdir and subdir.endswith('_1') and \
            '00-cvwo-occ2x2-grid_' in subdir:
            add_to_results = bsdf_pll_convex_occ_results
        elif 'bsdf' in subdir and subdir.endswith('_1') and \
            '00-cvwo-occ-grid_' in subdir:
            add_to_results = bsdf_convex_occ_results
        elif 'bsdf' in subdir and subdir.endswith('_1') and \
            '00-cvwo-grid_' in subdir:
            add_to_results = bsdf_convex_results
        elif 'bsdf' in subdir and subdir.endswith('_1') and \
            ('00-t09d_' in subdir or '00-t11_' in subdir) and \
            'robotocc' in subdir:
            # Prioritize t11 results over t09d, if they exist.
            subdir_t11 = subdir.replace('00-t09d_', '00-t11_')
            if '00-t09d_' in subdir and subdir_t11 in os.listdir(
                file_utils.evaluation_dir()):
                continue
            add_to_results = bsdf_pll_robotocc_results
        elif 'pll' in subdir and subdir.endswith('_1') and \
            ('t09d_' in subdir or 't11_' in subdir) and 'robotocc' in subdir:
            # Prioritize t11 results over t09d, if they exist.
            subdir_t11 = subdir.replace('t09d_', 't11_')
            if 't09d_' in subdir and subdir_t11 in os.listdir(
                file_utils.evaluation_dir()):
                continue
            add_to_results = pll_robotocc_results
        elif 'bsdf' in subdir and subdir.endswith('_1') \
            and '00_' in subdir and 'robotocc' in subdir:
            add_to_results = bsdf_robotocc_results
        elif 'bsdf' in subdir and subdir.endswith('_1') and '00_' in subdir:
            add_to_results = bsdf_only_results
        elif 'pll' in subdir and subdir.endswith('_1') and '00_' in subdir:
            add_to_results = pll_vision_results
        elif 'pll' in subdir and subdir.endswith('_1') and '04_' in subdir:
            add_to_results = pll_size_results
        elif 'pll' in subdir and subdir.endswith('_1') and '07_' in subdir:
            add_to_results = pll_blind_b_results
        elif 'pll' in subdir and subdir.endswith('_0') and '09_' in subdir:
            add_to_results = pll_blind_t_results
        elif '02_' in subdir and subdir.endswith('_2'):
            add_to_results = bsdf_pll_results
        elif 'bsdf' in subdir and subdir.endswith('_1') \
            and '00-cvwo012525_' in subdir and 'robotocc' not in subdir:
            add_to_results = bsdf_pll_convex_results
        elif '03_' in subdir and subdir.endswith('_2'):
            add_to_results = nerf_on_results
        elif subdir.endswith('_GT'):
            print(f'Found GT {subdir}...', end='')
            experiment_results = file_utils.load_results_yaml_in_subdir(subdir)
            add_experiment_to_gt_results(experiment_results, gt_results)
            print(f'done.')
            continue
        else:
            print(f'  Skipping {subdir}')
            continue

        print(f'Found {subdir}...', end='')

        # Add the experiment's results to the overall results.
        experiment_results = file_utils.load_results_yaml_in_subdir(subdir)
        add_experiment_to_overall_results(experiment_results, add_to_results)

        print(f'done.')

    # Iterate over every robot dynamics prediction subdirectory.
    dpqs = []
    for subdir in os.listdir(file_utils.robot_dynamics_dir()):
        if not op.isdir(op.join(file_utils.robot_dynamics_dir(), subdir)):
            print(f'  No robot dynamics results in non-directory {subdir}')
            continue

        # Check that the directory contains the predicted robot files.
        if np.all(np.array([
            op.exists(op.join(file_utils.robot_dynamics_dir(), subdir, f)) \
            for f in robot_dynamics_predictions.FILES_TO_GENERATE
        ])):
            dpqs.append(
                robot_dynamics_predictions.DynamicsPredictionQuantifier(subdir))
            print(f'=== Prepared to analyze {subdir} ===')
        else:
            print(f'{subdir} did not have all files.')

    cdm = robot_dynamics_predictions.ConglomeratedDynamicsMetrics(dpqs)
    cdm.export_statistics()
    add_all_robot_predictions_to_overall_results(
        vysics_results=bsdf_pll_robotocc_results,
        bsdf_results=bsdf_robotocc_results,
        pll_results=pll_robotocc_results,
        gt_results=gt_robotocc_results
    )

    # Save the collected results.
    file_utils.save_results_to_yaml(
        bsdf_pll_results, file_utils.evaluation_dir(), filename='bsdf_pll.yaml')
    file_utils.save_results_to_yaml(
        nerf_on_results, file_utils.evaluation_dir(), filename='nerf_on.yaml')
    file_utils.save_results_to_yaml(
        bsdf_octree_results, file_utils.evaluation_dir(),
        filename='bsdf_octree.yaml')
    file_utils.save_results_to_yaml(
        bsdf_only_results, file_utils.evaluation_dir(),
        filename='bsdf_only.yaml')
    file_utils.save_results_to_yaml(
        bsdf_convex_results, file_utils.evaluation_dir(),
        filename='bsdf_convex.yaml')
    file_utils.save_results_to_yaml(
        bsdf_pll_convex_results, file_utils.evaluation_dir(),
        filename='bsdf_pll_convex.yaml')
    file_utils.save_results_to_yaml(
        bsdf_convex_occ_results, file_utils.evaluation_dir(),
        filename='bsdf_convex_occ.yaml')
    file_utils.save_results_to_yaml(
        bsdf_pll_convex_occ_results, file_utils.evaluation_dir(),
        filename='bsdf_pll_convex_occ.yaml')
    file_utils.save_results_to_yaml(
        bsdf_pll_robotocc_results, file_utils.evaluation_dir(),
        filename='bsdf_pll_robotocc.yaml')
    file_utils.save_results_to_yaml(
        bsdf_robotocc_results, file_utils.evaluation_dir(),
        filename='bsdf_robotocc.yaml')
    file_utils.save_results_to_yaml(
        gt_robotocc_results, file_utils.evaluation_dir(),
        filename='gt_robotocc.yaml')
    file_utils.save_results_to_yaml(
        pll_robotocc_results, file_utils.evaluation_dir(),
        filename='pll_robotocc.yaml')
    if not exclude_pll:
        file_utils.save_results_to_yaml(
            pll_vision_results, file_utils.evaluation_dir(),
            filename='pll_vision.yaml')
        file_utils.save_results_to_yaml(
            pll_size_results, file_utils.evaluation_dir(),
            filename='pll_size.yaml')
        file_utils.save_results_to_yaml(
            pll_blind_b_results, file_utils.evaluation_dir(),
            filename='pll_blind_b.yaml')
        file_utils.save_results_to_yaml(
            pll_blind_t_results, file_utils.evaluation_dir(),
            filename='pll_blind_t.yaml')
    file_utils.save_results_to_yaml(
        gt_results, file_utils.evaluation_dir(), filename='gt.yaml')


# Use 'plot' command to load the previously generated yaml files with results
# and to generate plots with them.
PLOT_TRACKING = False
PLOT_TRACKING_SCATTERS = False
PLOT_DYNAMICS = True
PLOT_GEOMETRY = False
PLOT_GEOMETRY_SCATTERS = True
PLOT_GT_COMPARISON = False
@cli.command('plot')
@click.option('--do-objects/--skip-objects',
              type=bool, default=False,
              help='Whether to plot object-level results or just aggregates.')
@click.option('--remote',
              is_flag=True,
              help='if running on a remote server, use virtual display to ' + \
                   'accelerate rendering')
def process_plot_command(do_objects: bool, remote: bool):
    # Load the gathered results.
    bsdf_pll_results = file_utils.load_gathered_results_yaml('bsdf_pll.yaml')
    nerf_on_results = file_utils.load_gathered_results_yaml('nerf_on.yaml')
    bsdf_only_results = file_utils.load_gathered_results_yaml('bsdf_only.yaml')
    bsdf_octree_results = file_utils.load_gathered_results_yaml(
        'bsdf_octree.yaml')
    pll_vision_results = file_utils.load_gathered_results_yaml(
        'pll_vision.yaml')
    pll_size_results = file_utils.load_gathered_results_yaml(
        'pll_size.yaml')
    pll_blind_b_results = file_utils.load_gathered_results_yaml(
        'pll_blind_b.yaml')
    pll_blind_t_results = file_utils.load_gathered_results_yaml(
        'pll_blind_t.yaml')
    bsdf_convex_results = file_utils.load_gathered_results_yaml(
        'bsdf_convex.yaml')
    bsdf_pll_convex_results = file_utils.load_gathered_results_yaml(
        'bsdf_pll_convex.yaml')
    bsdf_convex_occ_results = file_utils.load_gathered_results_yaml(
        'bsdf_convex_occ.yaml')
    bsdf_pll_convex_occ_results = file_utils.load_gathered_results_yaml(
        'bsdf_pll_convex_occ.yaml')
    bsdf_pll_robotocc_results = file_utils.load_gathered_results_yaml(
        'bsdf_pll_robotocc.yaml')
    bsdf_robotocc_results = file_utils.load_gathered_results_yaml(
        'bsdf_robotocc.yaml')
    pll_robotocc_results = file_utils.load_gathered_results_yaml(
        'pll_robotocc.yaml')
    gt_robotocc_results = file_utils.load_gathered_results_yaml(
        'gt_robotocc.yaml')

    # Ground truth results.
    gt_results = file_utils.load_gathered_results_yaml('gt.yaml')

    # Load an empty results dictionary for checking which metrics are valid for
    # which category/against which tracking.
    empty_results = file_utils.load_empty_results_yaml()

    change_display = False
    if remote and os.environ.get('DISPLAY') != ':99':
        change_display = True
        # Run Xvfb to create a virtual display.
        print("Running Xvfb (virtual display) for rendering.")
        import subprocess
        xvfb_process = subprocess.Popen(
            ['Xvfb', ':99', '-screen', '0', '640x480x24'])
        print(f"Changing display environment from " + \
                f"{os.environ['DISPLAY']} variable to :99")
        old_display = os.environ['DISPLAY']
        os.environ['DISPLAY'] = ':99'

    results_plotter = ResultsPlotter(
        bsdf_pll_results=bsdf_pll_results,
        nerf_on_results=nerf_on_results,
        bsdf_only_results=bsdf_only_results,
        bsdf_octree_results=bsdf_octree_results,
        pll_vision_results=pll_vision_results,
        pll_size_results=pll_size_results,
        pll_blind_b_results=pll_blind_b_results,
        pll_blind_t_results=pll_blind_t_results,
        bsdf_convex_results=bsdf_convex_results,
        bsdf_pll_convex_results=bsdf_pll_convex_results,
        bsdf_convex_occ_results=bsdf_convex_occ_results,
        bsdf_pll_convex_occ_results=bsdf_pll_convex_occ_results,
        bsdf_pll_robotocc_results=bsdf_pll_robotocc_results,
        bsdf_robotocc_results=bsdf_robotocc_results,
        gt_results=gt_results,
        do_objects=do_objects
    )

    # results_plotter.plot_tracking_geometry_correlation(
    #     [BSDF_CONVEX_OCC, BSDF_PLL_CONVEX_OCC])
    # results_plotter.plot_tracking_geometry_diff_correlation(
    #     BSDF_CONVEX_OCC, [BSDF_CONVEX_OCC, BSDF_PLL_CONVEX_OCC])
    # results_plotter.plot_tracking_geometry_correlation(
    #     [BSDF_CONVEX, BSDF_PLL_CONVEX])
    # results_plotter.plot_tracking_geometry_diff_correlation(
    #     BSDF_CONVEX, [BSDF_CONVEX, BSDF_PLL_CONVEX])

    for metric in ERROR_LABELS.keys():
        # # Tracking.
        # for trajectory in ['full', 'toss']:
        #     if metric in \
        #         empty_results['tracking_metrics']['against_tagslam'].keys():
        #         if PLOT_TRACKING:
        #             print(f'Plotting TagSLAM {trajectory}, {metric}')
        #             results_plotter.plot_tagslam_tracking_error_vs_data(
        #                 trajectory, metric)
        #         if PLOT_TRACKING_SCATTERS:
        #             print(f'Plotting TagSLAM {trajectory} scatters, {metric}')
        #             results_plotter.plot_object_tagslam_tracking_scatter(
        #                 trajectory, metric, [BSDF_CONVEX, BSDF_PLL_CONVEX], False)
        #                 # trajectory, metric, [BSDF_CONVEX_OCC, BSDF_PLL_CONVEX_OCC], False)
        #     if metric in \
        #         empty_results['tracking_metrics']['against_bundlesdf'].keys():
        #         if PLOT_TRACKING:
        #             print(f'Plotting BundleSDF {trajectory}, {metric}')
        #             results_plotter.plot_bundlesdf_tracking_error_vs_data(
        #                 trajectory, metric)

        # # Dynamics.
        # for toss_subset in ['all_tosses', 'training_tosses', 'unseen_tosses']:
        #     for dynamics_category in [
        #         'dynamics_rollout_metrics', 'dynamics_single_step_metrics']:
        #         if metric in \
        #             empty_results[dynamics_category]['against_tagslam'
        #             ].keys():
        #             if 'auc' in metric or 'training' in toss_subset:
        #                 print(f'Manually skipping some metrics')
        #                 continue
        #             if PLOT_DYNAMICS:
        #                 print(f'Plotting TagSLAM {toss_subset}, ' + \
        #                     f'{dynamics_category}, {metric}')
        #                 results_plotter.plot_tagslam_dynamics_error_vs_data(
        #                     toss_subset, dynamics_category, metric)
        # for dynamics_category in [
        #     'dynamics_rollout_metrics', 'dynamics_single_step_metrics']:
        #     if metric in \
        #         empty_results[dynamics_category]['against_bundlesdf'
        #         ].keys():
        #         toss_subset = 'training_tosses'
        #         if 'auc' in metric or 'training' in toss_subset:
        #             print(f'Manually skipping some metrics')
        #             continue
        #         if PLOT_DYNAMICS:
        #             print(f'Plotting BundleSDF {toss_subset}, ' + \
        #                 f'{dynamics_category}, {metric}')
        #             results_plotter.plot_bundlesdf_dynamics_error_vs_data(
        #                 toss_subset, dynamics_category, metric)

        # # Ground truth dynamics comparison.
        # for toss_subset in ['all_tosses', 'training_tosses', 'unseen_tosses']:
        #     if metric in empty_results['dynamics_rollout_metrics'][
        #         'against_tagslam'].keys():
        #         if PLOT_GT_COMPARISON:
        #             print(f'Plotting ground truth dynamics comparison, ' + \
        #                   f'{toss_subset}, {metric}')
        #             results_plotter.plot_gt_dynamics_comparison(
        #                 toss_subset, metric)

        # Geometry.
        if metric in \
            empty_results['geometry_metrics']['full_geometry'].keys():
            if PLOT_GEOMETRY:
                print(f'Plotting full geometry {metric}')
                results_plotter.plot_geometry_error_vs_data(
                    'full_geometry', metric)
            if PLOT_GEOMETRY_SCATTERS:
                print(f'Plotting full geometry geometry scatters {metric}')
                results_plotter.plot_object_scatter(
                    # 'full_geometry', metric, [BSDF_ONLY, BSDF_OCTREE, BSDF_CONVEX, BSDF_PLL_CONVEX], True)
                    # 'full_geometry', metric, [BSDF_CONVEX_OCC, BSDF_PLL_CONVEX_OCC], False)
                    # 'full_geometry', metric, [BSDF_ROBOTOCC, BSDF_PLL_ROBOTOCC], False)
                    # 'convex_hull', metric, [BSDF_CONVEX, BSDF_PLL_CONVEX], False)
                    'full_geometry', metric, [BSDF_ROBOTOCC, BSDF_PLL_ROBOTOCC], False)

        ### hull_to_full not plotted because not all experiments are evaluated with hull_to_full
        ### and preliminary results show that hull_to_full does not outperform full_geometry.

        # Dynamics.
        if metric in \
            empty_results['robot_dynamics_rollout_metrics'].keys():
            if PLOT_DYNAMICS:
                print(f'Plotting full geometry dynamics scatters {metric}')
                results_plotter.plot_object_scatter(
                    # 'full_geometry', metric, [BSDF_ONLY, BSDF_OCTREE, BSDF_CONVEX, BSDF_PLL_CONVEX], True)
                    # 'full_geometry', metric, [BSDF_CONVEX_OCC, BSDF_PLL_CONVEX_OCC], False)
                    # 'full_geometry', metric, [BSDF_ROBOTOCC, BSDF_PLL_ROBOTOCC], False)
                    # 'convex_hull', metric, [BSDF_CONVEX, BSDF_PLL_CONVEX], False)
                    'full_geometry', metric, [BSDF_ROBOTOCC, BSDF_PLL_ROBOTOCC], False)


    # Robot dynamics prediction plots.
    cdm = robot_dynamics_predictions.cdm_from_overall_results(
        vysics_results=bsdf_pll_robotocc_results,
        bsdf_results=bsdf_robotocc_results,
        pll_results=pll_robotocc_results,
        gt_results=gt_robotocc_results,
        save_dir=op.join(results_plotter.plot_dir, 'robot_dynamics')
    )
    cdm.plot()

    if remote and change_display:
        # Close the virtual display.
        print(f"Closing Xvfb (virtual display).")
        xvfb_process.terminate()
        xvfb_process.wait()
        os.environ['DISPLAY'] = old_display
        print(f"Recovered {os.environ['DISPLAY']=}.")

if __name__ == '__main__':
    cli()
