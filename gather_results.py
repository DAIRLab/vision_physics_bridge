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

import eval_utils, file_utils

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
    'cube': 100 * 0.1048
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
    #'f_score': TODO this isn't implemented so exclude from dictionary
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
    #'f_score': TODO this isn't implemented so exclude from dictionary
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
for i in range(31, 100):
    T_SCORE_PER_DOF[i] = 1.960


# pll_vision_results
# pll_size_results
# pll_blind_b_results
# pll_blind_t_results

BSDF_PLL_COLOR = '#7030a0'  #'#398537'  #'#8c59b3'
NERF_ON_COLOR = '#92668d'
BSDF_ONLY_COLOR = '#f9a602'  #'#a1b8e1'  #'#8eaadb'  #'#4472c4'  #'cd5b45'
PLL_VISION_COLOR = '#833785'
PLL_SIZE_COLOR = '#4a0042'
PLL_BLIND_B_COLOR = '#92668d'
PLL_BLIND_T_COLOR = '##95001a'

BSDF_PLL_LABEL = 'Vysics'
NERF_ON_LABEL = 'BundleSDF-PLL NeRF Online'
BSDF_ONLY_LABEL = 'BundleSDF [1]'
PLL_VISION_LABEL = 'PLL with Vision Supervision'
PLL_SIZE_LABEL = 'PLL with Vision-Supervised Size Only'
PLL_BLIND_B_LABEL = 'Blind PLL on Vision-Based Tracking'
PLL_BLIND_T_LABEL = 'Blind PLL on Fiducial-Based Tracking'

LINEWIDTH = 5
MARKERSIZE = 100


PLOTS_TO_PRINT = [
    'all_tagged_objs_penetration_true_geom_predicted_traj_tagslam_v_data_unseen_tosses_dynamics_rollout_metrics',
    'all_tagged_objs_adds_error_tagslam_v_data_unseen_tosses_dynamics_rollout_metrics',
    'all_tagged_objs_add_error_tagslam_v_data_unseen_tosses_dynamics_rollout_metrics',
    'all_tagged_objs_add_error_v_data_full.txt',
    'all_tagged_objs_adds_error_v_data_full.txt'
]


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
    if len(ys) <= 1:
        return None, None, None

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
    object = vision_asset.split('_')[0]
    tag_key = 'tagless_objects' if object in file_utils.TAGLESS_OBJECTS else \
        'tagged_objects'

    start_toss = int(vision_asset.split('_')[-1].split('-')[0])
    end_toss = start_toss if '-' not in vision_asset else \
        int(vision_asset.split('-')[1])
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

        elif category == 'tracking_metrics':
            # Tracking and dynamics metrics need to be conglomerated by toss.
            # Iterate over every comparison against, e.g. against_bundlesdf.
            for against, subsubresults in exp_subresults.items():
                # Iterate over every metric, e.g. position_error.
                for metric, traj_results in subsubresults.items():

                    # Iterate over every trajectory, e.g. toss_1.
                    toss_means = []
                    toss_aucs = []
                    for traj_name, reported_errors in traj_results.items():
                        # If full trajectory, can use the mean and AUC directly.
                        if traj_name == 'full':
                            subresults = recursive_dict_add(
                                d=subresults,
                                val=reported_errors['mean'],
                                keys=[category, 'full', against, metric, 'mean']
                            )
                            subresults = recursive_dict_add(
                                d=subresults,
                                val=reported_errors['auc'],
                                keys=[category, 'full', against, metric,
                                      'mean_auc']
                            )
                            continue

                        # Otherwise, need to conglomerate by toss.
                        toss_means.append(reported_errors['mean'])
                        toss_aucs.append(reported_errors['auc'])

                    # Add the mean and AUC for the tosses.
                    subresults = recursive_dict_add(
                        d=subresults,
                        val=np.mean(toss_means).item(),
                        keys=[category, 'toss', against, metric, 'mean']
                    )
                    subresults = recursive_dict_add(
                        d=subresults,
                        val=np.mean(toss_aucs).item(),
                        keys=[category, 'toss', against, metric, 'mean_auc']
                    )
            continue

        # Dynamics metrics need to be split by training and evaluation tosses.
        # Tracking and dynamics metrics need to be conglomerated by toss.
        # Iterate over every comparison against, e.g. against_bundlesdf.
        for against, subsubresults in exp_subresults.items():
            # Iterate over every metric, e.g. position_error.
            for metric, traj_results in subsubresults.items():

                # Iterate over every trajectory, e.g. toss_1.
                training_toss_means = []
                training_toss_aucs = []
                test_toss_means = []
                test_toss_aucs = []
                for traj_name, reported_errors in traj_results.items():
                    toss_num = int(traj_name.split('toss_')[-1])
                    if toss_num in range(start_toss, end_toss + 1):
                        training_toss_means.append(reported_errors['mean'])
                        training_toss_aucs.append(reported_errors['auc'])
                    else:
                        test_toss_means.append(reported_errors['mean'])
                        test_toss_aucs.append(reported_errors['auc'])

                # Add the mean and AUC for the tosses.
                subresults = recursive_dict_add(
                    d=subresults,
                    val=None if None in training_toss_means else \
                        np.mean(training_toss_means).item(),
                    keys=[category, 'training_tosses', against, metric,
                    'mean']
                )
                subresults = recursive_dict_add(
                    d=subresults,
                    val=None if None in training_toss_aucs else \
                        np.mean(training_toss_aucs).item(),
                    keys=[category, 'training_tosses', against, metric,
                    'mean_auc']
                )
                if against == 'against_tagslam':
                    subresults = recursive_dict_add(
                        d=subresults,
                        val=None if None in test_toss_means else \
                            np.mean(test_toss_means).item(),
                        keys=[category, 'unseen_tosses', against, metric,
                        'mean']
                    )
                    subresults = recursive_dict_add(
                        d=subresults,
                        val=None if None in test_toss_aucs else \
                            np.mean(test_toss_aucs).item(),
                        keys=[category, 'unseen_tosses', against, metric,
                        'mean_auc']
                    )
                    subresults = recursive_dict_add(
                        d=subresults,
                        val=None if None in training_toss_means+test_toss_means\
                            else np.mean(
                                training_toss_means + test_toss_means).item(),
                        keys=[category, 'all_tosses', against, metric,
                        'mean']
                    )
                    subresults = recursive_dict_add(
                        d=subresults,
                        val=None if None in training_toss_aucs+test_toss_aucs \
                            else np.mean(
                                training_toss_aucs + test_toss_aucs).item(),
                        keys=[category, 'all_tosses', against, metric,
                        'mean_auc']
                    )

class ResultsPlotter:
    """Generate plots of results stored in result dictionaries."""
    def __init__(self, bsdf_pll_results: dict, nerf_on_results: dict,
                 bsdf_only_results: dict, #pll_only_results: dict,
                 #pll_tagslam_results: dict, pll_blind_results: dict,
                 pll_vision_results: dict, pll_size_results: dict,
                 pll_blind_b_results: dict, pll_blind_t_results,
                 do_objects: bool):
        # Store the results dictionaries.
        self.bsdf_pll_results = bsdf_pll_results
        self.nerf_on_results = nerf_on_results
        self.bsdf_only_results = bsdf_only_results
        # self.pll_only_results = pll_only_results
        # self.pll_tagslam_results = pll_tagslam_results
        # self.pll_blind_results = pll_blind_results
        self.pll_vision_results = pll_vision_results
        self.pll_size_results = pll_size_results
        self.pll_blind_b_results = pll_blind_b_results
        self.pll_blind_t_results = pll_blind_t_results

        # Get a plotting directory.
        self.plot_dir = file_utils.plot_dir()
        print(f'Preparing to store to {self.plot_dir}')

        self.do_objects = do_objects

        # Extract all the objects.
        self.tagged_objects = [
            key for key in bsdf_pll_results['tagged_objects'].keys()]
        self.tagless_objects = [
            key for key in bsdf_pll_results['tagless_objects'].keys()]

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

        # Prepare to zip in consistent order:  BSDF-PLL, NeRF Online, BundleSDF
        # Only, PLL Vision, PLL Size, PLL Blind B, PLL Blind T.
        means = [all_objects_bsdf_pll_mean, all_objects_nerf_on_mean,
                 all_objects_bsdf_only_mean, all_objects_pll_vision_mean,
                 all_objects_pll_size_mean, all_objects_pll_blind_b_mean,
                 all_objects_pll_blind_t_mean]
        result_dicts = [self.bsdf_pll_results, self.nerf_on_results,
                        self.bsdf_only_results, self.pll_vision_results,
                        self.pll_size_results, self.pll_blind_b_results,
                        self.pll_blind_t_results]

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

            obj_means = [bsdf_pll_mean, nerf_on_mean, bsdf_only_mean,
                         pll_vision_mean, pll_size_mean, pll_blind_b_mean,
                         pll_blind_t_mean]

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
                pbt_data=pll_blind_t_mean,
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
            ylabel=ERROR_LABELS[geometry_metric], xlabel=NUM_TOSSES_LABEL,
            title=f'All Objects {hull_or_full.replace("_", " ")} Geometry'.title(),
            filename=f'all_objs_{geometry_metric}_v_data_{hull_or_full}',
            subdir='geometry',
            save_to_txt=True)

    def plot_tagslam_dynamics_error_vs_data(
            self, toss_subset: str, dynamics_category: str,
            dynamics_metric: str):
        """Dynamics metrics against TagSLAM.  Only doable for tagged objects and
        not for BundleSDF-only."""
        title_add = 'Rollout' if dynamics_category=='dynamics_rollout_metrics' \
            else 'Single-Step'

        # Keep track of all objects.
        all_objects_bsdf_pll_mean = [[], []]
        all_objects_nerf_on_mean = [[], []]
        all_objects_pll_vision_mean = [[], []]
        all_objects_pll_size_mean = [[], []]
        all_objects_pll_blind_b_mean = [[], []]
        all_objects_pll_blind_t_mean = [[], []]

        all_objects_bsdf_pll_auc = [[], []]
        all_objects_nerf_on_auc = [[], []]
        all_objects_pll_vision_auc = [[], []]
        all_objects_pll_size_auc = [[], []]
        all_objects_pll_blind_b_auc = [[], []]
        all_objects_pll_blind_t_auc = [[], []]

        # Prepare to zip in consistent order:  BSDF-PLL, NeRF Online, PLL
        # Vision, PLL Size, PLL Blind B, PLL Blind T.
        means = [all_objects_bsdf_pll_mean, all_objects_nerf_on_mean,
                 all_objects_pll_vision_mean, all_objects_pll_size_mean,
                 all_objects_pll_blind_b_mean, all_objects_pll_blind_t_mean]
        aucs = [all_objects_bsdf_pll_auc, all_objects_nerf_on_auc,
                all_objects_pll_vision_auc, all_objects_pll_size_auc,
                all_objects_pll_blind_b_auc, all_objects_pll_blind_t_auc]
        result_dicts = [self.bsdf_pll_results, self.nerf_on_results,
                        self.pll_vision_results, self.pll_size_results,
                        self.pll_blind_b_results, self.pll_blind_t_results]

        for obj in self.tagged_objects:
            scale = METRIC_SCALING[dynamics_metric]
            auc_scale = METRIC_SCALING['auc']

            bsdf_pll_mean = [[], []]
            nerf_on_mean = [[], []]
            pll_vision_mean = [[], []]
            pll_size_mean = [[], []]
            pll_blind_b_mean = [[], []]
            pll_blind_t_mean = [[], []]

            bsdf_pll_auc = [[], []]
            nerf_on_auc = [[], []]
            pll_vision_auc = [[], []]
            pll_size_auc = [[], []]
            pll_blind_b_auc = [[], []]
            pll_blind_t_auc = [[], []]

            obj_means = [bsdf_pll_mean, nerf_on_mean, pll_vision_mean,
                         pll_size_mean, pll_blind_b_mean, pll_blind_t_mean]
            obj_aucs = [bsdf_pll_auc, nerf_on_auc, pll_vision_auc,
                        pll_size_auc, pll_blind_b_auc, pll_blind_t_auc]

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
                pv_data=pll_vision_mean, ps_data=pll_size_mean,
                pbb_data=pll_blind_b_mean, pbt_data=pll_blind_t_mean,
                ylabel=ERROR_LABELS[dynamics_metric], xlabel=NUM_TOSSES_LABEL,
                title=f'{obj} {toss_subset.replace("_", " ")} Dynamics '.title() + \
                    f'{title_add} Prediction'.title(),
                filename=f'{obj}_{dynamics_metric}_tagslam_v_data_' + \
                    f'{toss_subset}_{dynamics_category}', subdir='dynamics')
            self._do_plot(
                bp_data=bsdf_pll_auc, n_data=nerf_on_auc,
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
        BundleSDF-only, not for PLL trained on TagSLAM, and since against
        BundleSDF this only works for all training tosses since predicting
        beyond the dataset requires TagSLAM."""
        if toss_subset != 'training_tosses':
            print(f'Cannot compute dynamics predictions w.r.t. BundleSDF ' + \
                  f'beyond the training set (told to do {toss_subset}; skip.')
            return

        title_add = 'Rollout' if dynamics_category=='dynamics_rollout_metrics' \
            else 'Single-Step'

        # Keep track of all objects.
        tagged_objects_bsdf_pll_mean = [[], []]
        tagged_objects_nerf_on_mean = [[], []]
        tagged_objects_pll_vision_mean = [[], []]
        tagged_objects_pll_size_mean = [[], []]
        tagged_objects_pll_blind_b_mean = [[], []]

        tagged_objects_bsdf_pll_auc = [[], []]
        tagged_objects_nerf_on_auc = [[], []]
        tagged_objects_pll_vision_auc = [[], []]
        tagged_objects_pll_size_auc = [[], []]
        tagged_objects_pll_blind_b_auc = [[], []]

        tagless_objects_bsdf_pll_mean = [[], []]
        tagless_objects_nerf_on_mean = [[], []]
        tagless_objects_pll_vision_mean = [[], []]
        tagless_objects_pll_size_mean = [[], []]
        tagless_objects_pll_blind_b_mean = [[], []]

        tagless_objects_bsdf_pll_auc = [[], []]
        tagless_objects_nerf_on_auc = [[], []]
        tagless_objects_pll_vision_auc = [[], []]
        tagless_objects_pll_size_auc = [[], []]
        tagless_objects_pll_blind_b_auc = [[], []]

        # Prepare to zip in consistent order:  BSDF-PLL, NeRF Online, PLL
        # Vision, PLL Size, PLL Blind B.
        tagged_means = [
            tagged_objects_bsdf_pll_mean, tagged_objects_nerf_on_mean,
            tagged_objects_pll_vision_mean, tagged_objects_pll_size_mean,
            tagged_objects_pll_blind_b_mean]
        tagged_aucs = [
            tagged_objects_bsdf_pll_auc, tagged_objects_nerf_on_auc,
            tagged_objects_pll_vision_auc, tagged_objects_pll_size_auc,
            tagged_objects_pll_blind_b_auc]
        tagless_means = [
            tagless_objects_bsdf_pll_mean, tagless_objects_nerf_on_mean,
            tagless_objects_pll_vision_mean, tagless_objects_pll_size_mean,
            tagless_objects_pll_blind_b_mean]
        tagless_aucs = [
            tagless_objects_bsdf_pll_auc, tagless_objects_nerf_on_auc,
            tagless_objects_pll_vision_auc, tagless_objects_pll_size_auc,
            tagless_objects_pll_blind_b_auc]
        result_dicts = [self.bsdf_pll_results, self.nerf_on_results,
                        self.pll_vision_results, self.pll_size_results,
                        self.pll_blind_b_results]

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
            pll_vision_mean = [[], []]
            pll_size_mean = [[], []]
            pll_blind_b_mean = [[], []]

            bsdf_pll_auc = [[], []]
            nerf_on_auc = [[], []]
            pll_vision_auc = [[], []]
            pll_size_auc = [[], []]
            pll_blind_b_auc = [[], []]

            obj_means = [bsdf_pll_mean, nerf_on_mean, pll_vision_mean,
                         pll_size_mean, pll_blind_b_mean]
            obj_aucs = [bsdf_pll_auc, nerf_on_auc, pll_vision_auc,
                        pll_size_auc, pll_blind_b_auc]

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
                pv_data=pll_vision_mean, ps_data=pll_size_mean,
                pbb_data=pll_blind_b_mean,
                ylabel=ERROR_LABELS[dynamics_metric],
                xlabel=NUM_TOSSES_LABEL,
                title=f'{obj} {toss_subset.replace("_", " ")} Dynamics '.title() + \
                    f'{title_add} Prediction'.title(),
                filename=f'{obj}_{dynamics_metric}_bsdf_v_data_{toss_subset}' +\
                    f'_{dynamics_category}', subdir='dynamics')
            self._do_plot(
                bp_data=bsdf_pll_auc, n_data=nerf_on_auc,
                pv_data=pll_vision_auc, ps_data=pll_size_auc,
                pbb_data=pll_blind_b_auc,
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
        all_objects_pll_vision_auc = \
            tagged_objects_pll_vision_auc + tagless_objects_pll_vision_auc
        all_objects_pll_size_auc = \
            tagged_objects_pll_size_auc + tagless_objects_pll_size_auc
        all_objects_pll_blind_b_auc = \
            tagged_objects_pll_blind_b_auc + tagless_objects_pll_blind_b_auc
        self._do_confidence_interval_plot(
            bp_data=tagged_objects_bsdf_pll_mean,
            n_data=tagged_objects_nerf_on_mean,
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
            pv_data=all_objects_pll_vision_auc,
            ps_data=all_objects_pll_size_auc,
            pbb_data=all_objects_pll_blind_b_auc,
            ylabel=AUC_LABELS[dynamics_metric], xlabel=NUM_TOSSES_LABEL,
            title=f'All Objects {toss_subset.replace("_", " ")} '.title() + \
                f'Dynamics {title_add} Prediction'.title(),
            filename=f'all_objs_{dynamics_metric}_bsdf_v_data_{toss_subset}' + \
                f'_{dynamics_category}_auc', subdir='dynamics')

    def plot_object_geometry_scatter(
            self, hull_or_full: str, geometry_metric: str):
        # Get the scale of the metric.
        scale = METRIC_SCALING[geometry_metric]

        # For now just do this for BundleSDF-PLL and BundleSDF-Only.
        by_object_bsdf_pll = {}  #[[], []] keys objects, vals [[toss_str], [ys]]
        by_object_bsdf_only = {}  #[[], []]

        # Prepare to zip in consistent order:  BSDF-PLL, BundleSDF Only.
        by_objects = [by_object_bsdf_pll, by_object_bsdf_only]
        result_dicts = [self.bsdf_pll_results, self.bsdf_only_results]

        # Iterate over all the approaches.
        for by_object, result_dict in zip(by_objects, result_dicts):

            # Iterate over all the objects.
            all_objects = self.tagless_objects + self.tagged_objects
            object_labels = ['tagless_objects'] * len(self.tagless_objects) + \
                ['tagged_objects'] * len(self.tagged_objects)

            for obj, tag_label in zip(all_objects, object_labels):
                if tag_label not in result_dict.keys():
                    continue
                if obj not in result_dict[tag_label].keys():
                    continue

                by_object[obj] = {}

                # Iterate over all the toss strings.
                for trained_on, result in result_dict[tag_label][obj].items():
                    toss_str = trained_on.split('trained_on_toss_')[-1]

                    by_object[obj][toss_str] = result['geometry_metrics'][
                        hull_or_full][geometry_metric] * scale

        # Generate the plots.
        title = 'Chamfer Distance' if \
            geometry_metric == 'chamfer_distance' and 'full' in hull_or_full \
            else f'Scatter {hull_or_full.replace("_", " ")} Geometry'.title()
        ylabel = '% object length' if geometry_metric == 'chamfer_distance' \
            and 'full' in hull_or_full else ERROR_LABELS[geometry_metric]
        self._do_scatter_plot(
            bp_data=by_object_bsdf_pll, bo_data=by_object_bsdf_only,
            ylabel=ylabel, xlabel_txt=all_objects,
            title=title,
            filename=f'scatter_{geometry_metric}_v_data_{hull_or_full}_normalized',
            subdir='object_geometry', save_to_txt=True, normalize=True)
        ylabel = 'Centimeters' if geometry_metric == 'chamfer_distance' \
            and 'full' in hull_or_full else ERROR_LABELS[geometry_metric]
        self._do_scatter_plot(
            bp_data=by_object_bsdf_pll, bo_data=by_object_bsdf_only,
            ylabel=ylabel, xlabel_txt=all_objects,
            title=title,
            filename=f'scatter_{geometry_metric}_v_data_{hull_or_full}_true_units',
            subdir='object_geometry', save_to_txt=True, normalize=False)

    def _do_scatter_plot(self, bp_data: list = None, bo_data: list = None,
                         ylabel: str = '', xlabel_txt: str = '',
                         title: str = '', filename: str = '', subdir: str = '',
                         save_to_txt: bool = False, normalize: bool = False):
        assert bp_data is not None and bo_data is not None

        # Make a plot for all the toss configurations.
        toss_groups = [['1', '2', '3', '4', '5']] #, ['1-2'], ['1-3'], ['1-4'], ['1-5']]
        for num_tosses, toss_strs in enumerate(toss_groups):
            num_tosses += 1   # Use 1-indexing.

            # Get longer lists repeating objects in groups, e.g. cube, cube, ...
            # x_txts = [obj for obj in xlabel_txt for _ in toss_strs]
            try:
                bp_ys = [bp_data[obj][toss_str] for obj in xlabel_txt for toss_str \
                    in toss_strs]
            except:
                pdb.set_trace()
            bo_ys = [bo_data[obj][toss_str] for obj in xlabel_txt for toss_str \
                in toss_strs]

            # Get in (n_object, n_toss_groups) 2D arrays.
            bp_ys = np.array(bp_ys).reshape(-1, len(toss_strs))
            bo_ys = np.array(bo_ys).reshape(-1, len(toss_strs))

            # Convert the data into % of normalized units.
            if normalize:
                for i, obj_name in enumerate(xlabel_txt):
                    bp_ys[i, :] /= CM_LENGTH_SCALES_BY_OBJ[obj_name]
                    bp_ys[i, :] *= 100
                    bo_ys[i, :] /= CM_LENGTH_SCALES_BY_OBJ[obj_name]
                    bo_ys[i, :] *= 100

            bp_ys_means = bp_ys.mean(axis=1).tolist()
            bo_ys_means = bo_ys.mean(axis=1).tolist()

            bp_ys = bp_ys.tolist()
            bo_ys = bo_ys.tolist()

            # Try sorting all the data so the average BundleSDF-only results are
            # in increasing order.
            combined = list(zip(
                xlabel_txt, bp_ys_means, bo_ys_means, bp_ys, bo_ys))
            sorted_combined = sorted(combined, key=lambda x: x[2])
            xlabel_txt, bp_ys_means, bo_ys_means, bp_ys, bo_ys = \
                zip(*sorted_combined)

            xs_means = np.array([i for i in range(len(xlabel_txt))])
            xs = np.array([i for i in range(len(xlabel_txt)) for _ in \
                           toss_strs])
            bp_ys = np.array(bp_ys).reshape(-1).tolist()
            bo_ys = np.array(bo_ys).reshape(-1).tolist()

            fig = plt.figure()
            ax = plt.gca()

            if save_to_txt:
                data_str = ''
                data_str += f'NORMALIZING? --> {normalize=}\n'
                data_str += f'{xlabel_txt=}'

            # Vysics.
            # plt.scatter(xs_means+0.2, bp_ys_means, s=MARKERSIZE*2,
            #     color=BSDF_PLL_COLOR, label=BSDF_PLL_LABEL)
            prefixes = [''] + ['_']*(len(xs_means)-1)
            for i in range(len(xs_means)):
                plt.plot([i-0.38, i+0.38], [bp_ys_means[i], bp_ys_means[i]],
                         linestyle='-', linewidth=LINEWIDTH*2,
                         color=BSDF_PLL_COLOR, label=prefixes[i]+BSDF_PLL_LABEL)
            # plt.plot(xs_means, bp_ys_means, '.', linestyle='-',
            #     markersize=MARKERSIZE/3, linewidth=LINEWIDTH,
            #     color=BSDF_PLL_COLOR, label=BSDF_PLL_LABEL)
            plt.scatter(xs+0.2, bp_ys, s=MARKERSIZE, color=BSDF_PLL_COLOR,
                        label='_')
            if save_to_txt:
                data_str += f'\nbp_data:\n{bp_ys}'

            # BundleSDF only.
            # plt.scatter(xs_means-0.2, bo_ys_means, s=MARKERSIZE*2,
            #     color=BSDF_ONLY_COLOR, label=BSDF_ONLY_LABEL)
            # plt.plot(xs_means, bo_ys_means, '.', linestyle='-',
            #     markersize=MARKERSIZE/3, linewidth=LINEWIDTH,
            #     color=BSDF_ONLY_COLOR,
            #     label=BSDF_ONLY_LABEL)
            for i in range(len(xs_means)):
                plt.plot([i-0.38, i+0.38], [bo_ys_means[i], bo_ys_means[i]],
                         linestyle='-', linewidth=LINEWIDTH*2,
                         color=BSDF_ONLY_COLOR,
                         label=prefixes[i]+BSDF_ONLY_LABEL)
            plt.scatter(xs-0.2, bo_ys, s=MARKERSIZE, color=BSDF_ONLY_COLOR,
                        label='_')
            if save_to_txt:
                data_str += f'\nbo_data:\n{bo_ys}'

            plt.xticks(range(len(xlabel_txt)), xlabel_txt, rotation=45,
                       ha='right', fontsize=20)

            if normalize:
                ylabel = ylabel.replace('[cm]', '[% length]')
            plt.ylabel(ylabel)
            plt.title(title)
            plt.legend()

            self._beautify_plot(fig, ax, xs_means, False, scatter=True,
                                normalized=normalize)

            plot_filename = filename.split('.')[0]
            plot_filename += f'_{num_tosses}.png'
            file_utils.assure_created(op.join(self.plot_dir, subdir))
            fig_path = op.join(self.plot_dir, subdir, plot_filename)
            fig.savefig(fig_path, dpi=100)
            plt.close()

            # Save the string to a text file as well.
            if save_to_txt:
                str_filepath = fig_path.replace('.png', '.txt')
                with open(str_filepath, 'w') as txt_file:
                    txt_file.write(data_str)
                print(f'Wrote to {str_filepath}')

    def _do_plot(self, bp_data: list = None, n_data: list = None,
                 bo_data: list = None, #po_data: list = None,
                 #pt_data: list = None, pb_data: list = None,
                 pv_data: list = None, ps_data: list = None,
                 pbb_data: list = None, pbt_data: list = None,
                 ylabel: str = '', xlabel: str = '', title: str = None,
                 filename: str = None, subdir: str = ''):
        # Skip individual object plots.
        if not self.do_objects:
            return

        fig = plt.figure()
        ax = plt.gca()

        if bp_data is not None and len(bp_data[0]) > 0:
            ax.plot(bp_data[0], bp_data[1], linewidth=LINEWIDTH,
                    color=BSDF_PLL_COLOR, label=BSDF_PLL_LABEL)
        if n_data is not None and len(n_data[0]) > 0:
            ax.plot(n_data[0], n_data[1], linewidth=LINEWIDTH,
                    color=NERF_ON_COLOR, label=NERF_ON_LABEL)
        if bo_data is not None and len(bo_data[0]) > 0:
            ax.plot(bo_data[0], bo_data[1], linewidth=LINEWIDTH,
                    color=BSDF_ONLY_COLOR, label=BSDF_ONLY_LABEL)
        if pv_data is not None and len(pv_data[0]) > 0:
            ax.plot(pv_data[0], pv_data[1], linewidth=LINEWIDTH,
                    color=PLL_VISION_COLOR, label=PLL_VISION_LABEL)
        if ps_data is not None and len(ps_data[0]) > 0:
            ax.plot(ps_data[0], ps_data[1], linewidth=LINEWIDTH,
                    color=PLL_SIZE_COLOR, label=PLL_SIZE_LABEL)
        if pbb_data is not None and len(pbb_data[0]) > 0:
            ax.plot(pbb_data[0], pbb_data[1], linewidth=LINEWIDTH,
                    color=PLL_BLIND_B_COLOR, label=PLL_BLIND_B_LABEL)
        if pbt_data is not None and len(pbt_data[0]) > 0:
            ax.plot(pbt_data[0], pbt_data[1], linewidth=LINEWIDTH,
                    color=PLL_BLIND_T_COLOR, label=PLL_BLIND_T_LABEL)

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

    def _add_to_string(self, string, data_name, x, y, l, u, ys, compare_with):
        pm = [(ui-li)/2 for ui, li in zip(u, l)]

        # Get a total mean and confidence interval.
        same_xs = [0] * len(ys)
        _, toty, totl, totu = xs_and_ys_to_x_mean_lower_upper(same_xs, ys)

        # Compare with the other data, using Welch's t-test.
        t, p = welchs_t_test(ys, compare_with)

        string += f'{data_name}:\n'
        string += f'{x=}\n'
        string += f'y = '
        for yi, pmi in zip(y, pm):
            string += f' & ${yi:.1f} \pm {pmi:.1f}$'
        string += f'\nCombined:  ${toty[0]:.1f} \pm {(totu[0]-totl[0])/2:.1f}$'
        string += f'\nWelchs t-test: {t=}, {p=}'
        string += f'\n\n'
        return string

    def _do_confidence_interval_plot(
            self, bp_data: list = None, n_data: list = None,
            bo_data: list = None,
            pv_data: list = None, ps_data: list = None,
            pbb_data: list = None, pbt_data: list = None,
            ylabel: str = '', xlabel: str = '',
            title: str = None, filename: str = None, subdir: str = '',
            save_to_txt: bool = False):

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
            ax.set_yscale('log')
            if normalized:
                ax.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
                ax.yaxis.set_minor_formatter(FormatStrFormatter("%.0f"))
            else:
                ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
                ax.yaxis.set_minor_formatter(FormatStrFormatter("%.1f"))

            fig.set_size_inches(13, 10)
            plt.subplots_adjust(bottom=0.15)

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
def process_gather_command():
    # Start dictionaries for each of the four result types.
    bsdf_pll_results = {}       # 02_2
    nerf_on_results = {}        # 03_2
    bsdf_only_results = {}      # bsdf 00_1

    pll_vision_results = {}     # pll 00_1
    pll_size_results = {}       # pll 04_1 (or pll 05_1)
    pll_blind_b_results = {}    # pll 07_1
    pll_blind_t_results = {}    # pll 09_0

    # Iterate over every evaluation subdirectory.
    for subdir in os.listdir(file_utils.evaluation_dir()):
        eval_subdir = op.join(file_utils.evaluation_dir(), subdir)
        if not op.exists(op.join(eval_subdir, 'results.yaml')):
            print(f'  No results in {subdir}.')
            continue

        if 'bsdf' in subdir and subdir.endswith('_1') and '00' in subdir:
            add_to_results = bsdf_only_results
        elif 'pll' in subdir and subdir.endswith('_1') and '00' in subdir:
            add_to_results = pll_vision_results
        elif 'pll' in subdir and subdir.endswith('_1') and '04' in subdir:
            add_to_results = pll_size_results
        elif 'pll' in subdir and subdir.endswith('_1') and '07' in subdir:
            add_to_results = pll_blind_b_results
        elif 'pll' in subdir and subdir.endswith('_0') and '09' in subdir:
            add_to_results = pll_blind_t_results
        elif '02' in subdir and subdir.endswith('_2'):
            add_to_results = bsdf_pll_results
        elif '03' in subdir and subdir.endswith('_2'):
            add_to_results = nerf_on_results
        else:
            print(f'  Skipping {subdir}')
            continue

        print(f'Found {subdir}...', end='')

        # Load the results.
        experiment_results = file_utils.load_results_yaml_in_subdir(subdir)

        # Add the experiment's results to the overall results.
        add_experiment_to_overall_results(experiment_results, add_to_results)
        print(f'done.')

    # Save the collected results.
    file_utils.save_results_to_yaml(
        bsdf_pll_results, file_utils.evaluation_dir(), filename='bsdf_pll.yaml')
    file_utils.save_results_to_yaml(
        nerf_on_results, file_utils.evaluation_dir(), filename='nerf_on.yaml')
    file_utils.save_results_to_yaml(
        bsdf_only_results, file_utils.evaluation_dir(),
        filename='bsdf_only.yaml')
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


# Use 'plot' command to load the previously generated yaml files with results
# and to generate plots with them.
PLOT_TRACKING = False
PLOT_DYNAMICS = False
PLOT_GEOMETRY = False
PLOT_GEOMETRY_SCATTERS = True
@cli.command('plot')
@click.option('--do-objects/--skip-objects',
              type=bool, default=False,
              help='Whether to plot object-level results or just aggregates.')
def process_plot_command(do_objects: bool):
    # Load the gathered results.
    bsdf_pll_results = file_utils.load_gathered_results_yaml('bsdf_pll.yaml')
    nerf_on_results = file_utils.load_gathered_results_yaml('nerf_on.yaml')
    bsdf_only_results = file_utils.load_gathered_results_yaml('bsdf_only.yaml')
    pll_vision_results = file_utils.load_gathered_results_yaml(
        'pll_vision.yaml')
    pll_size_results = file_utils.load_gathered_results_yaml(
        'pll_size.yaml')
    pll_blind_b_results = file_utils.load_gathered_results_yaml(
        'pll_blind_b.yaml')
    pll_blind_t_results = file_utils.load_gathered_results_yaml(
        'pll_blind_t.yaml')


    # Load an empty results dictionary for checking which metrics are valid for
    # which category/against which tracking.
    empty_results = file_utils.load_empty_results_yaml()

    results_plotter = ResultsPlotter(
        bsdf_pll_results=bsdf_pll_results,
        nerf_on_results=nerf_on_results,
        bsdf_only_results=bsdf_only_results,
        pll_vision_results=pll_vision_results,
        pll_size_results=pll_size_results,
        pll_blind_b_results=pll_blind_b_results,
        pll_blind_t_results=pll_blind_t_results,
        do_objects=do_objects
    )

    for metric in ERROR_LABELS.keys():
        for trajectory in ['full', 'toss']:
            if metric in \
                empty_results['tracking_metrics']['against_tagslam'].keys():
                if PLOT_TRACKING:
                    print(f'Plotting TagSLAM {trajectory}, {metric}')
                    results_plotter.plot_tagslam_tracking_error_vs_data(
                        trajectory, metric)
            if metric in \
                empty_results['tracking_metrics']['against_bundlesdf'].keys():
                if PLOT_TRACKING:
                    print(f'Plotting BundleSDF {trajectory}, {metric}')
                    results_plotter.plot_bundlesdf_tracking_error_vs_data(
                        trajectory, metric)

        for toss_subset in ['all_tosses', 'training_tosses', 'unseen_tosses']:
            for dynamics_category in [
                'dynamics_rollout_metrics', 'dynamics_single_step_metrics']:
                if metric in \
                    empty_results[dynamics_category]['against_tagslam'
                    ].keys():
                    if 'auc' in metric or 'training' in toss_subset:
                        print(f'Manually skipping some metrics')
                        continue
                    if PLOT_DYNAMICS:
                        print(f'Plotting TagSLAM {toss_subset}, ' + \
                            f'{dynamics_category}, {metric}')
                        results_plotter.plot_tagslam_dynamics_error_vs_data(
                            toss_subset, dynamics_category, metric)
        for dynamics_category in [
            'dynamics_rollout_metrics', 'dynamics_single_step_metrics']:
            if metric in \
                empty_results[dynamics_category]['against_bundlesdf'
                ].keys():
                toss_subset = 'training_tosses'
                if 'auc' in metric or 'training' in toss_subset:
                    print(f'Manually skipping some metrics')
                    continue
                if PLOT_DYNAMICS:
                    print(f'Plotting BundleSDF {toss_subset}, ' + \
                        f'{dynamics_category}, {metric}')
                    results_plotter.plot_bundlesdf_dynamics_error_vs_data(
                        toss_subset, dynamics_category, metric)

        if metric in \
            empty_results['geometry_metrics']['convex_hull'].keys():
            if PLOT_GEOMETRY:
                print(f'Plotting convex hull geometry {metric}')
                results_plotter.plot_geometry_error_vs_data(
                    'convex_hull', metric)
            if PLOT_GEOMETRY_SCATTERS:
                print(f'Plotting convex hull geometry scatters {metric}')
                results_plotter.plot_object_geometry_scatter(
                    'convex_hull', metric)
        if metric in \
            empty_results['geometry_metrics']['full_geometry'].keys():
            if PLOT_GEOMETRY:
                print(f'Plotting full geometry {metric}')
                results_plotter.plot_geometry_error_vs_data(
                    'full_geometry', metric)
            if PLOT_GEOMETRY_SCATTERS:
                print(f'Plotting full geometry geometry scatters {metric}')
                results_plotter.plot_object_geometry_scatter(
                    'full_geometry', metric)


if __name__ == '__main__':
    cli()
