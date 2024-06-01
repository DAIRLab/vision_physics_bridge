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

"""

import click
import numpy as np
import os
import os.path as op
import pdb

from matplotlib import rc, rcParams
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, NullFormatter

import eval_utils, file_utils

# Some settings on the plot generation.
rc('legend', fontsize=30)
plt.rc('axes', titlesize=40)    # fontsize of the axes title
plt.rc('axes', labelsize=40)    # fontsize of the x and y labels


ERROR_LABELS = {
    'position_error': 'Position Error [m]',
    'rotation_error': 'Rotation Error [deg]',
    'add_error': 'ADD Error [m]',
    'adds_error': 'ADD-S Error [m]',
    'penetration_learned_geom_tagslam_traj': \
        'Learned Geometry Penetration [m]',
    'penetration_true_geom_estimated_traj': \
        'True Geometry Penetration [m]',
    'penetration_learned_geom_estimated_traj': \
        'Learned Geometry Penetration [m]',
    'penetration_true_geom_predicted_traj': \
        'Predicted Penetration [m]',
    'volume_error': 'Relative Volume Error',
    'chamfer_distance': 'Chamfer Distance [m]',
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
    'position_error': 1,                            # [m]
    'rotation_error': 180/np.pi,                    # [deg]
    'add_error': 1,                                 # [m]
    'adds_error': 1,                                # [m]
    'penetration_learned_geom_tagslam_traj': 1,     # [m]
    'penetration_true_geom_estimated_traj': 1,      # [m]
    'penetration_learned_geom_estimated_traj': 1,   # [m]
    'penetration_true_geom_predicted_traj': 1,      # [m]
    'volume_error': 1,                              # weird fractional units
    'chamfer_distance': 1,                          # [m]
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

BSDF_PLL_COLOR = '#01256e'
NERF_ON_COLOR = '#92668d'
BSDF_ONLY_COLOR = '#398537'
PLL_ONLY_COLOR = '#95001a'
BSDF_PLL_LABEL = 'BundleSDF-PLL'
NERF_ON_LABEL = 'BundleSDF-PLL NeRF Online'
BSDF_ONLY_LABEL = 'BundleSDF Only'
PLL_ONLY_LABEL = 'PLL Only'
LINEWIDTH = 5


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
        if 'against_bundlesdf' not in subresults.keys():
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
                    val=np.mean(training_toss_means).item(),
                    keys=[category, 'training_tosses', against, metric,
                    'mean']
                )
                subresults = recursive_dict_add(
                    d=subresults,
                    val=np.mean(training_toss_aucs).item(),
                    keys=[category, 'training_tosses', against, metric,
                    'mean_auc']
                )
                if against == 'against_tagslam':
                    subresults = recursive_dict_add(
                        d=subresults,
                        val=np.mean(test_toss_means).item(),
                        keys=[category, 'unseen_tosses', against, metric,
                        'mean']
                    )
                    subresults = recursive_dict_add(
                        d=subresults,
                        val=np.mean(test_toss_aucs).item(),
                        keys=[category, 'unseen_tosses', against, metric,
                        'mean_auc']
                    )
                    subresults = recursive_dict_add(
                        d=subresults,
                        val=np.mean(
                            training_toss_means + test_toss_means).item(),
                        keys=[category, 'all_tosses', against, metric,
                        'mean']
                    )
                    subresults = recursive_dict_add(
                        d=subresults,
                        val=np.mean(
                            training_toss_aucs + test_toss_aucs).item(),
                        keys=[category, 'all_tosses', against, metric,
                        'mean_auc']
                    )

class ResultsPlotter:
    """Generate plots of results stored in result dictionaries."""
    def __init__(self, bsdf_pll_results: dict, nerf_on_results: dict,
                 bsdf_only_results: dict, pll_only_results: dict):
        # Store the results dictionaries.
        self.bsdf_pll_results = bsdf_pll_results
        self.nerf_on_results = nerf_on_results
        self.bsdf_only_results = bsdf_only_results
        self.pll_only_results = pll_only_results

        # Get a plotting directory.
        self.plot_dir = file_utils.plot_dir()
        print(f'Preparing to store to {self.plot_dir}')

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
        all_objects_bsdf_pll_auc = [[], []]
        all_objects_bsdf_only_mean = [[], []]
        all_objects_bsdf_only_auc = [[], []]

        for obj in self.tagged_objects:
            scale = METRIC_SCALING[tracking_metric]
            auc_scale = METRIC_SCALING['auc']

            # Get the BundleSDF-PLL results.  Store as a 2D array [[xs], [ys]].
            bsdf_pll_mean = [[], []]
            bsdf_pll_auc = [[], []]
            for tosses, results in self.bsdf_pll_results['tagged_objects'][obj
                ].items():
                tosses = tosses.split('trained_on_toss_')[-1]
                start_toss = int(tosses.split('-')[0])
                end_toss = int(tosses.split('-')[-1])
                n_tosses = end_toss - start_toss + 1

                error = results['tracking_metrics'][full_or_toss][
                    'against_tagslam'][tracking_metric]['mean'] * scale
                auc = results['tracking_metrics'][full_or_toss][
                    'against_tagslam'][tracking_metric]['mean_auc'] * auc_scale

                bsdf_pll_mean[0].append(n_tosses)
                bsdf_pll_auc[0].append(n_tosses)
                bsdf_pll_mean[1].append(error)
                bsdf_pll_auc[1].append(auc)
                all_objects_bsdf_pll_mean[0].append(n_tosses)
                all_objects_bsdf_pll_auc[0].append(n_tosses)
                all_objects_bsdf_pll_mean[1].append(error)
                all_objects_bsdf_pll_auc[1].append(auc)

            # Get the BundleSDF only results.
            bsdf_only_mean = [[], []]
            bsdf_only_auc = [[], []]
            for tosses, results in self.bsdf_only_results['tagged_objects'][obj
                ].items():
                tosses = tosses.split('trained_on_toss_')[-1]
                start_toss = int(tosses.split('-')[0])
                end_toss = int(tosses.split('-')[-1])
                n_tosses = end_toss - start_toss + 1

                error = results['tracking_metrics'][full_or_toss][
                    'against_tagslam'][tracking_metric]['mean'] * scale
                auc = results['tracking_metrics'][full_or_toss][
                    'against_tagslam'][tracking_metric]['mean_auc'] * auc_scale

                bsdf_only_mean[0].append(n_tosses)
                bsdf_only_auc[0].append(n_tosses)
                bsdf_only_mean[1].append(error)
                bsdf_only_auc[1].append(auc)
                all_objects_bsdf_only_mean[0].append(n_tosses)
                all_objects_bsdf_only_auc[0].append(n_tosses)
                all_objects_bsdf_only_mean[1].append(error)
                all_objects_bsdf_only_auc[1].append(auc)

            # Generate the plots.
            self._do_plot(
                bp_data=bsdf_pll_mean, bo_data=bsdf_only_mean,
                ylabel=ERROR_LABELS[tracking_metric], xlabel=NUM_TOSSES_LABEL,
                title=f'{obj} {full_or_toss} Trajectory'.title(),
                filename=f'{obj}_{tracking_metric}_v_data_{full_or_toss}.png',
                subdir='tracking')
            self._do_plot(
                bp_data=bsdf_pll_auc, bo_data=bsdf_only_auc,
                ylabel=AUC_LABELS[tracking_metric], xlabel=NUM_TOSSES_LABEL,
                title=f'{obj} {full_or_toss} Trajectory'.title(),
                filename=f'{obj}_{tracking_metric}_v_data_{full_or_toss}' + \
                    f'_auc.png', subdir='tracking')
            
        # Do a confidence interval plot that aggregates all the objects.
        self._do_confidence_interval_plot(
            bp_data=all_objects_bsdf_pll_mean,
            bo_data=all_objects_bsdf_only_mean,
            ylabel=ERROR_LABELS[tracking_metric], xlabel=NUM_TOSSES_LABEL,
            title=f'All Tagged Objects {full_or_toss.capitalize()} Trajectory',
            filename=f'all_tagged_objs_{tracking_metric}_v_data_' + \
                f'{full_or_toss}.png', subdir='tracking')
        self._do_confidence_interval_plot(
            bp_data=all_objects_bsdf_pll_auc,
            bo_data=all_objects_bsdf_only_auc,
            ylabel=AUC_LABELS[tracking_metric], xlabel=NUM_TOSSES_LABEL,
            title=f'All Tagged Objects {full_or_toss.capitalize()} Trajectory',
            filename=f'all_tagged_objs_{tracking_metric}_v_data_' + \
                f'{full_or_toss}_auc.png', subdir='tracking')

    def plot_bundlesdf_tracking_error_vs_data(
            self, full_or_toss: str, tracking_metric: str):
        """Tracking metrics against BundleSDF.  Doable for all objects but not
        for PLL-only."""
        # Keep track of all objects.
        all_objects_bsdf_pll_mean = [[], []]
        all_objects_bsdf_pll_auc = [[], []]
        all_objects_bsdf_only_mean = [[], []]
        all_objects_bsdf_only_auc = [[], []]

        # Do both tagged and tagless objects.
        all_objects = self.tagless_objects + self.tagged_objects
        object_labels = ['tagless_objects'] * len(self.tagless_objects) + \
            ['tagged_objects'] * len(self.tagged_objects)
        for obj, tag_label in zip(all_objects, object_labels):
            scale = METRIC_SCALING[tracking_metric]
            auc_scale = METRIC_SCALING['auc']

            # Get the BundleSDF-PLL results.  Store as a 2D array [[xs], [ys]].
            bsdf_pll_mean = [[], []]
            bsdf_pll_auc = [[], []]
            for tosses, results in self.bsdf_pll_results[tag_label][obj
                ].items():
                tosses = tosses.split('trained_on_toss_')[-1]
                start_toss = int(tosses.split('-')[0])
                end_toss = int(tosses.split('-')[-1])
                n_tosses = end_toss - start_toss + 1

                error = results['tracking_metrics'][full_or_toss][
                    'against_bundlesdf'][tracking_metric]['mean'] * scale
                auc = results['tracking_metrics'][full_or_toss][
                    'against_bundlesdf'][tracking_metric]['mean_auc'] * \
                        auc_scale

                bsdf_pll_mean[0].append(n_tosses)
                bsdf_pll_auc[0].append(n_tosses)
                bsdf_pll_mean[1].append(error)
                bsdf_pll_auc[1].append(auc)
                all_objects_bsdf_pll_mean[0].append(n_tosses)
                all_objects_bsdf_pll_auc[0].append(n_tosses)
                all_objects_bsdf_pll_mean[1].append(error)
                all_objects_bsdf_pll_auc[1].append(auc)

            # Get the BundleSDF only results.
            bsdf_only_mean = [[], []]
            bsdf_only_auc = [[], []]
            for tosses, results in self.bsdf_only_results[tag_label][obj
                ].items():
                tosses = tosses.split('trained_on_toss_')[-1]
                start_toss = int(tosses.split('-')[0])
                end_toss = int(tosses.split('-')[-1])
                n_tosses = end_toss - start_toss + 1

                error = results['tracking_metrics'][full_or_toss][
                    'against_bundlesdf'][tracking_metric]['mean'] * scale
                auc = results['tracking_metrics'][full_or_toss][
                    'against_bundlesdf'][tracking_metric]['mean_auc'] * \
                        auc_scale

                bsdf_only_mean[0].append(n_tosses)
                bsdf_only_auc[0].append(n_tosses)
                bsdf_only_mean[1].append(error)
                bsdf_only_auc[1].append(auc)
                all_objects_bsdf_only_mean[0].append(n_tosses)
                all_objects_bsdf_only_auc[0].append(n_tosses)
                all_objects_bsdf_only_mean[1].append(error)
                all_objects_bsdf_only_auc[1].append(auc)

            # Generate the plots.
            self._do_plot(
                bp_data=bsdf_pll_mean, bo_data=bsdf_only_mean,
                ylabel=ERROR_LABELS[tracking_metric], xlabel=NUM_TOSSES_LABEL,
                title=f'{obj} {full_or_toss} Trajectory'.title(),
                filename=f'{obj}_{tracking_metric}_v_data_{full_or_toss}.png',
                subdir='tracking')
            self._do_plot(
                bp_data=bsdf_pll_auc, bo_data=bsdf_only_auc,
                ylabel=AUC_LABELS[tracking_metric], xlabel=NUM_TOSSES_LABEL,
                title=f'{obj} {full_or_toss} Trajectory'.title(),
                filename=f'{obj}_{tracking_metric}_v_data_{full_or_toss}_auc.png',
                subdir='tracking')

        # Do a confidence interval plot that aggregates all the objects.
        self._do_confidence_interval_plot(
            bp_data=all_objects_bsdf_pll_mean,
            bo_data=all_objects_bsdf_only_mean,
            ylabel=ERROR_LABELS[tracking_metric], xlabel=NUM_TOSSES_LABEL,
            title=f'All Objects {full_or_toss.capitalize()} Trajectory',
            filename=f'all_objs_{tracking_metric}_v_data_' + \
                f'{full_or_toss}.png', subdir='tracking')
        self._do_confidence_interval_plot(
            bp_data=all_objects_bsdf_pll_auc,
            bo_data=all_objects_bsdf_only_auc,
            ylabel=AUC_LABELS[tracking_metric], xlabel=NUM_TOSSES_LABEL,
            title=f'All Objects {full_or_toss.capitalize()} Trajectory',
            filename=f'all_objs_{tracking_metric}_v_data_' + \
                f'{full_or_toss}_auc.png', subdir='tracking')

    def plot_geometry_error_vs_data(
            self, hull_or_full: str, geometry_metric: str):
        """Geometry metrics.  Doable for all objects and approaches."""
        # Keep track of all objects.
        all_objects_bsdf_pll_mean = [[], []]
        all_objects_bsdf_only_mean = [[], []]
        all_objects_pll_only_mean = [[], []]

        # Do both tagged and tagless objects.
        all_objects = self.tagless_objects + self.tagged_objects
        object_labels = ['tagless_objects'] * len(self.tagless_objects) + \
            ['tagged_objects'] * len(self.tagged_objects)
        for obj, tag_label in zip(all_objects, object_labels):
            scale = METRIC_SCALING[geometry_metric]

            # Get the BundleSDF-PLL results.  Store as a 2D array [[xs], [ys]].
            bsdf_pll_mean = [[], []]
            for tosses, results in self.bsdf_pll_results[tag_label][obj
                ].items():
                tosses = tosses.split('trained_on_toss_')[-1]
                start_toss = int(tosses.split('-')[0])
                end_toss = int(tosses.split('-')[-1])
                n_tosses = end_toss - start_toss + 1

                error = results['geometry_metrics'][hull_or_full][
                    geometry_metric] * scale

                bsdf_pll_mean[0].append(n_tosses)
                bsdf_pll_mean[1].append(error)
                all_objects_bsdf_pll_mean[0].append(n_tosses)
                all_objects_bsdf_pll_mean[1].append(error)

            # Get the BundleSDF only results.
            bsdf_only_mean = [[], []]
            for tosses, results in self.bsdf_only_results[tag_label][obj
                ].items():
                tosses = tosses.split('trained_on_toss_')[-1]
                start_toss = int(tosses.split('-')[0])
                end_toss = int(tosses.split('-')[-1])
                n_tosses = end_toss - start_toss + 1

                error = results['geometry_metrics'][hull_or_full][
                    geometry_metric] * scale

                bsdf_only_mean[0].append(n_tosses)
                bsdf_only_mean[1].append(error)
                all_objects_bsdf_only_mean[0].append(n_tosses)
                all_objects_bsdf_only_mean[1].append(error)

            # Get the PLL only results.
            pll_only_mean = [[], []]
            if tag_label not in self.pll_only_results.keys():
                pll_only_mean = None
            else:
                for tosses, results in self.pll_only_results[tag_label][obj
                    ].items():
                    tosses = tosses.split('trained_on_toss_')[-1]
                    start_toss = int(tosses.split('-')[0])
                    end_toss = int(tosses.split('-')[-1])
                    n_tosses = end_toss - start_toss + 1

                    error = results['geometry_metrics'][hull_or_full][
                        geometry_metric] * scale

                    pll_only_mean[0].append(n_tosses)
                    pll_only_mean[1].append(error)
                    all_objects_pll_only_mean[0].append(n_tosses)
                    all_objects_pll_only_mean[1].append(error)

            # Generate the plots.
            self._do_plot(
                bp_data=bsdf_pll_mean, bo_data=bsdf_only_mean,
                po_data=pll_only_mean,
                ylabel=ERROR_LABELS[geometry_metric], xlabel=NUM_TOSSES_LABEL,
                title=f'{obj} {hull_or_full.replace("_", " ")} Geometry'.title(),
                filename=f'{obj}_{geometry_metric}_v_data_{hull_or_full}.png',
                subdir='geometry')

        # Do a confidence interval plot that aggregates all the objects.
        self._do_confidence_interval_plot(
            bp_data=all_objects_bsdf_pll_mean,
            bo_data=all_objects_bsdf_only_mean,
            po_data=all_objects_pll_only_mean,
            ylabel=ERROR_LABELS[geometry_metric], xlabel=NUM_TOSSES_LABEL,
            title=f'All Objects {hull_or_full.replace("_", " ")} Geometry'.title(),
            filename=f'all_objs_{geometry_metric}_v_data_{hull_or_full}.png',
            subdir='geometry')

    def plot_tagslam_dynamics_rollout_error_vs_data(
            self, toss_subset: str, dynamics_metric: str):
        """Dynamics metrics against TagSLAM.  Only doable for tagged objects and
        not for BundleSDF-only."""
        # Keep track of all objects.
        all_objects_bsdf_pll_mean = [[], []]
        all_objects_bsdf_pll_auc = [[], []]
        all_objects_pll_only_mean = [[], []]
        all_objects_pll_only_auc = [[], []]

        for obj in self.tagged_objects:
            scale = METRIC_SCALING[dynamics_metric]
            auc_scale = METRIC_SCALING['auc']

            # Get the BundleSDF-PLL results.  Store as a 2D array [[xs], [ys]].
            bsdf_pll_mean = [[], []]
            bsdf_pll_auc = [[], []]
            for tosses, results in self.bsdf_pll_results['tagged_objects'][obj
                ].items():
                tosses = tosses.split('trained_on_toss_')[-1]
                start_toss = int(tosses.split('-')[0])
                end_toss = int(tosses.split('-')[-1])
                n_tosses = end_toss - start_toss + 1

                error = results['dynamics_rollout_metrics'][toss_subset][
                    'against_tagslam'][dynamics_metric]['mean'] * scale
                auc = results['dynamics_rollout_metrics'][toss_subset][
                    'against_tagslam'][dynamics_metric]['mean_auc'] * auc_scale

                bsdf_pll_mean[0].append(n_tosses)
                bsdf_pll_auc[0].append(n_tosses)
                bsdf_pll_mean[1].append(error)
                bsdf_pll_auc[1].append(auc)
                all_objects_bsdf_pll_mean[0].append(n_tosses)
                all_objects_bsdf_pll_auc[0].append(n_tosses)
                all_objects_bsdf_pll_mean[1].append(error)
                all_objects_bsdf_pll_auc[1].append(auc)

            # Get the PLL only results.
            pll_only_mean = [[], []]
            pll_only_auc = [[], []]
            for tosses, results in self.pll_only_results['tagged_objects'][obj
                ].items():
                tosses = tosses.split('trained_on_toss_')[-1]
                start_toss = int(tosses.split('-')[0])
                end_toss = int(tosses.split('-')[-1])
                n_tosses = end_toss - start_toss + 1

                error = results['dynamics_rollout_metrics'][toss_subset][
                    'against_tagslam'][dynamics_metric]['mean'] * scale
                auc = results['dynamics_rollout_metrics'][toss_subset][
                    'against_tagslam'][dynamics_metric]['mean_auc'] * auc_scale

                pll_only_mean[0].append(n_tosses)
                pll_only_auc[0].append(n_tosses)
                pll_only_mean[1].append(error)
                pll_only_auc[1].append(auc)
                all_objects_pll_only_mean[0].append(n_tosses)
                all_objects_pll_only_auc[0].append(n_tosses)
                all_objects_pll_only_mean[1].append(error)
                all_objects_pll_only_auc[1].append(auc)

            # Generate the plots.
            self._do_plot(
                bp_data=bsdf_pll_mean, po_data=pll_only_mean,
                ylabel=ERROR_LABELS[dynamics_metric], xlabel=NUM_TOSSES_LABEL,
                title=f'{obj} {toss_subset.replace("_", " ")} Dynamics '.title() + \
                    f'Prediction'.title(),
                filename=f'{obj}_{dynamics_metric}_tagslam_v_data_{toss_subset}.png',
                subdir='dynamics')
            self._do_plot(
                bp_data=bsdf_pll_auc, po_data=pll_only_auc,
                ylabel=AUC_LABELS[dynamics_metric], xlabel=NUM_TOSSES_LABEL,
                title=f'{obj} {toss_subset.replace("_", " ")} Dynamics '.title() + \
                    f'Prediction'.title(),
                filename=f'{obj}_{dynamics_metric}_tagslam_v_data_{toss_subset}' + \
                    f'_auc.png', subdir='dynamics')

        # Do a confidence interval plot that aggregates all the objects.
        self._do_confidence_interval_plot(
            bp_data=all_objects_bsdf_pll_mean,
            po_data=all_objects_pll_only_mean,
            ylabel=ERROR_LABELS[dynamics_metric], xlabel=NUM_TOSSES_LABEL,
            title=f'All Tagged Objects {toss_subset.replace("_", " ")} '.title() + \
                f'Dynamics Prediction'.title(),
            filename=f'all_tagged_objs_{dynamics_metric}_tagslam_v_data_' + \
                f'{toss_subset}.png', subdir='dynamics')
        self._do_confidence_interval_plot(
            bp_data=all_objects_bsdf_pll_auc,
            po_data=all_objects_pll_only_auc,
            ylabel=AUC_LABELS[dynamics_metric], xlabel=NUM_TOSSES_LABEL,
            title=f'All Tagged Objects {toss_subset.replace("_", " ")} '.title() + \
                f'Dynamics Prediction'.title(),
            filename=f'all_tagged_objs_{dynamics_metric}_tagslam_v_data_' + \
                f'{toss_subset}_auc.png', subdir='dynamics')

    def plot_bundlesdf_dynamics_rollout_error_vs_data(
            self, toss_subset: str, dynamics_metric: str):
        """Dynamics metrics against BundleSDF.  Doable for all objects, not
        not for BundleSDF-only, and since against BundleSDF this only works for
        all training tosses since predicting beyond the dataset requires
        TagSLAM."""
        if toss_subset != 'training_tosses':
            print(f'Cannot compute dynamics predictions w.r.t. BundleSDF ' + \
                  f'beyond the training set (told to do {toss_subset}; skip.')
            return

        # Keep track of all objects.
        all_objects_bsdf_pll_mean = [[], []]
        all_objects_bsdf_pll_auc = [[], []]
        all_objects_pll_only_mean = [[], []]
        all_objects_pll_only_auc = [[], []]

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
            if toss_subset != 'training_tosses':
                pass

            scale = METRIC_SCALING[dynamics_metric]
            auc_scale = METRIC_SCALING['auc']

            # Get the BundleSDF-PLL results.  Store as a 2D array [[xs], [ys]].
            bsdf_pll_mean = [[], []]
            bsdf_pll_auc = [[], []]
            for tosses, results in self.bsdf_pll_results[tag_label][obj
                ].items():
                tosses = tosses.split('trained_on_toss_')[-1]
                start_toss = int(tosses.split('-')[0])
                end_toss = int(tosses.split('-')[-1])
                n_tosses = end_toss - start_toss + 1

                error = results['dynamics_rollout_metrics'][toss_subset][
                    'against_bundlesdf'][dynamics_metric]['mean'] * scale
                auc = results['dynamics_rollout_metrics'][toss_subset][
                    'against_bundlesdf'][dynamics_metric]['mean_auc'] * \
                        auc_scale

                bsdf_pll_mean[0].append(n_tosses)
                bsdf_pll_auc[0].append(n_tosses)
                bsdf_pll_mean[1].append(error)
                bsdf_pll_auc[1].append(auc)
                all_objects_bsdf_pll_mean[0].append(n_tosses)
                all_objects_bsdf_pll_auc[0].append(n_tosses)
                all_objects_bsdf_pll_mean[1].append(error)
                all_objects_bsdf_pll_auc[1].append(auc)

            # Get the PLL only results.
            pll_only_mean = [[], []]
            pll_only_auc = [[], []]
            if tag_label not in self.pll_only_results:
                pll_only_mean = None
                pll_only_auc = None
            else:
                for tosses, results in self.pll_only_results[tag_label][obj
                    ].items():
                    tosses = tosses.split('trained_on_toss_')[-1]
                    start_toss = int(tosses.split('-')[0])
                    end_toss = int(tosses.split('-')[-1])
                    n_tosses = end_toss - start_toss + 1

                    error = results['dynamics_rollout_metrics'][toss_subset][
                        'against_bundlesdf'][dynamics_metric]['mean'] * scale
                    auc = results['dynamics_rollout_metrics'][toss_subset][
                        'against_bundlesdf'][dynamics_metric]['mean_auc'] * \
                            auc_scale

                    pll_only_mean[0].append(n_tosses)
                    pll_only_auc[0].append(n_tosses)
                    pll_only_mean[1].append(error)
                    pll_only_auc[1].append(auc)
                    all_objects_pll_only_mean[0].append(n_tosses)
                    all_objects_pll_only_auc[0].append(n_tosses)
                    all_objects_pll_only_mean[1].append(error)
                    all_objects_pll_only_auc[1].append(auc)

            # Generate the plots.
            self._do_plot(
                bp_data=bsdf_pll_mean, po_data=pll_only_mean,
                ylabel=ERROR_LABELS[dynamics_metric], xlabel=NUM_TOSSES_LABEL,
                title=f'{obj} {toss_subset.replace("_", " ")} Dynamics '.title() + \
                    f'Prediction'.title(),
                filename=f'{obj}_{dynamics_metric}_bsdf_v_data_{toss_subset}.png',
                subdir='dynamics')
            self._do_plot(
                bp_data=bsdf_pll_auc, po_data=pll_only_auc,
                ylabel=AUC_LABELS[dynamics_metric], xlabel=NUM_TOSSES_LABEL,
                title=f'{obj} {toss_subset.replace("_", " ")} Dynamics '.title() + \
                    f'Prediction'.title(),
                filename=f'{obj}_{dynamics_metric}_bsdf_v_data_{toss_subset}' + \
                    f'_auc.png', subdir='dynamics')

        # Do a confidence interval plot that aggregates all the objects.
        self._do_confidence_interval_plot(
            bp_data=all_objects_bsdf_pll_mean,
            po_data=all_objects_pll_only_mean,
            ylabel=ERROR_LABELS[dynamics_metric], xlabel=NUM_TOSSES_LABEL,
            title=f'All Objects {toss_subset.replace("_", " ")} '.title() + \
                f'Dynamics Prediction'.title(),
            filename=f'all_objs_{dynamics_metric}_bsdf_v_data_{toss_subset}.png',
            subdir='dynamics')
        self._do_confidence_interval_plot(
            bp_data=all_objects_bsdf_pll_auc,
            po_data=all_objects_pll_only_auc,
            ylabel=AUC_LABELS[dynamics_metric], xlabel=NUM_TOSSES_LABEL,
            title=f'All Objects {toss_subset.replace("_", " ")} '.title() + \
                f'Dynamics Prediction'.title(),
            filename=f'all_objs_{dynamics_metric}_bsdf_v_data_{toss_subset}_auc.png',
            subdir='dynamics')

    def _do_plot(self, bp_data: list = None, n_data: list = None,
                 bo_data: list = None, po_data: list = None, ylabel: str = '',
                 xlabel: str = '', title: str = None, filename: str = None,
                 subdir: str = ''):
        fig = plt.figure()
        ax = plt.gca()

        if bp_data is not None:
            ax.plot(bp_data[0], bp_data[1], linewidth=LINEWIDTH,
                    color=BSDF_PLL_COLOR, label=BSDF_PLL_LABEL)
        if n_data is not None:
            ax.plot(n_data[0], n_data[1], linewidth=LINEWIDTH,
                    color=NERF_ON_COLOR, label=NERF_ON_LABEL)
        if bo_data is not None:
            ax.plot(bo_data[0], bo_data[1], linewidth=LINEWIDTH,
                    color=BSDF_ONLY_COLOR, label=BSDF_ONLY_LABEL)
        if po_data is not None:
            ax.plot(po_data[0], po_data[1], linewidth=LINEWIDTH,
                    color=PLL_ONLY_COLOR, label=PLL_ONLY_LABEL)

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

    def _do_confidence_interval_plot(
            self, bp_data: list = None, n_data: list = None,
            bo_data: list = None, po_data: list = None, ylabel: str = '',
            xlabel: str = '', title: str = None, filename: str = None,
            subdir: str = ''):
        fig = plt.figure()
        ax = plt.gca()

        if bp_data is not None:
            # Convert the data to mean/lower/upper.
            x, y, l, u = xs_and_ys_to_x_mean_lower_upper(bp_data[0], bp_data[1])
            ax.plot(x, y, linewidth=LINEWIDTH,
                    color=BSDF_PLL_COLOR, label=BSDF_PLL_LABEL)
            ax.fill_between(x, l, u, alpha=0.3, color=BSDF_PLL_COLOR)
        if n_data is not None:
            x, y, l, u = xs_and_ys_to_x_mean_lower_upper(n_data[0], n_data[1])
            ax.plot(x, y, linewidth=LINEWIDTH,
                    color=NERF_ON_COLOR, label=NERF_ON_LABEL)
            ax.fill_between(x, l, u, alpha=0.3, color=BSDF_PLL_COLOR)
        if bo_data is not None:
            x, y, l, u = xs_and_ys_to_x_mean_lower_upper(bo_data[0], bo_data[1])
            ax.plot(x, y, linewidth=LINEWIDTH,
                    color=BSDF_ONLY_COLOR, label=BSDF_ONLY_LABEL)
            ax.fill_between(x, l, u, alpha=0.3, color=BSDF_ONLY_COLOR)
        if po_data is not None:
            x, y, l, u = xs_and_ys_to_x_mean_lower_upper(po_data[0], po_data[1])
            ax.plot(x, y, linewidth=LINEWIDTH,
                    color=PLL_ONLY_COLOR, label=PLL_ONLY_LABEL)
            ax.fill_between(x, l, u, alpha=0.3, color=PLL_ONLY_COLOR)

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

    def _beautify_plot(self, fig, ax, x_markers, auc: bool):
        """Perform all the nice formatting on a plot."""

        ax.xaxis.set_major_formatter(NullFormatter())
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(NullFormatter())

        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.set_xticks(x_markers)
        ax.set_xticklabels(x_markers)

        ax.tick_params(axis='x', which='minor', bottom=False, labelsize=20)
        ax.tick_params(axis='x', which='major', bottom=False, labelsize=20)

        ax.tick_params(axis='y', which='minor', labelsize=20)
        ax.tick_params(axis='y', which='major', labelsize=20)

        if auc:
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
            ax.yaxis.set_minor_formatter(FormatStrFormatter("%.0f"))
            ax.set_ylim(0, 110)
        else:
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
            ax.yaxis.set_minor_formatter(FormatStrFormatter("%.3f"))
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.0f"))

        ax.yaxis.grid(True, which='both')
        ax.xaxis.grid(True, which='major')

        handles, labels = plt.gca().get_legend_handles_labels()

        plt.legend(handles, labels)
        plt.legend(prop=dict(weight='bold'))

        fig.set_size_inches(13, 13)


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

        # Add AUC to the results.
        add_auc_to_results(results)

        # Save the results.
        file_utils.save_results_to_yaml(results, eval_subdir)
        print(f'done.')


# Use 'gather' command to gather all the results into yaml files, one for
# BundleSDF-PLL, one for BundleSDF-only, and another for PLL-only.
@cli.command('gather')
def process_gather_command():
    # Start dictionaries for each of the four result types.
    bsdf_pll_results = {}
    nerf_on_results = {}
    bsdf_only_results = {}
    pll_only_results = {}
    pll_tagslam_results = {}

    # Iterate over every evaluation subdirectory.
    for subdir in os.listdir(file_utils.evaluation_dir()):
        eval_subdir = op.join(file_utils.evaluation_dir(), subdir)
        if not op.exists(op.join(eval_subdir, 'results.yaml')):
            print(f'  No results in {subdir}.')
            continue

        if 'bsdf' in subdir:
            add_to_results = bsdf_only_results
        elif 'pll' in subdir and subdir.endswith('_1'):
            add_to_results = pll_only_results
        elif 'pll' in subdir and subdir.endswith('_0'):
            add_to_results = pll_tagslam_results
        elif '02' in subdir:
            add_to_results = bsdf_pll_results
        elif '03' in subdir:
            add_to_results = nerf_on_results
        else:
            print(f'  Skipping {subdir} because looking for 02 or 03 ' + \
                  f'BundleSDF IDs.')
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
        pll_only_results, file_utils.evaluation_dir(), filename='pll_only.yaml')
    file_utils.save_results_to_yaml(
        pll_tagslam_results, file_utils.evaluation_dir(),
        filename='pll_tagslam.yaml')


# Use 'plot' command to load the previously generated yaml files with results
# and to generate plots with them.
PLOT_TRACKING = False
PLOT_DYNAMICS = False
PLOT_GEOMETRY = True
@cli.command('plot')
def process_plot_command():
    # Load the gathered results.
    bsdf_pll_results = file_utils.load_gathered_results_yaml('bsdf_pll.yaml')
    nerf_on_results = file_utils.load_gathered_results_yaml('nerf_on.yaml')
    bsdf_only_results = file_utils.load_gathered_results_yaml('bsdf_only.yaml')
    pll_only_results = file_utils.load_gathered_results_yaml('pll_only.yaml')

    # Load an empty results dictionary for checking which metrics are valid for
    # which category/against which tracking.
    empty_results = file_utils.load_empty_results_yaml()

    results_plotter = ResultsPlotter(
        bsdf_pll_results=bsdf_pll_results,
        nerf_on_results=nerf_on_results,
        bsdf_only_results=bsdf_only_results,
        pll_only_results=pll_only_results
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
            if metric in \
                empty_results['dynamics_rollout_metrics']['against_tagslam'
                ].keys():
                if PLOT_DYNAMICS:
                    print(f'Plotting TagSLAM {toss_subset}, {metric}')
                    results_plotter.plot_tagslam_dynamics_rollout_error_vs_data(
                        toss_subset, metric)
        if metric in \
            empty_results['dynamics_rollout_metrics']['against_bundlesdf'
            ].keys():
            toss_subset = 'training_tosses'
            if PLOT_DYNAMICS:
                print(f'Plotting BundleSDF {toss_subset}, {metric}')
                results_plotter.plot_bundlesdf_dynamics_rollout_error_vs_data(
                    toss_subset, metric)

        if metric in \
            empty_results['geometry_metrics']['convex_hull'].keys():
            if PLOT_GEOMETRY:
                print(f'Plotting convex hull geometry {metric}')
                results_plotter.plot_geometry_error_vs_data(
                    'convex_hull', metric)
        if metric in \
            empty_results['geometry_metrics']['full_geometry'].keys():
            if PLOT_GEOMETRY:
                print(f'Plotting full geometry {metric}')
                results_plotter.plot_geometry_error_vs_data(
                    'full_geometry', metric)


if __name__ == '__main__':
    cli()
