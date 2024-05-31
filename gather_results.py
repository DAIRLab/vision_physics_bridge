"""Script to gather evaluation results.  Writes a few yaml files:

    bsdf_pll_online_nerf?
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
                        against_bundlesdf:
                            metric_1:     <-- all of these from training tosses
                                mean: XXX
                                mean_auc: XXX
                            metric_2: ...
                            metric_3: ...
                            ...
                        against_tagslam:
                            training_tosses:
                                metric_1:
                                    mean: XXX
                                    mean_auc: XXX
                                metric_2: ...
                                metric_3: ...
                                ...
                            unseen_tosses: ...
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
                            against_tagslam:
                                metric_1:
                                    mean: XXX
                                    mean_auc: XXX
                                metric_2: ...
                                metric_3: ...
                                ...
                        full: ...
                    dynamics_rollout_metrics:
                        metric_1:         <-- all of these from training tosses
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
    bsdf_only: ...          <-- same as bsdf_pll but without dynamics sections
    pll_only: ...           <-- same as bsdf_pll

"""

import click
import numpy as np
import os
import os.path as op
import pdb

import eval_utils, file_utils


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

    start_toss = vision_asset.split('_')[-1].split('-')[0]
    end_toss = start_toss if '-' not in vision_asset else \
        vision_asset.split('-')[1]
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

        # Tracking and dynamics metrics need to be conglomerated by toss.
        # Iterate over every comparison against, e.g. against_bundlesdf.
        for against, subsubresults in exp_subresults.items():
            # Iterate over every metric, e.g. position_error.
            for metric, trajectory_results in subsubresults.items():

                # Iterate over every trajectory, e.g. toss_1.
                toss_means = []
                toss_aucs = []
                for traj_name, reported_errors in trajectory_results.items():
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
                            keys=[category, 'full', against, metric, 'mean_auc']
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
            print(f'Skipping {subdir}.')
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
    # Start dictionaries for each of the three result types.
    bsdf_pll_results = {}
    bsdf_only_results = {}
    pll_only_results = {}

    # Iterate over every evaluation subdirectory.
    for subdir in os.listdir(file_utils.evaluation_dir()):
        eval_subdir = op.join(file_utils.evaluation_dir(), subdir)
        if not op.exists(op.join(eval_subdir, 'results.yaml')):
            print(f'  No results in {subdir}.')
            continue

        if 'bsdf' in subdir:
            add_to_results = bsdf_only_results
        elif 'pll' in subdir:
            add_to_results = pll_only_results
        elif '02' in subdir:
            add_to_results = bsdf_pll_results
        else:
            print(f'  Skipping {subdir} because looking for 02 BundleSDF IDs.')
            continue

        print(f'Found {subdir}...', end='')

        # Load the results.
        experiment_results = file_utils.load_results_yaml_in_subdir(subdir)

        # Add the experiment's results to the overall results.
        add_experiment_to_overall_results(experiment_results, add_to_results)

        # # Save the results.
        # file_utils.save_results_to_yaml(experiment_results, eval_subdir)
        print(f'done.')

    # Save the collected results.
    file_utils.save_results_to_yaml(
        bsdf_pll_results, file_utils.evaluation_dir(), filename='bsdf_pll.yaml')
    file_utils.save_results_to_yaml(
        bsdf_only_results, file_utils.evaluation_dir(),
        filename='bsdf_only.yaml')
    file_utils.save_results_to_yaml(
        pll_only_results, file_utils.evaluation_dir(), filename='pll_only.yaml')





if __name__ == '__main__':
    cli()
