'''
The config.yaml file describes each object and the rosbag files and the timestamps for each experiment of the object.
The file looks like this: 
dataset:
  bakingbox: 
    0: 
    1: 100
    2: 100
    3: 100
  robotoccdual_bottle: 
    0: 
    1: 101
    2: 101
Find the id of experiments that share a rosbag for an object. 
For example, if 1-6 are using rosbag 100 and 7-10 are using rosbag 101, return 1-6 and 7-10. 
Then, create the dataset for as many experiments in a continuous rosbag as possible, using 
the obtained ids. 
Also output a file recording the ids: assets/config_obj_bag_to_exps.yaml: 
robotocc_egg:
  312:
  - 1
  - 6
robotocc_milk:
  305:
  - 6
  - 6
  308:
  - 1
  - 5
'''

import yaml
import os
import time

config_file = 'assets/config.yaml'

with open(config_file, 'r') as stream:
    config = yaml.safe_load(stream)

objs_bags_to_exp = {}
for object in config['dataset']:
    print(object)
    obj_bag_to_exp = {}
    for exp in config['dataset'][object]:
        if exp == 0:
            continue
        rosbag_id = config['dataset'][object][exp]
        if rosbag_id not in obj_bag_to_exp:
            obj_bag_to_exp[rosbag_id] = []
        obj_bag_to_exp[rosbag_id].append(exp)

    obj_bag_to_exp_minmax = {}
    for rosbag_id in obj_bag_to_exp:
        exp_min = min(obj_bag_to_exp[rosbag_id])
        exp_max = max(obj_bag_to_exp[rosbag_id])
        obj_bag_to_exp_minmax[rosbag_id] = [exp_min, exp_max]

    objs_bags_to_exp[object] = obj_bag_to_exp_minmax

# Save the results to a yaml file
output_file = 'assets/config_obj_bag_to_exps.yaml'
with open(output_file, 'w') as stream:
    yaml.dump(objs_bags_to_exp, stream)

# For each object, for each rosbag, run the command: 
# python create_dataset.py --vision-asset=${object}_${min}_${max} if min != max
time_start = time.time()
for object in objs_bags_to_exp:
    if 'robotocc_' not in object:
        continue
    for rosbag_id in objs_bags_to_exp[object]:
        exp_min = objs_bags_to_exp[object][rosbag_id][0]
        exp_max = objs_bags_to_exp[object][rosbag_id][1]
        if exp_min == exp_max:
            continue
        time_now = time.time()
        command = f'python create_dataset.py --vision-asset={object}_{exp_min}-{exp_max}'
        print(command)
        os.system(command)
        print(f'===========Time taken: {time.time() - time_now}')
print(f'===========Total time taken: {time.time() - time_start}')
