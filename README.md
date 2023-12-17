# cnets_data_generation
A package that prepares images and pose data for 6D pose tracking. 

### Data generation
To process a new object, you first need to add the start time and end time of each toss to [config.yaml](./assets/config.yaml). To process a rosbag, run `example_{object}.py`. Replace `ROSBAG_NAME` and `ODOM_ROSBAG_NAME` with `raw_XXX.bag` and `odom_XXX.bag`. Replace `TOSS_ID` with your toss id. Every object has its own file.

### Pose Evaluation
To evaluate a trajectory tracking, run `eval_{object}.py` with your `toss_id`. Replace `ROSBAG_NAME` and `ODOM_ROSBAG_NAME` with `raw_XXX.bag` and `odom_XXX.bag`. Replace `TOSS_ID` with your toss id. 

### Benchmark a Video
Run `experiment.py` with correct `type` and `toss_id`. This script will output ADD, ADDS and chamfer distance which are used as metrics. Below is an example:
```
python3 experiment.py --type=cube --toss_id=1
```

### Data Conversion
To convert poses in `ob_in_cam` from BundleSDF results to `dair_pll` format, you need to run `data_preparation.py` with correct `toss_id`, `type` arguments. `use_gt` means whether generate trajectory data with `TagSLAM` output or not.