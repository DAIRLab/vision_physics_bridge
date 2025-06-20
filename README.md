# cnets_data_generation
A git submodule of the Vysics repo containing the glue between BundleSDF and PLL.

## TO DO:
- [ ] link to project page
- [ ] link to Vysics and PLL repos
- [ ] docker/virtual environment information
- Document:
    - [x] dataset creation
    - [x] running experiments
    - [x] evaluation
 - Small changes:
    - [ ] Should `dair_pll` be renamed everywhere `dair_pll_vision`?

## Terminology
One Vysics run is associated with one BundleSDF run with a set of pose estimates, one PLL run, and one BundleSDF NeRF run whose shape jointly optimized visible and physible geometries.  To keep track of these results, we use the following labels:
 - `{ASSET_TYPE}` and `{EXP_NUM}` make up an `{ASSET_NAME}` via `{ASSET_TYPE}_{EXP_NUM}`, e.g. `robotocc_bottle`, `1` make up the asset `robotocc_bottle_1`.  These are the names for the experiments.
 - `{BSDF_ID}` and `{NERF_BSDF_ID}`, which can be any string.  This labels a BundleSDF run and a BundleSDF NeRF sub-run, respectively.  For one "experiment" (denoted by `{ASSET_NAME}`), there can be multiple "runs" (denoted by `{BSDF_ID}`) for generating results.  For one BundleSDF "run" (denoted by `{BSDF_ID}`), there can be multiple "NeRF runs" (denoted by `{NERF_BSDF_ID}`), always including `{BSDF_ID}` itself.  This structure is inherited by the BundleSDF repo design.
 - `{PLL_ID}` can be any string.  It similarly labels a PLL run.  Many PLL runs can be associated with the trajectory results of one BundleSDF run.
 - `{CYCLE_ITER}` is a number 1+ and represents how many iterations of BundleSDF pose estimates were in an experiment's history.  In the Vysics RSS 2025 paper, all results were cycle iteration 1, though it is possible to get higher cycle iterations by running BundleSDF for the second time as a full BundleSDF run (with new `{BSDF_ID}`) and not just a NeRF-only run (a new `{NERF_BSDF_ID}` inheriting an existing `{BSDF_ID}`); this creates a new set of pose estimates that factor in the physics insights from the last associated PLL run.  All of these count as cycle iteration 1 results:
    - Running BundleSDF for the first time.
    - Running PLL on a BundleSDF cycle iteration 1 result's pose estimates.
    - Running BundleSDF NeRF-only, using PLL cycle iteration 1 results.

## Structure of Input Data
Our dataset are converted from rosbags. See later sections for a detailed explanation for that process. To make it easier to get started, we provide sample data that are processed are ready to use. Download from the following links:
- [RGBD data](https://drive.google.com/file/d/1zxacGHDthBNOG4REeZ690BsS2fh1rKJV/view?usp=sharing)
- [Robot data](https://drive.google.com/file/d/1NIC812JiFWl6HeBUdagdTd1FR1bnSr1Q/view?usp=sharing)
- [Object shape data](https://drive.google.com/file/d/1eJwmfBEL7pL500-4vI3Uwk2eZiEYBo2N/view?usp=sharing) (for evaluation only, save to somewhere you like for now)

The downloaded data should be extracted and put in the following file structure:
```
PROJECT_ROOT
├── data
│   └── {ASSET_NAME} (e.g., robotocc_bakingbox_1)
│       ├── rgb/    (PNG files)
│       ├── depth/  (PNG files, stored in mm, uint16 format. Filename same as rgb)
│       ├── masks/  (PNG files. Filename same as rgb. 0 is background. Else is foreground)
│       └── cam_K.txt   (3x3 intrinsic matrix, use space and enter to delimit)
└── cnets-data-generation
    └── dataset
        └── {ASSET_NAME}
            ├── bundlesdf_timestamps.txt
            ├── ee_positions.txt
            ├── joint_angles.txt
            ├── joint_efforts.txt
            ├── joint_names.txt
            ├── joint_times.txt
            ├── joint_velocities.txt
            ├── synced_ee_positions.txt
            ├── synced_joint_angles.txt
            ├── synced_joint_efforts.txt
            └── synced_joint_velocities.txt
```
There are other data files that are needed for inference, but they are already included in the repo. These include:
```
PROJECT_ROOT
├── cnets-data-generation
│   ├── table_calibration/table_heights.yaml
│   ├── robot_dynamics
│   │   ├── franka_control_node.yaml
│   │   └── franka_hw_controllers.yaml
│   └── assets
│       ├── true_urdfs/bsdf_mesh_average_dynamics.urdf
│       ├── cam_K_robotocc.txt
│       ├── config.yaml
│       └── realsense_pose_robotocc.yaml
└── dair_pll
    └── assets
        ├── precomputed_vision_functions/...
        ├── franka_with_ee.urdf
        ├── franka_without_geom_with_ee.urdf
        └── vision_template.urdf
```
If you create your own dataset, you may need to modify these files. 

## Running Experiments
A script for running a Vysics experiment after obtaining/generating the dataset is provided as `run_all.sh`. It contains 6 steps:
1. **Run BundleSDF with tracking and NeRF:**  generates pose estimates and a vision-only shape estimate.  _Run from `vysics` repo._
2. **Convert from BundleSDF to PLL:**  converts trajectory to PLL format and extracts vision insights to factor into physics learning.  _Run from this `cnets-data-generation` repo._
3. **Run PLL:**  generates a physics-based shape estimate and other dynamics parameter estimates.  _Run from `dair_pll_vision` repo._ **[TODO change name]**
4. **Convert from PLL to BundleSDF:**  moves physics-based shape information into vision input directory.  _Run from this `cnets-data-generation` repo._
5. **Run BundleSDF with NeRF only:**  generates a vision- and physics-informed new shape estimate.  _Run from `vysics` repo._
6. **Convert from BundleSDF to PLL:**  required step before evaluating.  _Run from this `cnets-data-generation` repo._


### Running BundleSDF
Start BundleSDF runs from the `vysics` **[TODO name?]** repo.

For running a BundleSDF _tracking and NeRF_ run from scratch with BundleSDF ID `{BSDF_ID}`:
```
python bundlenets/run_custom.py --vision-asset={ASSET_NAME} --run-name={BSDF_ID} --use-gui=0 --debug-level=2
```

For running a BundleSDF NeRF run only with NeRF ID `{NERF_BSDF_ID}` under an existing BundleSDF tracking result with ID `{BSDF_ID}` using supervision from PLL run `{PLL_ID}`:
```
python bundlenets/run_custom.py --vision-asset={ASSET_NAME} --tracking-run-name={BSDF_ID} --share-tracking --run-name={NERF_BSDF_ID} --pll-id={PLL_ID} --use-gui=0 --debug-level=2
```
For both, add `--cycle-iteration={CYCLE_ITER}` as necessary.


### Converting from BundleSDF to PLL
After iteration 1 of BundleSDF run with BundleSDF ID `{BSDF_ID}`:
```
python conversion_bsdf_to_pll.py --vision-asset={ASSET_NAME} --bundlesdf-id={BSDF_ID}
```
By default, this assumes `{BSDF_ID}` is a cycle iteration 1 BundleSDF run, and that the NeRF geometry result to convert is under the shared ID `{BSDF_ID}`.  To change either, include the flags `--cycle-iteration={CYCLE_ITER}` and/or `--nerf-bundlesdf-id={NERF_BSDF_ID}`.

BundleSDF-to-PLL conversion does the following:
 - **Pose conversion:**  Converts the camera-frame, 4x4 homogeneous pose estimate transforms into PLL format, which uses world-frame coordinates, represents orientation with quaternions, includes velocity estimates, and includes Franka states, if the experiment featured any robot interactions.
 - **Visible geometry insights:**  Generates sampled points on the visible portion of the BundleSDF shape estimate's convex hull.

There are other flags that may be of interest for faster conversions; see [the BundleSDF-to-PLL converter](./conversion_bsdf_to_pll.py) for documentation.

### Running PLL
Start a PLL run from the `dair_pll` (**[TODO rename]**) repo with:
```
python examples/contactnets_vision.py --vision-asset={ASSET_NAME} --run-name={PLL_ID} --bundlesdf-id={BSDF_ID}
```
By default, this uses the pose _and_ geometry estimates from `{BSDF_ID}` as both the BundleSDF and NeRF IDs in cycle iteration 1.  Add `--nerf-bundlesdf-id={NERF_BSDF_ID}` and/or `--cycle-iteration={CYCLE_ITER}` as necessary.

There are other flags that may be of interest such as adjusting loss weights; see **[TODO: link?]** the Vysics PLL script for documentation.


### Converting from PLL to BundleSDF
After iteration 1 of PLL run with PLL ID `{PLL_ID}`:
```
python conversion_pll_to_bsdf.py --vision-asset={ASSET_NAME} --pll-id={PLL_ID}
```
Add the appropriate `--cycle-iteration={CYCLE_ITER}` if necessary.

PLL-to-BundleSDF conversion does the following:
 - **Physible geometry insights:**  Moves geometry outputs from PLL into input directories accessible to future BundleSDF runs.

## Structure of Result Files
With the introduced terminology, locations for results are:
 - Pose estimates come from running BundleSDF.  These are represented as 4x4 homogeneous transforms of the object pose in the camera frame.  The estimates are located within a BundleSDF output folder:
    ```
    results/{ASSET_NAME}/bundlesdf_iteration_{CYCLE_ITER}/bundlesdf_id_{BSDF_ID}/ob_in_cam/
    ```
    - To be usable by downstream PLL runs, these poses need to be converted into PLL format which will be located at a PLL input folder: **[TODO `dair_pll` rename]**
        ```
        dair_pll/assets/vision_{ASSET_TYPE}/{ASSET_NAME}/toss/bundlesdf_iteration_{CYCLE_ITER}/bundlesdf_id_{BSDF_ID}/{EXP_NUM}.pt
        ```
    - The `toss` subdirectory indicates trajectories where all of the contained dynamics can be explained by object-table and object-robot interactions, compared to the `full` subdirectory whose trajectories can include unmodeled human interventions.  For Vysics RSS 2025 experiments, there were no human interventions, so `toss` and `full` trajectories are the same in length and content.
 - Visible shape estimates come from running a NeRF experiment under a BundleSDF pose tracking experiment.  The shape estimate is located within a BundleSDF output folder:
    ```
    results/{ASSET_NAME}/bundlesdf_iteration_{CYCLE_ITER}/bundlesdf_id_{BSDF_ID}/nerf_runs/bundlesdf_id_{NERF_BSDF_ID}/mesh_cleaned.obj (normalized)
    
    results/{ASSET_NAME}/bundlesdf_iteration_{CYCLE_ITER}/bundlesdf_id_{BSDF_ID}/nerf_runs/bundlesdf_id_{NERF_BSDF_ID}/textured_mesh.obj (true scale)
    ```
    - To be usable by downstream PLL runs, this visible mesh needs to produce a set of points sampled on the visible portions of the convex hull of the above .obj file, subject to our vision-based PLL supervision loss (paper Eq. (6)).  This set of points, along with other geometry information, will be located within a PLL input folder:
      ```
      dair_pll/assets/vision_{ASSET_TYPE}/{ASSET_NAME}/geom_for_pll/bundlesdf_iteration_{CYCLE_ITER}/bundlesdf_id_{BSDF_ID}/
      ```
 - Physics parameters, including physible shape, come from running a PLL experiment and are located in a PLL output folder:  **[TODO `with_bundlesdf_mesh.urdf` is misleadingly named]** **[TODO might need to change `dair_pll`]**
    ```
    dair_pll/results/vision_{ASSET_TYPE}/{ASSET_NAME}/bundlesdf_iteration_{CYCLE_ITER}/pll_id_{PLL_ID}/urdfs/
      - with_bundlesdf_mesh.urdf
      - body_best.obj
    ```
    - To be usable by downstream BundleSDF runs, the PLL geometry estimate needs to produce a set of points and associated signed distances subject to our support point loss (paper Eq. (8)) and a set of points and associated signed distance lower bounds subject to our hyperplane-constrained loss (paper Eq. (11)).  These sets of points, along with other geometry information, will be located within a BundleSDF input folder:
      ```
      geometry/{ASSET_NAME}/bundlesdf_iteration_{CYCLE_ITER}/pll_id_{PLL_ID}/
      ```
   - This BundleSDF input directory contains `from_support_points` and `from_mesh_surface` subfolders.  It worked well from our experience to supervise via support point loss (paper Eq. (8)) with data in `from_support_points` and via hyperplane-constrained loss (paper Eq. (11)) with data in `from_mesh_surface`.  This helps ensure the stricter support point loss is applied on points that confidently hypothesized contact, while the less strict hyperplane-constrained loss can still be applied more broadly without encouraging unnecessary geometric convexity where visible signals may say otherwise.

## Evaluation
### Align the Ground Truth Mesh to the Estimation
We use a combination of manual alignment and ICP to align the meshes. You can use the following script to do the manual part through keyboard, and the ICP will follow automatically:
```
python mesh_processing.py manual_icp --vision-asset={ASSET_NAME} --bundlesdf-id={BSDF_ID} --nerf-bundlesdf-id={NERF_BSDF_ID} --cycle-iteration=1
```
Or you can manually align the meshes using external tools (e.g., meshlab), and put the ground truth mesh that is aligned with the `textured_mesh.obj` output mesh to the follow path:
```
results/{ASSET_NAME}/bundlesdf_iteration_{CYCLE_ITER}/bundlesdf_id_{BSDF_ID}/nerf_runs/bundlesdf_id_{NERF_BSDF_ID}/true_geom_aligned_meshlab.obj
```
Then run `mesh_processing.py` with an extra flag `--meshlab`, which will only do the ICP part.

Either way, a folder for saving the evaluation results of this experiment will be created: 
```
cnets-data-generation/evaluation/{ASSET_NAME}_bsdf_{BSDF_ID}_{NERF_BSDF_ID}_{CYCLE_ITER}/
```
and the aligned ground truth mesh will be saved to `true_geom_aligned_assist.obj` under this folder.

### Geometry Evaluation for an Experiment
Evaluating a Vysics or BundleSDF run for geometry metrics can be done with [the evaluation script](evaluate.py) via:
```
python evaluate.py --vision-asset={ASSET_NAME} --bundlesdf-id={BSDF_ID} --nerf-bundlesdf-id={NERF_BSDF_ID} --cycle-iteration={CYCLE_ITER}
```

Other flags of interest include `--do-videos` for generating helpful videos, among others.  See [the evaluation script](evaluate.py) for documentation.

This script saves evaluation results to the evaluation folder mentioned above, including `results.yaml`, which reports the geometric errors. 

### Dynamics Evaluation for an Experiment
Generate dynamics predictions for Vysics, BundleSDF, PLL, and/or ground truth geometry with robot interactions using [the robot dynamics predictions script](robot_dynamics_predictions.py) via:
```
python robot_dynamics_predictions.py gen {ASSET_NAME} vysics bsdf pll gt --debug
```
...where you can specify any subset of `vysics`, `bsdf`, `pll`, and `gt` flags if you want to run simulation on the output of certain methods.

> ⚠️ **NOTE:**  This requires editing / adding an entry to the `PLL_BSDF_NERF_IDS_FROM_VISION_ASSET` dictionary in [robot_dynamics_predictions.py](robot_dynamics_predictions.py) so it knows what experiment runs (specified by `{BSDF_ID}`, `{NERF_BSDF_ID}`, and `{PLL_ID}`) should be associated with `vysics`, `bsdf`, and `pll` for the given `{ASSET_NAME}`.  

This script creates a folder at:
```
cnets-data-generation/robot_dynamics/{ASSET_NAME}/
```
...containing numerical and visual simulation results.


## Analyzing Results
### Quantitative Exports
Analyze results from many runs with [the gather results script](gather_results.py) via:
```
python gather_results.py gather
```
This traverses through all of the subfolders within each of these directories:
```
cnets-data-generation/evaluation/
cnets-data-generation/robot_dynamics/
```
...and writes yaml files into the `evaluation` directory for each experiment type, e.g. `bsdf_pll_robotocc.yaml` for Vysics **[TODO could make this `vysics.yaml` instead]**, `bsdf_robotocc.yaml` for BundleSDF, and `pll_robotocc.yaml` for PLL.

> ⚠️ **NOTE:**  This requires editing / adding lines to the `process_gather_command` function to help it interpret what subfolders of the `evaluation` directory should be included under Vysics, BundleSDF, PLL, or other labels you define.  You can use different matching patterns for the experiments and export them to different yaml files as desired.

### Visual Exports
Generate plots resulting from the exported results from the `gather` command above with [the gather results script](gather_results.py) via:
```
python gather_results.py plot
```
This opens every expected generated yaml file from the `gather` command and generates plots in the plot subdirectory:
```
cnets-data-generation/plots/
```
Currently this will make a `MMDD` subdirectory for the given month and day, then within which there will be geometry and dynamics plots.


# Creating a New Dataset
This repo includes [a dataset creation script](create_dataset.py) to generate new datasets from ROS bag inputs.  There are a few manual steps required in addition to running this script.

1. Define a new `{ASSET_NAME}` and annotate the interactions in the dataset by editing the experiment configuration file at [assets/config.yaml](./assets/config.yaml).  This requires adding the following keys/entries into the file:
   ```
   dataset:
     {ASSET_TYPE}:
       0:  # Leave toss 0 blank
       1: {BAG_NUM}
       ... # Repeat for every {EXP_NUM} for the given {ASSET_TYPE}
   tosses:
      {ASSET_TYPE}:
         - toss: 0   # Leave toss 0 details blank
         - toss: 1
           start_time:
             secs: X
             nsecs: Y
           end_time:
             secs: Z
             nsecs: W
         ... # Repeat for every {EXP_NUM} for the given {ASSET_TYPE}
   ```
2. Put `raw_{BAG_NUM}.bag` into the `cnets-data-generation/rosbags/` folder.  As designed, this ROS bag needs to contain the following topics:
   - `/camera/aligned_depth_to_color/image_raw` for depth images
   - `/camera/color/image_raw` for RGB images
   - `/panda/joint_states` for Franka joint states
   
   For further customization of extracting data from ROS bags, edits can be made to [the ROS bag processor](rosbag_processor.py) and how that functionality gets called from [the dataset creator](create_dataset.py).

3. Create files for camera parameters. For Vysics data, we save extrinsics at: `cnets-data-generation/assets/realsense_pose_robotocc.yaml`; intrinsics at: `cnets-data-generation/assets/cam_K_robotocc.txt`. 

4. (Optional) Compensate for camera depth offsets. There can be a bias in the depth camera returns that do not fully line up with the camera extrinsic calibration, which usually relies on the RGB sensor. The magnitude of this bias can be a few millimeters. If high precision is desired, you may want to calibrate the depth bias with respect to some external reference measurements (e.g., from TagSLAM) if available. 

   To calibrate the depth bias, uncomment `inspect_camera_alignments.interactive_offset_adjustment()` in `create_dataset.py` when you generate the data of a new experiment for the first time. Then run 
   ```
   python create_dataset.py --vision-asset=${ASSET_NAME}
   ```
   An interactive GUI will guide you to find the optimal bias value. 

   Use the result to hardcode `self.depth_offset_mm` in `self._create_images()` in `create_dataset.py`. Comment `inspect_camera_alignments.interactive_offset_adjustment()`. 

5. Generate the dataset for an asset from the rosbag via:
   ```
   python create_dataset.py --vision-asset={ASSET_NAME}
   ```
   This writes RGB, depth, and camera intrinsics to a `data/{ASSET_NAME}/` folder.  It also writes timestamps associated with each image, robot states, and synchronized robot states for each image timestamp to a `cnets-data-generation/dataset/{ASSET_NAME}/` folder. 

   When creating the dataset, it may call `compute_table_offsets.py`, which estimates the height of the table surface from the data and saves the result to 
   ```
   cnets-data-generation/table_calibration/
      - {ASSET_NAME}.txt
      - {ASSET_NAME}.png
      - {ASSET_NAME}_eps.png
   ```
   You can also run the script manually with matplotlib-backed GUI guidance: 
   ```
   python compute_table_offsets.py single --vision-asset={ASSET_NAME}
   ```

6. Combine the table height information into a single file. 
   A subsequent call to:
   ```
   python compute_table_offsets.py combine
   ```
   ...will combine the individual experiment results compatibly into [table_heights.yaml](table_calibration/table_heights.yaml), which contains a key `{ASSET_TYPE}` with sub-keys for every contained `{EXP_NUM}` with a float for the table height to use in each experiment.

   You may avoid running `python compute_table_offsets.py single --vision-asset={ASSET_NAME}` for every `{ASSET_NAME}` and reuse a single height for many `{ASSET_NAME}`'s by editing the `get_table_height_from_log` function in [the offset helper script](compute_table_offsets.py) to detect from the `{ASSET_NAME}` what offset to use.  The `combine` command will still be required after these changes in order to write the values to [table_heights.yaml](table_calibration/table_heights.yaml). 

7. Create object masks.  The masks need to be in the `data/{ASSET_NAME}/masks/` folder.  This can be done a number of ways, and ultimately [XMem](https://github.com/hkchengrex/XMem) worked sufficiently well for us.  **[TODO any other information required here?]**

After these steps, experiments can be run on the new `{ASSET_NAME}` via the [experiment instructions](#running-experiments).
