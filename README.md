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

## Structure and Terminology
One Vysics run is associated with one BundleSDF run with a set of pose estimates, one PLL run, and one BundleSDF NeRF run whose shape jointly optimized visible and physible geometries.  To keep track of these results, we use the following labels:
 - `{ASSET_TYPE}` and `{EXP_NUM}` make up an `{ASSET_NAME}` via `{ASSET_TYPE}_{EXP_NUM}`, e.g. `robotocc_bottle`, `1` make up the asset `robotocc_bottle_1`.  These are the names for the experiments.
 - `{BSDF_ID}` and `{NERF_BSDF_ID}`, which can be any string.  This labels a BundleSDF run and a BundleSDF NeRF sub-run, respectively.  For one "experiment" (denoted by `{ASSET_NAME}`), there can be multiple "runs" (denoted by `{BSDF_ID}`) for generating results.  For one BundleSDF "run" (denoted by `{BSDF_ID}`), there can be multiple "NeRF runs" (denoted by `{NERF_BSDF_ID}`), always including `{BSDF_ID}` itself.  This structure is inherited by the BundleSDF repo design.
 - `{PLL_ID}` can be any string.  It similarly labels a PLL run.  Many PLL runs can be associated with the trajectory results of one BundleSDF run.
 - `{CYCLE_ITER}` is a number 1+ and represents how many iterations of BundleSDF pose estimates were in an experiment's history.  In the Vysics RSS 2025 paper, all results were cycle iteration 1, though it is possible to get higher cycle iterations by running BundleSDF for the second time as a full BundleSDF run (with new `{BSDF_ID}`) and not just a NeRF-only run (a new `{NERF_BSDF_ID}` inheriting an existing `{BSDF_ID}`); this creates a new set of pose estimates that factor in the physics insights from the last associated PLL run.  All of these count as cycle iteration 1 results:
    - Running BundleSDF for the first time.
    - Running PLL on a BundleSDF cycle iteration 1 result's pose estimates.
    - Running BundleSDF NeRF-only, using PLL cycle iteration 1 results.

With this terminology in mind, locations for results are:
 - Pose estimates come from running BundleSDF.  These are represented as 4x4 homogeneous transforms of the object pose in the camera frame.  The estimates are located within a BundleSDF output folder:
    ```
    vysics/results/{ASSET_NAME}/bundlesdf_iteration_{CYCLE_ITER}/bundlesdf_id_{BSDF_ID}/ob_in_cam/
    ```
    - To be usable by downstream PLL runs, these poses need to be converted into PLL format which will be located at a PLL input folder: **[TODO `dair_pll` rename]**
        ```
        vysics/dair_pll/assets/vision_{ASSET_TYPE}/{ASSET_NAME}/toss/bundlesdf_iteration_{CYCLE_ITER}/bundlesdf_id_{BSDF_ID}/{EXP_NUM}.pt
        ```
    - The `toss` subdirectory indicates trajectories where all of the contained dynamics can be explained by object-table and object-robot interactions, compared to the `full` subdirectory whose trajectories can include unmodeled human interventions.  For Vysics RSS 2025 experiments, there were no human interventions, so `toss` and `full` trajectories are the same in length and content.
 - Visible shape estimates come from running a NeRF experiment under a BundleSDF pose tracking experiment.  The to-scale shape estimate is located within a BundleSDF output folder:
    ```
    vysics/results/{ASSET_NAME}/bundlesdf_iteration_{CYCLE_ITER}/bundlesdf_id_{BSDF_ID}/nerf_runs/bundlesdf_id_{NERF_BSDF_ID}/mesh_cleaned.obj
    ```
    - To be usable by downstream PLL runs, this visible mesh needs to produce a set of points sampled on the visible portions of the convex hull of the above .obj file, subject to our vision-based PLL supervision loss (paper Eq. (6)).  This set of points, along with other geometry information, will be located within a PLL input folder:
      ```
      vysics/dair_pll/assets/vision_{ASSET_TYPE}/{ASSET_NAME}/geom_for_pll/bundlesdf_iteration_{CYCLE_ITER}/bundlesdf_id_{BSDF_ID}/
      ```
 - Physics parameters, including physible shape, come from running a PLL experiment and are located in a PLL output folder:  **[TODO `with_bundlesdf_mesh.urdf` is misleadingly named]** **[TODO might need to change `dair_pll`]**
    ```
    vysics/dair_pll/results/vision_{ASSET_TYPE}/{ASSET_NAME}/bundlesdf_iteration_{CYCLE_ITER}/pll_id_{PLL_ID}/urdfs/
      - with_bundlesdf_mesh.urdf
      - body_best.obj
    ```
    - To be usable by downstream BundleSDF runs, the PLL geometry estimate needs to produce a set of points and associated signed distances subject to our support point loss (paper Eq. (8)) and a set of points and associated signed distance lower bounds subject to our hyperplane-constrained loss (paper Eq. (11)).  These sets of points, along with other geometry information, will be located within a BundleSDF input folder:
      ```
      vysics/geometry/{ASSET_NAME}/bundlesdf_iteration_{CYCLE_ITER}/pll_id_{PLL_ID}/
      ```
   - This BundleSDF input directory contains `from_support_points` and `from_mesh_surface` subfolders.  It worked well from our experience to supervise via support point loss (paper Eq. (8)) with data in `from_support_points` and via hyperplane-constrained loss (paper Eq. (11)) with data in `from_mesh_surface`.  This helps ensure the stricter support point loss is applied on points that confidently hypothesized contact, while the less strict hyperplane-constrained loss can still be applied more broadly without encouraging unnecessary geometric convexity where visible signals may say otherwise.

# Running Experiments
The overview steps for running a Vysics experiment after obtaining/generating the dataset are:
1. **Run BundleSDF with tracking and NeRF:**  generates pose estimates and a vision-only shape estimate.  _Run from `vysics` repo._
2. **Convert from BundleSDF to PLL:**  converts trajectory to PLL format and extracts vision insights to factor into physics learning.  _Run from this `cnets-data-generation` repo._
3. **Run PLL:**  generates a physics-based shape estimate and other dynamics parameter estimates.  _Run from `dair_pll_vision` repo._ **[TODO change name]**
4. **Convert from PLL to BundleSDF:**  moves physics-based shape information into vision input directory.  _Run from this `cnets-data-generation` repo._
5. **Run BundleSDF with NeRF only:**  generates a vision- and physics-informed new shape estimate.  _Run from `vysics` repo._
6. **Convert from BundleSDF to PLL:**  required step before evaluating.  _Run from this `cnets-data-generation` repo._
7. **Run evaluations.**  _Run from this `cnets-data-generation` repo._


## Running BundleSDF
Start BundleSDF runs from the `vysics` **[TODO name?]** repo.

For running a BundleSDF _tracking and NeRF_ run from scratch with BundleSDF ID `{BSDF_ID}`:
```
python bundlenets/run_custom.py --vision-asset={ASSET_NAME} --run-name={BSDF_ID} --use-segmenter=1 --use-gui=0 --debug-level=2  # TODO check this
```

For running a BundleSDF NeRF run only with NeRF ID `{NERF_BSDF_ID}` under an existing BundleSDF tracking result with ID `{BSDF_ID}` using supervision from PLL run `{PLL_ID}`:
```
python bundlenets/run_custom.py --vision-asset={ASSET_NAME} --tracking-run-name={BSDF_ID} --share-tracking --run-name={NERF_BSDF_ID} --pll-id={PLL_ID} --use-segmenter=1 --use-gui=0 --debug-level=2  # TODO check this
```
For both, add `--cycle-iteration={CYCLE_ITER}` as necessary.


## Converting from BundleSDF to PLL
After iteration 1 of BundleSDF run with BundleSDF ID `{BSDF_ID}`:
```
python conversion_bsdf_to_pll.py --vision-asset={ASSET_NAME} --bundlesdf-id={BSDF_ID}
```
By default, this assumes `{BSDF_ID}` is a cycle iteration 1 BundleSDF run, and that the NeRF geometry result to convert is under the shared ID `{BSDF_ID}`.  To change either, include the flags `--cycle-iteration={CYCLE_ITER}` and/or `--nerf-bundlesdf-id={NERF_BSDF_ID}`.

BundleSDF-to-PLL conversion does the following:
 - **Pose conversion:**  Converts the camera-frame, 4x4 homogeneous pose estimate transforms into PLL format, which uses world-frame coordinates, represents orientation with quaternions, includes velocity estimates, and includes Franka states, if the experiment featured any robot interactions.
 - **Visible geometry insights:**  Generates sampled points on the visible portion of the BundleSDF shape estimate's convex hull.

There are other flags that may be of interest for faster conversions; see [the BundleSDF-to-PLL converter](./conversion_bsdf_to_pll.py) for documentation.

## Running PLL
Start a PLL run from the `dair_pll` (**[TODO rename]**) repo with:
```
python examples/contactnets_vision.py --vision-asset={ASSET_NAME} --run-name={PLL_ID} --bundlesdf-id={BSDF_ID}
```
By default, this uses the pose _and_ geometry estimates from `{BSDF_ID}` as both the BundleSDF and NeRF IDs in cycle iteration 1.  Add `--nerf-bundlesdf-id={NERF_BSDF_ID}` and/or `--cycle-iteration={CYCLE_ITER}` as necessary.

There are other flags that may be of interest such as adjusting loss weights; see **[TODO: link?]** the Vysics PLL script for documentation.


## Converting from BundleSDF to PLL
After iteration 1 of PLL run with PLL ID `{PLL_ID}`:
```
python conversion_pll_to_bsdf.py --vision-asset={ASSET_NAME} --pll-id={PLL_ID}
```
Add the appropriate `--cycle-iteration={CYCLE_ITER}` if necessary.

PLL-to-BundleSDF conversion does the following:
 - **Physible geometry insights:**  Moves geometry outputs from PLL into input directories accessible to future BundleSDF runs.


## Evaluating
### Geometry Evaluation for a Single Run
Evaluating a Vysics, BundleSDF, or PLL run for geometry metrics can be done with [the evaluation script](evaluate.py) via:
```
python evaluate.py --vision-asset={ASSET_NAME} --bundlesdf-id={BSDF_ID} --nerf-bundlesdf-id={NERF_BSDF_ID} --pll-id={PLL_ID} --cycle-iteration={CYCLE_ITER}
```
 - For a Vysics run, include the `{BSDF_ID}` and `{NERF_BSDF_ID}` from the last BundleSDF run (exclude any `{PLL_ID}`).
 - For a BundleSDF run, include the `{BSDF_ID}` (and `{NERF_BSDF_ID}` as necessary; there should be no `{PLL_ID}`).
 - For a PLL run, include only `{PLL_ID}` (exclude `{BSDF_ID}` and `{NERF_BSDF_ID}`).

Other flags of interest include `--do-videos` for generating helpful videos, among others.  See [the evaluation script](evaluate.py) for documentation.

This script creates a folder at:
```
# For Vysics and BundleSDF runs:
vysics/cnets-data-generation/evaluation/{ASSET_NAME}_bsdf_{BSDF_ID}_{NERF_BSDF_ID}_{CYCLE_ITER}/

# For PLL runs:
vysics/cnets-data-generation/evaluation/{ASSET_NAME}_pll_{PLL_ID}_{CYCLE_ITER}/
```
...containing files including `results.yaml`, which reports the geometric errors.  **[TODO this requires having a GT aligned mesh somehow...]**


### Dynamics Evaluation for a Given Experiment
Generate dynamics predictions for Vysics, BundleSDF, PLL, and/or ground truth geometry with robot interactions using [the robot dynamics predictions script](robot_dynamics_predictions.py) via:
```
python robot_dynamics_predictions.py gen {ASSET_NAME} vysics bsdf pll gt --debug
```
...where you can exclude any of the `vysics`, `bsdf`, `pll`, and `gt` flags if not all of them are desired.

> ⚠️ **NOTE:**  This requires editing / adding an entry to the `PLL_BSDF_NERF_IDS_FROM_VISION_ASSET` dictionary in [robot_dynamics_predictions.py](robot_dynamics_predictions.py) so it knows what Vysics, BundleSDF, and PLL runs are associated with each other for the given `{ASSET_NAME}`.  This means only one set of Vysics, BundleSDF, and PLL runs can be associated with a given `{ASSET_NAME}` at a time.

This script creates a folder at:
```
vysics/cnets-data-generation/robot_dynamics/{ASSET_NAME}/
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
vysics/cnets-data-generation/evaluation/
vysics/cnets-data-generation/robot_dynamics/
```
...and writes yaml files into the `evaluation` directory for each experiment type, e.g. `bsdf_pll.yaml` for Vysics **[TODO could make this `vysics.yaml` instead]**, `bsdf.yaml` for BundleSDF, and `pll.yaml` for PLL.

> ⚠️ **NOTE:**  This requires editing / adding lines to the `process_gather_command` function to help it interpret what kind of experiment the subfolders of the `evaluation` directory are.  You can define different types of experiments and export them to different yaml files as desired.

### Visual Exports
Generate plots resulting from the exported results from the `gather` command above with [the gather results script](gather_results.py) via:
```
python gather_results.py plot
```
This opens every expected generated yaml file from the `gather` command and generates plots in the plot subdirectory:
```
vysics/cnets-data-generation/plots/
```
Currently this will make a `MMDD` subdirectory for the given month and day, then within which there will be geometry and dynamics plots.


# Creating a New Dataset
This repo includes [a dataset creation script](create_dataset.py) to generate new datasets from ROS bag inputs.  There are a few manual steps required in addition to running this script.

1. Define a new `{ASSET_NAME}` and annotate the interactions in the dataset by editing [the experiment configuration file](./assets/config.yaml).  This requires adding the following keys/entries into the file:
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
2. Put `raw_{BAG_NUM}.bag` into the `vysics/cnets-data-generation/rosbags/` folder.  As designed, this ROS bag needs to contain the following topics:
   - `/camera/aligned_depth_to_color/image_raw` for depth images
   - `/camera/color/image_raw` for RGB images
   - `/panda/joint_states` for Franka joint states
   
   For further customization of extracting data from ROS bags, edits can be made to [the ROS bag processor](rosbag_processor.py) and how that functionality gets called from [the dataset creator](create_dataset.py).

3. Compensate for camera depth offsets.  The requirement for a new dataset is that [table_heights.yaml](table_calibration/table_heights.yaml) contains a key `{ASSET_TYPE}` with sub-keys for every contained `{EXP_NUM}` with a float for the offset to impose on each experiment.  If undesired, these keys should still exist but the offset can be set to 0.  There can be a bias in the depth camera returns that do not fully line up with the camera extrinsic calibration, which usually relies on the RGB sensor.  If higher precision is desired, this file is the opportunity to adjust that offset.  This can be done manually in the file, or there is [an offset helper script](compute_table_offsets.py) with some utilities to assist in finding a depth reading offset to impose for better alignment between the RGB and D sensors.  For example, you can use a matplotlib-backed GUI for a single experiment via:
   ```
   python compute_table_offsets.py single --vision-asset={ASSET_NAME}
   ```
   This writes some files:
   ```
   vysics/cnets-data-generation/table_calibration/
      - {ASSET_NAME}.txt
      - {ASSET_NAME}.png
      - {ASSET_NAME}_eps.png
   ```
   A subsequent call to:
   ```
   python compute_table_offsets.py combine
   ```
   ...will combine the individual experiment results compatibly into [table_heights.yaml](table_calibration/table_heights.yaml).

   Alternative to running the `single` command and to reuse a single offset for many experiments is to edit the `get_table_height_from_log` function in [the offset helper script](compute_table_offsets.py) to detect from the `{ASSET_NAME}` what offset to use.  The `combine` command will still be required after these changes in order to write the values to [table_heights.yaml](table_calibration/table_heights.yaml).

4. Write dataset files via:
   ```
   python create_dataset.py --vision-asset={ASSET_NAME}
   ```
   This requires the previous steps to be completed and writes RGB, depth, and camera intrinsics to a `vysics/data/{ASSET_NAME}/` folder.  It also writes timestamps associated with each image, robot states, and synchronized robot states for each image timestamp to a `vysics/cnets-data-generation/dataset/{ASSET_NAME}/` folder.

5. Create object masks.  The masks need to be in the `vysics/data/{ASSET_NAME}/masks/` folder.  This can be done a number of ways, and ultimately [XMem](https://github.com/hkchengrex/XMem) worked sufficiently well for us.  **[TODO any other information required here?]**

After these steps, experiments can be run on the new `{ASSET_NAME}` via the [experiment instructions](#running-experiments).
