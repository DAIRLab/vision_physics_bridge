#!/bin/sh

asset=bakingbox_1
python mesh_processing.py manual_icp --vision-asset=robotocc_${asset} --bundlesdf-id=00-onescript --nerf-bundlesdf-id=00-onescript --cycle-iteration=1 --meshlab
python evaluate.py --vision-asset=robotocc_${asset} --bundlesdf-id=00-onescript --nerf-bundlesdf-id=00-onescript --cycle-iteration=1 #--do-videos