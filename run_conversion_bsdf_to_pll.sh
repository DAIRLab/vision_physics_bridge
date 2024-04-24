#!/bin/sh

### Convert pll output to bundlesdf input in the geometry folder

# source /opt/ros/noetic/setup.bash
# source ../../robot_filter/venv/bin/activate

# python conversion_bsdf_to_pll.py --vision-asset=cube_1 --bundlesdf-id=00 --cycle-iteration=1
# python conversion_bsdf_to_pll.py --vision-asset=cube_2 --bundlesdf-id=00 --cycle-iteration=1
# python conversion_bsdf_to_pll.py --vision-asset=cube_3 --bundlesdf-id=00 --cycle-iteration=1
python conversion_bsdf_to_pll.py --vision-asset=cube_4 --bundlesdf-id=00 --cycle-iteration=1