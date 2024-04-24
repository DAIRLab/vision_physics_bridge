#!/bin/sh

### Convert pll output to bundlesdf input in the geometry folder

# source /opt/ros/noetic/setup.bash
# source ../../robot_filter/venv/bin/activate

### iter 0
# python conversion_pll_to_bsdf.py --vision-asset=cube_1 --pll-id=mt00 --cycle-iteration=0
# python conversion_pll_to_bsdf.py --vision-asset=cube_1 --pll-id=00 --cycle-iteration=0
# python conversion_pll_to_bsdf.py --vision-asset=cube_2 --pll-id=00 --cycle-iteration=0
# python conversion_pll_to_bsdf.py --vision-asset=cube_3 --pll-id=00 --cycle-iteration=0
# python conversion_pll_to_bsdf.py --vision-asset=cube_4 --pll-id=00 --cycle-iteration=0

### iter 1
# python conversion_pll_to_bsdf.py --vision-asset=cube_1 --pll-id=01 --cycle-iteration=1
python conversion_pll_to_bsdf.py --vision-asset=cube_2 --pll-id=01 --cycle-iteration=1
# python conversion_pll_to_bsdf.py --vision-asset=cube_3 --pll-id=01 --cycle-iteration=1
# python conversion_pll_to_bsdf.py --vision-asset=cube_4 --pll-id=01 --cycle-iteration=1