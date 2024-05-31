#!/bin/bash

# Do 1-5 tosses for untagged objects to get as many as possible.
# python adjust_start_frames.py --vision-asset=croc_1-5
# python adjust_start_frames.py --vision-asset=crushedcan_1-5
# python adjust_start_frames.py --vision-asset=gallon_1-5
# python adjust_start_frames.py --vision-asset=greencan_1-5
# python adjust_start_frames.py --vision-asset=oatly_1-5
# python adjust_start_frames.py --vision-asset=pinkcan_1-5
# python adjust_start_frames.py --vision-asset=stapler_1-5
python adjust_start_frames.py --vision-asset=styrofoam_1-5

# Can run on fewer tosses for TagSLAM, since these will do 1-10 anyway.
# Note:  These needed BundleSDF ID 00 instead of 02.
# python adjust_start_frames.py --vision-asset=cube_1-5
# python adjust_start_frames.py --vision-asset=napkin_1-5
# python adjust_start_frames.py --vision-asset=half_1-5
# python adjust_start_frames.py --vision-asset=milk_1-5
# python adjust_start_frames.py --vision-asset=egg_1-5

# Put least likely objects to be useful last.
# python adjust_start_frames.py --vision-asset=icetray_1-5
# python adjust_start_frames.py --vision-asset=mug_1-5
# python adjust_start_frames.py --vision-asset=prism_1-3
# python adjust_start_frames.py --vision-asset=toblerone_1-3
