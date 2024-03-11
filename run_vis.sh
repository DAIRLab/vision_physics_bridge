#!/bin/sh

### Generate videos to visualize the tracking. 

# source /opt/ros/noetic/setup.bash
# source ../../robot_filter/venv/bin/activate

python overlay_cube_old.py 1 1
python overlay_cube_old.py 1 2
python overlay_cube_old.py 2 1
python overlay_cube_old.py 2 2
python overlay_cube_old.py 3 1 # did not lose track
python overlay_cube_old.py 3 2 # did not lose track
python overlay_cube_old.py 4 1
python overlay_cube_old.py 4 2
python overlay_cube_old.py 5 1 # the frames mismatch
python overlay_cube_old.py 5 2 # the frames mismatch
python overlay_cube_old.py 6 1
python overlay_cube_old.py 6 2
python overlay_cube_old.py 7 1
python overlay_cube_old.py 7 2
python overlay_cube_old.py 8 1 # no tracking
python overlay_cube_old.py 8 2 # no tracking
python overlay_cube_old.py 9 1
python overlay_cube_old.py 9 2
python overlay_cube_old.py 10 1
python overlay_cube_old.py 10 2