#!/bin/bash

ani_dir=animations

mkdir -p $ani_dir

# Make an animation of response map
convert -scale 30% -delay 500 -loop 0 figures/response_map/GEFS_MUR/diff_hours*.png $ani_dir/diff_GEFS_OSTIA.mp4



# Make an animation of correlation map
#convert -delay 500 -loop 0 figures/corr_map/GEFS_MUR/corr_map_hours*.png $ani_dir/corr_map.mp4
