#!/bin/bash

source 00_setup.sh


input_root=$gendata_dir/stat
output_root=$fig_dir/skill_spread




output_file_prefix="spread_skill_20230107-20230112_PAT00"
output_filename=$output_root/${output_file_prefix}.png
obs_file=$gendata_dir/verification/PRISM_2023-01-07_2023-01-12.nc

fcst_files=(
    $input_root/20230107/WESTWRF-GEFS_2023-01-12.nc
    $input_root/20230107/PAT00_AMP2.0_2023-01-12.nc
    $input_root/20230107/PAT00_AMP1.0_2023-01-12.nc
    $input_root/20230107/AMP0_2023-01-12.nc
    $input_root/20230107/PAT00_AMP-1.0_2023-01-12.nc
)

if [ ] ; then
fcst_files=(
    $input_root/20230107/WESTWRF-GEFS_2023-01-12.nc
    $input_root/20230107/PAT01_AMP-1.0_2023-01-12.nc
    $input_root/20230107/AMP0_2023-01-12.nc
    $input_root/20230107/PAT01_AMP1.0_2023-01-12.nc
    $input_root/20230107/PAT01_AMP2.0_2023-01-12.nc
    $input_root/20230107/PAT01_AMP4.0_2023-01-12.nc
)
fi

mkdir -p $output_root
python3 ./src/plot_skill_spread.py \
    --input-obs-file  $obs_file \
    --input-fcst-files ${fcst_files[@]} \
    --output $output_filename                 \
    --title  $output_filename              \
    --lat-rng 30 45                        \
    --precip-max 50                        \
    --precip-interval 2.0                  \
    --precip-threshold 5.0                  \
    --no-display
