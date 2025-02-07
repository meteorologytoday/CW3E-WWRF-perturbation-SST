#!/bin/bash

test_ERA5_file=/expanse/lustre/scratch/t2hsu/temp_project/ERA5/24hr/sea_surface_temperature/ERA5-sea_surface_temperature-2023-01-01_00.nc \
test_WRF_file=~/temp_project/CW3E_WRF_RUNS/0.08deg/exp_Baseline01/runs/Baseline01_ens28/output/wrfout/wrfout_d01_2023-01-01_00\:00\:00_temp
test_PRISM_file=/expanse/lustre/scratch/t2hsu/temp_project/data/PRISM-30yr/PRISM-ppt-01-01.nc


python3 src/make_masks.py \
    --test-PRISM-file $test_PRISM_file \
    --test-ERA5-file $test_ERA5_file \
    --test-WRF-file  $test_WRF_file  \
    --nproc 6 \
    --output gendata/mask.nc
