#!/bin/bash

source 00_setup.sh

output_dir=$gendata_dir/region_mask/

mkdir -p $output_dir


test_PRISM_file=/data/SO3/t2hsu/data/PRISM/PRISM_stable_4kmD2/PRISM_ppt_stable_4kmD2-2002-11-14.nc
test_regridded_file=SO3_gendata/regrid_data/exp_20230107/PAT00_AMP0.0/30/SST-2023-01-07T00:00:00.nc
python3 src/make_masks.py \
    --test-PRISM-file $test_PRISM_file \
    --test-regridded-file $test_regridded_file \
    --nproc 6 \
    --output $output_dir/mask.nc





#test_ERA5_file=/expanse/lustre/scratch/t2hsu/temp_project/ERA5/24hr/sea_surface_temperature/ERA5-sea_surface_temperature-2023-01-01_00.nc \
#test_WRF_file=~/temp_project/CW3E_WRF_RUNS/0.08deg/exp_Baseline01/runs/Baseline01_ens28/output/wrfout/wrfout_d01_2023-01-01_00\:00\:00_temp


if [ ] ; then
python3 src/make_masks.py \
    --test-PRISM-file $test_PRISM_file \
    --test-ERA5-file $test_ERA5_file \
    --test-WRF-file  $test_WRF_file  \
    --nproc 6 \
    --output gendata/mask.nc
fi


