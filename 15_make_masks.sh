#!/bin/bash

source 000_setup.sh

output_dir=$gendata_dir/region_mask/

mkdir -p $output_dir
test_PRISM_file=/data/SO3/t2hsu/data/PRISM/PRISM_stable_4kmD2/PRISM_ppt_stable_4kmD2-2002-11-14.nc
test_regridded_file=SO3_gendata/regrid_data/dx0.5/exp_20230107/PAT00_AMP0.0/30/SST-2023-01-07T00:00:00.nc
python3 src/make_masks.py \
    --test-PRISM-file $test_PRISM_file \
    --test-regridded-file $test_regridded_file \
    --nproc 6 \
    --output $output_dir/mask.nc




