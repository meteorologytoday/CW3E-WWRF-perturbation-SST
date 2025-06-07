#!/bin/bash

source 000_setup.sh


WRF_file=/data/SO3/t2hsu/data/PROCESSED_CW3E_WRF_RUNS/0.08deg/exp_20230107/runs/PAT00_AMP0.0/00/output/wrfout/wrfout_d01_2023-01-07_00:00:00_temp
PRISM_file=/data/SO3/t2hsu/data/PRISM/PRISM_stable_4kmD1/PRISM-ppt-1981-01-01.nc

for dx in 2.0 0.5 ; do

    python3 src/gen_regrid_file.py \
        --WRF-file $WRF_file \
        --PRISM-file $PRISM_file \
        --lat-rng    0    70  \
        --lon-rng    170  250  \
        --dlat $dx \
        --dlon $dx \
        --output $gendata_dir/regrid_idx_dx${dx}.nc
    
done    
