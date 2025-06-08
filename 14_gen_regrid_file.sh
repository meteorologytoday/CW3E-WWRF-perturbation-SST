#!/bin/bash

source 000_setup.sh


WRF_file=/data/SO3/t2hsu/data/PROCESSED_CW3E_WRF_RUNS/0.08deg/exp_20230107/runs/PAT00_AMP0.0/00/output/wrfout/wrfout_d01_2023-01-07_00:00:00_temp
PRISM_file=/data/SO3/t2hsu/data/PRISM/PRISM_stable_4kmD1/PRISM-ppt-1981-01-01.nc


for dx in 0.5 ; do

    output_dir=$gendata_dir/dx${dx}

    mkdir -p $output_dir

    output_WRF2mygrid_file=$gendata_dir/regrid_idx_dx${dx}_WRF2mygrid.nc
    python3 src/gen_regrid_file.py \
        --input-file $WRF_file \
        --input-type "WRF"     \
        --lat-rng    0    70   \
        --lon-rng    170  250  \
        --dlat $dx             \
        --dlon $dx             \
        --output $output_WRF2mygrid_file

    python3 src/gen_regrid_file.py \
        --input-file $PRISM_file   \
        --input-type "PRISM"       \
        --lat-rng    0    70       \
        --lon-rng    170  250      \
        --dlat $dx                 \
        --dlon $dx                 \
        --output $output_dir/regrid_idx_dx${dx}_PRISM2mygrid.nc
    
    python3 src/gen_regrid_file.py \
        --input-file $output_WRF2mygrid_file   \
        --input-type "regrid"       \
        --lat-rng    0    70       \
        --lon-rng    170  250      \
        --dlat $dx                 \
        --dlon $dx                 \
        --output $output_dir/regrid_idx_dx${dx}_mygrid2sengrid_ocn.nc
 
    python3 src/gen_regrid_file.py \
        --input-file $output_WRF2mygrid_file   \
        --input-type "regrid"       \
        --lat-rng    0    70       \
        --lon-rng    170  250      \
        --dlat $dx                 \
        --dlon $dx                 \
        --output $output_dir/regrid_idx_dx${dx}_mygrid2sengrid_target.nc
  
done    
