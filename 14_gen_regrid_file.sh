#!/bin/bash


python3 src/gen_regrid_file.py \
    --WRF-file ~/temp_project/PROCESSED_CW3E_WRF_RUNS/0.08deg/exp_Baseline01/runs/Baseline01_ens00/output/wrfout/wrfout_d01_2023-01-01_00\:00\:00_temp \
    --PRISM-file ~/temp_project/data/PRISM_stable_4kmD1/PRISM-ppt-1988-01-17.nc \
    --lat-rng    0    70  \
    --lon-rng    170  250  \
    --dlat 0.5 \
    --dlon 0.5 \
    --output gendata/regrid_idx.nc
    
