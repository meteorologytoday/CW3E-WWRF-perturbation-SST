#!/bin/bash

python3 src/make_alaska_gulf_masks.py \
    --test-ERA5-file /expanse/lustre/scratch/t2hsu/temp_project/ERA5/24hr/sea_surface_temperature/ERA5-sea_surface_temperature-2023-01-01_00.nc \
    --output gendata/mask_SWUS.nc
