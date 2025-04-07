#!/bin/bash

test_PRISM_file=/expanse/lustre/scratch/t2hsu/temp_project/data/PRISM-30yr/PRISM-ppt-01-01.nc

python3 src/make_masks.py \
    --test-PRISM-file $test_PRISM_file \
    --nproc 6 \
    --output gendata/mask_PRISM.nc
