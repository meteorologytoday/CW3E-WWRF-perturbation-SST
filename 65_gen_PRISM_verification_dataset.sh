#!/bin/bash

python3 src/gen_PRISM_verification_data.py \
    --date-beg 2023-01-01 \
    --test-days 10        \
    --region-file gendata/mask_SWUS.nc \
    --output gendata/verification_PRISM.nc
