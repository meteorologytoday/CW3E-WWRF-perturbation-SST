#!/bin/bash

mask_file=gendata/mask_PRISM.nc
half_window_size=0

python3 src/gen_PRISM_historical_data.py \
    --year-rng 1982 2024 \
    --valid-months 1 2 3 9 10 11 12 \
    --half-window-size $half_window_size \
    --region-file $mask_file \
    --output-dir gendata/PRISM_historical/half_window_size-${half_window_size}
