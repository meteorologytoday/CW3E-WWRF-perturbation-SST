#!/bin/bash

source 60_verification_setup.sh

mkdir -p $output_dir

python3 src/gen_PRISM_verification_data.py \
    --date-beg $verification_date_beg \
    --test-days $verification_days        \
    --region-file $mask_file \
    --output $output_PRISM
