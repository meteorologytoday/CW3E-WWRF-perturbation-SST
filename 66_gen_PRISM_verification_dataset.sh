#!/bin/bash

source 00_setup.sh

#source 60_verification_setup.sh

verification_date_beg="2023-01-07T00:00:00"
verification_days=10
mask_file=$gendata_dir/mask
output_dir=$gendata_dir/verification
output=$output_dir/verification_${verification_date_beg}.nc

mkdir -p $output_dir

python3 src/gen_PRISM_verification_data.py \
    --date-beg $verification_date_beg      \
    --test-days $verification_days         \
    --region-file $mask_file               \
    --output $output
