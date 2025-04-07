#!/bin/bash

if [ "$1" == "" ]; then
    echo "Error: Need to provide an argument as the input configuration file."
    exit 1
fi
config_file="$1"
echo "Configuration file: $config_file"
source $config_file

set -x
python3 src/gen_ERA5_verification_data.py \
    --date-beg $verification_date_beg \
    --test-days $verification_days    \
    --region-file $mask_file          \
    --output  $output_ERA5




