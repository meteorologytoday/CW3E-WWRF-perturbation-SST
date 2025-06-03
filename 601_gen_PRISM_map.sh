#!/bin/bash

source 00_setup.sh

verification_date_beg="2023-01-07T00:00:00"
verification_date_end="2023-01-12T00:00:00"
output_root=$gendata_dir/verification

regrid_file=$gendata_dir/regrid_idx.nc

mkdir -p $output_root

python3 src/gen_PRISM_verification_data.py \
    --time-beg $verification_date_beg      \
    --time-end $verification_date_end      \
    --regrid-file $regrid_file \
    --output-root $output_root
