#!/bin/bash

source 00_setup.sh

nproc=1

params=(
    "2023-01-07T00:00:00" "2023010700::gefs::0-79" "20230107::PAT00_AMP0.0::0-30|exp_20230107::PAT00_AMP-1.0::0-30"
)

regrid_file=$gendata_dir/regrid_idx.nc

nparams=3
for (( i=0 ; i < $(( ${#params[@]} / $nparams )) ; i++ )); do
    
    exp_beg_time="${params[$(( i * $nparams  + 0 ))]}"
    
    expset1="${params[$(( i * $nparams  + 1 ))]}"
    expset2="${params[$(( i * $nparams  + 2 ))]}"

    echo ":: ${exp_beg_time}"
    echo ":: ${expset1}"
    echo ":: ${expset2}"
    
    input_root=$gendata_dir/regrid_data
    output_root=$fig_dir/rainfall_verification
    output=$output_root/test.nc
    
    mkdir -p $output_root
    
    time_beg=$(( 24 * 0 ))
    time_end=$(( 24 * 6 + 6 ))
    time_stride=6
    python3 ./src/plot_precip_verification.py     \
        --input-root $input_root                  \
        --expsets  $expset1 $expset2                \
        --output $output                          \
        --regrid-file $regrid_file                \
        --exp-beg-time $exp_beg_time            \
        --time-beg $time_beg \
        --time-end $time_end \
        --time-stride $time_stride \
        --clim-year-range 1990 2015                        \
        --nproc $nproc 

done
