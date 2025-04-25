#!/bin/bash

source 00_setup.sh


input_root=$gendata_dir/stat
output_root=$fig_dir/stat_comparison

params=(
    "$input_root/20230107" "2023-01-12" WESTWRF-GEFS AMP0
)
    
mkdir -p $output_root

nparams=4
for (( i=0 ; i < $(( ${#params[@]} / $nparams )) ; i++ )); do
    
    input_dir="${params[$(( i * $nparams  + 0 ))]}"
    timestamp="${params[$(( i * $nparams  + 1 ))]}"
    input1="${params[$(( i * $nparams  + 2 ))]}"
    input2="${params[$(( i * $nparams  + 3 ))]}"
    

    
    echo ":: input_dir = $input_dir"
    echo ":: timestamp = $timestamp"
    echo ":: input1 = $input1"
    echo ":: input2 = $input2"
    

    input1_file=$input_dir/${input1}_${timestamp}.nc
    input2_file=$input_dir/${input2}_${timestamp}.nc
    
    output_file=$output_root/${input1}_${input2}_${timestamp}.png

    python3 ./src/plot_stat_comparison.py \
        --input-files $input1_file  $input2_file \
        --output $output_file                  \
        --lat-rng 30 50                        \
        --lon-rng $(( 360 - 130 )) $(( 360 - 110 )) 

done
