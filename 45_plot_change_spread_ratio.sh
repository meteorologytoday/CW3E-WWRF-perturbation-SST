#!/bin/bash

source 00_setup.sh


input_root=$gendata_dir/stat
output_root=$fig_dir/change_spread_ratio

params=(
    "20230107_WESTWRF-ALL" "$input_root/20230107" "2023-01-12" AMP1 AMP-1 WESTWRF-ALL
)
    
mkdir -p $output_root

nparams=6
for (( i=0 ; i < $(( ${#params[@]} / $nparams )) ; i++ )); do
    
    output_filename="${params[$(( i * $nparams  + 0 ))]}"
    input_dir="${params[$(( i * $nparams  + 1 ))]}"
    timestamp="${params[$(( i * $nparams  + 2 ))]}"
    forcing1_input="${params[$(( i * $nparams  + 3 ))]}"
    forcing2_input="${params[$(( i * $nparams  + 4 ))]}"
    target_input="${params[$(( i * $nparams  + 5 ))]}"
    
    echo ":: input_dir = $input_dir"
    echo ":: forcing1_input = $forcing1_input"
    echo ":: forcing2_input = $forcing2_input"
    echo ":: target_input = $target_input"

    forcing1_input_file=$input_dir/${forcing1_input}_${timestamp}.nc
    forcing2_input_file=$input_dir/${forcing2_input}_${timestamp}.nc
    target_input_file=$input_dir/${target_input}_${timestamp}.nc
    
    output_file=$output_root/${output_filename}.png

    python3 ./src/plot_change_spread_ratio.py \
        --input-forcing-files $forcing1_input_file  $forcing2_input_file \
        --input-target-file $target_input_file \
        --output $output_file                  \
        --title  $output_filename              \
        --lat-rng 30 50                        \
        --lon-rng $(( 360 - 130 )) $(( 360 - 110 )) 

#        --lat-rng 10 70                        \
#        --lon-rng $(( 360 - 179 )) $(( 360 - 105 )) 


done
