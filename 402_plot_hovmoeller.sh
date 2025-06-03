#!/bin/bash

nproc=1

source 000_setup.sh
source 999_trapkill.sh

stat_dir=$gendata_dir/stat
output_dir=$fig_dir/hovmoeller

params=(
    "20230107_constSST" "20230107" "2023-01-07" "0,120,6" "PAT01_AMP-1.0,PAT00_AMP0.0,PAT01_AMP1.0,PAT01_AMP2.0" "IVT"
    "20230107_patternSST" "20230107" "2023-01-07" "0,120,6" "PAT00_AMP0.0,PAT00_AMP1.0,PAT00_AMP2.0" "IVT"
)

mkdir -p $output_dir

nparams=6
for (( i=0 ; i < $(( ${#params[@]} / $nparams )) ; i++ )); do


    comparison_name="${params[$(( i * $nparams + 0 ))]}"
    stat_input_dir="${params[$(( i * $nparams + 1 ))]}"
    exp_beg_time="${params[$(( i * $nparams + 2 ))]}"
    timeinfo="${params[$(( i * $nparams + 3 ))]}"
    casenames="${params[$(( i * $nparams + 4 ))]}"
    varnames="${params[$(( i * $nparams + 5 ))]}"
   
    echo ":: comparison_name   = $comparison_name" 
    echo ":: stat_input_dir    = $stat_input_dir"
    echo ":: exp_beg_time      = $exp_beg_time"
    echo ":: timeinfo          = $timeinfo"
    echo ":: casenames         = $casenames"
    echo ":: varnames          = $varnames"


    IFS=',' read -ra casenames <<< "$casenames" 
    IFS=',' read -ra timeinfo <<< "$timeinfo"

    input_root=$stat_dir/$stat_input_dir

    python3 ./src/plot_hovmoeller.py  \
        --input-root $input_root      \
        --casenames ${casenames[@]}   \
        --varnames $varnames          \
        --exp-beg-time $exp_beg_time  \
        --time-beg ${timeinfo[0]}    \
        --time-end ${timeinfo[1]}    \
        --time-stride ${timeinfo[2]} \
        --spatio-direction south_north \
        --latlon-rng $(( 360 - 125 )) $(( 360 - 120 )) \
        --plot-latlon-rng 25 50 \
        --output-dir $output_dir      \
        --output-prefix $comparison_name \
        --ref-index 1 \
        --nproc $nproc               

done


echo "Done."
