#!/bin/bash

source 00_setup.sh


source 60_verification_setup.sh    

params=(
    Baseline01  BLANK    Perturb1  PAT00_AMP1.0
    Perturb1  PAT00_AMP-1.0    Perturb1  PAT00_AMP1.0
)
nparams=4
for (( i=0 ; i < $(( ${#params[@]} / $nparams )) ; i++ )); do
    
    expname1="${params[$(( i * $nparams  + 0 ))]}"
    subgroup1="${params[$(( i * $nparams + 1 ))]}"
    
    expname2="${params[$(( i * $nparams  + 2 ))]}"
    subgroup2="${params[$(( i * $nparams + 3 ))]}"
    
    echo ":: expname1  = $expname1"
    echo ":: subgroup1 = $subgroup1"
    
    echo ":: expname2  = $expname2"
    echo ":: subgroup2 = $subgroup2"
    
   
    subgroup1_str=$( subgroupLabel "$subgroup1") 
    subgroup2_str=$( subgroupLabel "$subgroup2") 
    
    data1_str=${expname1}${subgroup1_str}
    data2_str=${expname2}${subgroup2_str}

 
    input_files="$output_CRPS_dir/CRPS_${label}_${expname1}${subgroup1_str}.nc  $output_CRPS_dir/CRPS_${label}_${expname2}${subgroup2_str}.nc"

    day_beg=0
    day_end=9


    output_dir=$fig_dir/$label/${data1_str}_vs_${data2_str}
    mkdir -p $output_dir

    output=$output_dir/CRPS_comparison_day_${day_beg}-${day_end}.png

    python3 ./src/plot_CRPS_comparison.py    \
        --input-files $input_files           \
        --output $output                     \
        --time-idx-rng ${day_beg} ${day_end} \
        --precip-max 400                     \
        --precip-threshold 5                 \
        --diff-precip-max  10                \
        --CRPS-ratio-max 0.1                 \
        --lat-rng 30 60                      \
        --lon-rng $(( 360 - 145 )) $(( 360 - 105 ))           \
        --no-display 

    for time_idx in 0 1 2 3 4 5 6 7 8 9 ; do
        output=$output_dir/CRPS_comparison_day${time_idx}.png

        python3 ./src/plot_CRPS_comparison.py \
            --input-files $input_files        \
            --output $output                  \
            --time-idx-rng $time_idx  $time_idx  \
            --lat-rng 30 60                   \
            --lon-rng $(( 360 - 145 )) $(( 360 - 105 ))           \
            --CRPS-ratio-max 0.2                \
            --no-display 
    done
done
