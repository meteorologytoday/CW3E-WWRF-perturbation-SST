#!/bin/bash

source 00_setup.sh
    


verification_dir=gendata/verification_0.08deg
output_dir=figures/diff

mkdir -p $output_dir

params=(
    Perturb1-1  PAT00_AMP-1.0 Baseline01 "BLANK" 
)

nparams=4

for (( i=0 ; i < $(( ${#params[@]} / $nparams )) ; i++ )); do

    expname1="${params[$(( i * $nparams + 0 ))]}"
    subgroup1="${params[$(( i * $nparams + 1 ))]}"

    expname2="${params[$(( i * $nparams + 2 ))]}"
    subgroup2="${params[$(( i * $nparams + 3 ))]}"




    echo ":: expname1 = $expname1"
    echo ":: subgroup1 = $subgroup1"
    echo ":: expname2 = $expname2"
    echo ":: subgroup2 = $subgroup2"
 
    subgroup1_str="_${subgroup1}"
    subgroup2_str="_${subgroup2}"

    for region in CA coastal sierra ; do

        input_WRF1=$verification_dir/verification_WRF_${expname1}${subgroup1_str}.nc
        input_WRF2=$verification_dir/verification_WRF_${expname2}${subgroup2_str}.nc
        
        output=$output_dir/verification_${region}_${expname1}${subgroup1_str}_VS_${expname2}${subgroup2_str}.png

        python3 ./src/plot_verification_diff.py    \
            --input-WRF1 $input_WRF1            \
            --input-WRF2 $input_WRF2            \
            --dataset1 ${expname1}${subgroup1_str} \
            --dataset2 ${expname2}${subgroup2_str} \
            --varnames ACC_total_precipitation \
            --output $output                  \
            --region $region                  \
            --no-display --no-legend
    done

done

