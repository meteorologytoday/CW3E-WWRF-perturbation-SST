#!/bin/bash

source 00_setup.sh
    


verification_dir=gendata/verification

mkdir -p $output_dir

params=(
    Baseline01 "BLANK" 
    Perturb1-1  PAT00_AMP-1.0
)

nparams=2
for (( i=0 ; i < $(( ${#params[@]} / $nparams )) ; i++ )); do

    expname="${params[$(( i * $nparams + 0 ))]}"
    subgroup="${params[$(( i * $nparams + 1 ))]}"

    echo ":: expname = $expname"
    echo ":: subgroup = $subgroup"
 
    if [ "$subgroup" = "BLANK" ] ; then

        subgroup_str=""
    
    else
        subgroup_str="_${subgroup}"

    fi

    input_WRF=$verification_dir/verification_WRF_${expname}${subgroup_str}.nc
    input_ERA5=gendata/verification_ERA5.nc
    input_PRISM=gendata/verification_PRISM.nc
    output=figures/verification_${expname}${subgroup_str}.png

    python3 ./src/plot_verification.py    \
        --input-WRF $input_WRF            \
        --input-ERA5 $input_ERA5          \
        --input-PRISM $input_PRISM        \
        --varnames total_precipitation  convective_precipitation large_scale_precipitation  \
        --output $output                  \
        --no-display --no-legend


done

