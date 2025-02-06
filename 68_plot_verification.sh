#!/bin/bash

source 00_setup.sh
    


verification_dir=gendata/verification_0.08deg

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
        
    subgroup_str="_${subgroup}"

    for region in CA coastal sierra ; do

        input_WRF=$verification_dir/verification_WRF_${expname}${subgroup_str}.nc
        input_ERA5=$verification_dir/verification_ERA5.nc
        input_PRISM=$verification_dir/verification_PRISM.nc
        output=figures/verification_${region}_${expname}${subgroup_str}.png

#            --varnames total_precipitation  convective_precipitation large_scale_precipitation  \
        python3 ./src/plot_verification.py    \
            --input-WRF $input_WRF            \
            --input-ERA5 $input_ERA5          \
            --input-PRISM $input_PRISM        \
            --varnames ACC_total_precipitation \
            --output $output                  \
            --region $region                  \
            --no-display --no-legend
    done

done

