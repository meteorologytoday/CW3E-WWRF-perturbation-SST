#!/bin/bash

source 00_setup.sh
    


verification_dir=gendata/verification_0.08deg

mkdir -p $output_dir

params=(
    Perturb1-1  PAT00_AMP1.0
    Baseline01  BLANK
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
        
    input_WRF="$input_WRF $verification_dir/verification_WRF_${expname}${subgroup_str}.nc"
done

input_ERA5=$verification_dir/verification_ERA5.nc
input_PRISM=$verification_dir/verification_PRISM.nc

for region in city_SF city_LA CA coastal sierra Dam_Oroville Dam_Shasta Dam_SevenOaks Dam_NewMelones ; do

    output=figures/verification_${region}.png

    python3 ./src/plot_verification.py    \
        --input-WRF $input_WRF            \
        --input-PRISM $input_PRISM        \
        --varnames ACC_total_precipitation \
        --output $output                  \
        --region $region                  \
        --no-display --no-legend
done


