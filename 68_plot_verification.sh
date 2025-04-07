#!/bin/bash

source 00_setup.sh

label=exp_20221224
 
verification_dir=gendata/$label/verification_0.08deg

mkdir -p $output_dir
    
params=(
    $label Perturb1    PAT00_AMP1.0
    $label Baseline01  BLANK
)

nparams=3

for (( i=0 ; i < $(( ${#params[@]} / $nparams )) ; i++ )); do
    expname="${params[$(( i * $nparams + 0 ))]}"
    group="${params[$(( i * $nparams + 1 ))]}"
    subgroup="${params[$(( i * $nparams + 2 ))]}"

    echo ":: expname  = $expname"
    echo ":: group    = $group"
    echo ":: subgroup = $subgroup"
 
    if [ "$subgroup" = "BLANK" ] ; then

        subgroup_str=""
    
    else
        subgroup_str="_${subgroup}"

    fi
        
    subgroup_str="_${subgroup}"
        
    input_WRF="$input_WRF $verification_dir/verification_WRF_${expname}_${group}${subgroup_str}.nc"
done

#input_ERA5=$verification_dir/verification_ERA5.nc
input_PRISM=$verification_dir/verification_PRISM.nc

mkdir -p figures/$label
for region in CA city_SF city_LA coastal sierra Dam_Oroville Dam_Shasta Dam_SevenOaks Dam_NewMelones ; do
#for region in CA ; do #city_SF city_LA coastal sierra Dam_Oroville Dam_Shasta Dam_SevenOaks Dam_NewMelones ; do

    output=figures/$label/verification_${region}.png

    python3 ./src/plot_verification.py    \
        --input-WRF $input_WRF            \
        --input-PRISM $input_PRISM        \
        --varnames ACC_total_precipitation \
        --output $output                  \
        --region $region                  \
        --no-display --no-legend
done


