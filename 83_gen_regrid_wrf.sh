#!/bin/bash

nproc=20
 
source 60_verification_setup.sh
source 98_trapkill.sh

wrfout_data_interval=$(( 3600 * 6 ))

varnames="TTL_RAIN IWV PH::200 WND::200 PH::850 WND::850 PSFC SST T2 IVT"
#varnames="TTL_RAIN"

varnames="TTL_RAIN IWV IVT SST"

nparams=3
for (( i=0 ; i < $(( ${#WRF_params[@]} / $nparams )) ; i++ )); do

    expname="${WRF_params[$(( i * $nparams + 0 ))]}"
    group="${WRF_params[$(( i * $nparams + 1 ))]}"
    ens_ids="${WRF_params[$(( i * $nparams + 2 ))]}"
    
    echo ":: expname = $expname"
    echo ":: group   = $group"
    echo ":: ens_ids = $ens_ids"
    
    input_WRF_root=$WRF_archived_root
    output_root=$output_wrf_regrid_dir
    
    python3 src/gen_regrid_wrf.py                        \
        --nproc $nproc                                   \
        --regrid-file $regrid_file                       \
        --expname $expname                               \
        --group $group                                   \
        --input-WRF-root $input_WRF_root                 \
        --exp-beg-time $exp_beg_time                     \
        --wrfout-data-interval $wrfout_data_interval     \
        --frames-per-wrfout-file $frames_per_wrfout_file \
        --wrfout-prefix $wrfout_prefix                   \
        --wrfout-suffix $wrfout_suffix                   \
        --input-style $naming_style                      \
        --ens-ids $ens_ids                               \
        --output-time-range  0  240                      \
        --output-root $output_root \
        --varnames $varnames 

done


echo "Done."
