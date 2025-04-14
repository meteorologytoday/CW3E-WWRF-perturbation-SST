#!/bin/bash

nproc=20
 
source 60_verification_setup.sh
source 98_trapkill.sh

wrfout_data_interval=$(( 3600 * 12 ))

varnames="PH::200 WND::200 WND::850 PSFC TTL_RAIN SST T2 IVT SSTSK"


nparams=2
for (( i=0 ; i < $(( ${#WRF_params[@]} / $nparams )) ; i++ )); do

    expname="${WRF_params[$(( i * $nparams + 0 ))]}"
    group="${WRF_params[$(( i * $nparams + 1 ))]}"
    
    echo ":: expname = $expname"
    echo ":: group   = $group"
    
    input_WRF_root=$WRF_archived_root
    output_root=$output_ens_stat_dir
    
    python3 src/gen_ensemble_analysis.py                 \
        --nproc $nproc                                   \
        --regrid-file $regrid_file                       \
        --expname $expname                               \
        --group $group                                   \
        --input-WRF-root $input_WRF_root                 \
        --exp-beg-time $exp_beg_time                     \
        --wrfout-data-interval $wrfout_data_interval     \
        --frames-per-wrfout-file $frames_per_wrfout_file \
        --wrfout-suffix "_temp"                          \
        --ens-ids $ens_ids                               \
        --output-time-range  0  240                      \
        --output-root $output_root \
        --varnames $varnames 

done


echo "Done."
