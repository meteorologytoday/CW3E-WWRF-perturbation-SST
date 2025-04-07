#!/bin/bash

nproc=1
 
source 60_verification_setup.sh
source 98_trapkill.sh

wrfout_data_interval=$(( 3600 * 6 ))

varnames="PSFC TTL_RAIN SSTSK T2 IVT"


nparams=3
for (( i=0 ; i < $(( ${#WRF_params[@]} / $nparams )) ; i++ )); do

    expname="${WRF_params[$(( i * $nparams + 0 ))]}"
    group="${WRF_params[$(( i * $nparams + 1 ))]}"
    subgroup="${WRF_params[$(( i * $nparams + 2 ))]}"

    echo ":: expname = $expname"
    echo ":: group   = $group"
    echo ":: subgroup = $subgroup"
 
    input_WRF_root=$WRF_archived_root
    output_root=$output_ens_stat_dir

    python3 src/gen_ensemble_analysis.py                 \
        --nproc $nproc                                   \
        --regrid-file $regrid_file                       \
        --expname $expname                               \
        --group $group                                   \
        --subgroup $subgroup                             \
        --input-WRF-root $input_WRF_root                 \
        --exp-beg-time $exp_beg_time                     \
        --wrfout-data-interval $wrfout_data_interval     \
        --frames-per-wrfout-file $frames_per_wrfout_file \
        --wrfout-suffix "_temp"                          \
        --ens-ids $ens_ids                               \
        --output-time-range  0  240                      \
        --output-root $output_root \
        --naming-style $naming_style \
        --varnames $varnames 

done


echo "Done."
