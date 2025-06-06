#!/bin/bash

source 60_verification_setup.sh 


nproc=5


nparams=3
for (( i=0 ; i < $(( ${#WRF_params[@]} / $nparams )) ; i++ )); do

    expname="${WRF_params[$(( i * $nparams + 0 ))]}"
    group="${WRF_params[$(( i * $nparams + 1 ))]}"
    subgroup="${WRF_params[$(( i * $nparams + 2 ))]}"

    echo ":: expname  = $expname"
    echo ":: group    = $group"
    echo ":: subgroup = $subgroup"
 
    input_root=$WRF_archived_root/${expname}
    output_file=$output_dir/verification_WRF_${expname}_${group}_${subgroup}.nc

    python3 src/gen_WRF_verification_data.py             \
        --expname $expname                               \
        --group   $group                                 \
        --subgroup $subgroup                             \
        --input-root $input_root                         \
        --exp-beg-time $exp_beg_time                     \
        --wrfout-data-interval $wrfout_data_interval     \
        --frames-per-wrfout-file $frames_per_wrfout_file \
        --wrfout-suffix "_temp"                          \
        --ens-ids $ens_ids                               \
        --time-beg 0                                     \
        --lead-days $verification_days                   \
        --mask $mask_file                                \
        --nproc $nproc                                   \
        --output $output_file 

done

wait

echo "Done."
