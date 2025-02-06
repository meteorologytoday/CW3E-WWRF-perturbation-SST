#!/bin/bash

nproc=10
 
if [ "$1" == "" ]; then
    echo "Error: Need to provide an argument as the input configuration file."
    exit 1
fi
config_file="$1"
echo "Configuration file: $config_file"
source $config_file

   
mkdir -p $output_dir
nparams=2
for (( i=0 ; i < $(( ${#WRF_params[@]} / $nparams )) ; i++ )); do

    expname="${WRF_params[$(( i * $nparams + 0 ))]}"
    subgroup="${WRF_params[$(( i * $nparams + 1 ))]}"

    echo ":: expname = $expname"
    echo ":: subgroup = $subgroup"
 
    input_root=$WRF_archived_root/exp_${expname}/runs

    output_file=$output_dir/verification_WRF_${expname}_${subgroup}.nc

    python3 src/gen_WRF_verification_data.py             \
        --expname $expname                               \
        --subgroup $subgroup                             \
        --regions CA sierra coastal                      \
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

echo "Done."
