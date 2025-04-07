#!/bin/bash


naming_style=v2
verification_days=10
WRF_archived_root=/expanse/lustre/scratch/t2hsu/temp_project/CW3E_WRF_RUNS/0.08deg


if [ "$1" == "" ]; then
    echo "Error: Need to provide an argument as the input configuration file."
    exit 1
fi
config_file="$1"
echo "Sourcing config file: $config_file"
source $config_file


if [ "$label" == "" ]; then
    echo "Error: Loading config failed. Variable `label` is still empty."
fi 

mask_file=gendata/mask.nc

output_dir=gendata/$label
output_PRISM=$output_dir/verification_PRISM.nc
output_CRPS_dir=$output_dir/CRPS
output_ens_stat_dir=$output_dir/ens_stat

exp_beg_time=$verification_date_beg
wrfout_data_interval=$(( 3600 * 24 ))
frames_per_wrfout_file=1


expname=$label
mkdir -p $output_dir
