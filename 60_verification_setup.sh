#!/bin/bash

source 00_setup.sh

naming_style=default
wrfout_prefix=wrfout_d01_
wrfout_suffix=_temp
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

mask_file=$gendata_dir/mask.nc

output_dir=$gendata_dir/$label
output_PRISM=$output_dir/verification_PRISM.nc
output_CRPS_dir=$output_dir/CRPS
output_ens_stat_dir=$output_dir/ens_stat

wrf_regrid_dir=$gendata_dir/regrid_data

exp_beg_time=$verification_date_beg
wrfout_data_interval=$(( 3600 * 24 ))
frames_per_wrfout_file=1


expname=$label
mkdir -p $output_dir
