#!/bin/bash


label=exp_20230107

verification_date_beg="2023-01-07T00:00:00"
verification_days=10
mask_file=gendata/mask.nc

output_dir=gendata/$label/verification_0.08deg
output_ERA5=$output_dir/verification_ERA5.nc
output_PRISM=$output_dir/verification_PRISM.nc

output_CRPS_dir=$output_dir/CRPS

regrid_file=$gendata_dir/regrid_idx.nc


# ===== WRF setting =====
WRF_archived_root=./data/PROCESSED_CW3E_WRF_RUNS/0.08deg
#WRF_archived_root=/expanse/lustre/scratch/t2hsu/temp_project/CW3E_WRF_RUNS/0.08deg
WRF_params=(
#    $label  "Baseline01" "BLANK"
    $label  "PAT00_AMP-1.0"  "0-30"
    $label  "PAT00_AMP1.0"   "0-30"
)

exp_beg_time=$verification_date_beg
wrfout_data_interval=$(( 3600 * 24 ))
frames_per_wrfout_file=1

