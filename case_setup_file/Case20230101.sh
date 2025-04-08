#!/bin/bash


label=exp_20230101

verification_date_beg="2023-01-01T00:00:00"
verification_days=10
mask_file=gendata/mask.nc

output_dir=gendata/$label/verification_0.08deg
output_ERA5=$output_dir/verification_ERA5.nc
output_PRISM=$output_dir/verification_PRISM.nc

output_CRPS_dir=$output_dir/CRPS

regrid_file=gendata/regrid_idx.nc


# ===== WRF setting =====
WRF_archived_root=./data/PROCESSED_CW3E_WRF_RUNS/0.08deg
#WRF_archived_root=/expanse/lustre/scratch/t2hsu/temp_project/CW3E_WRF_RUNS/0.08deg
WRF_params=(
#    $label  "Baseline01" "BLANK"
    $label  "Perturb1-1"   "PAT00_AMP1.0"
    $label  "Perturb1-1"   "PAT00_AMP-1.0"
)

exp_beg_time=$verification_date_beg
wrfout_data_interval=$(( 3600 * 24 ))
frames_per_wrfout_file=1
#ens_ids="0-16,18,21,22,24-30"
ens_ids="0-30"
#ens_ids="2-6,8,15-20,22-30"
#ens_ids="2-6,22-30"
#ens_ids="2-6"


naming_style=v1




