#!/bin/bash


label=AR2023JAN

verification_date_beg="2023-01-01T00:00:00"
verification_days=10
mask_file=gendata/mask.nc

output_dir=gendata/verification_0.08deg
output_ERA5=$output_dir/verification_ERA5.nc
output_PRISM=$output_dir/verification_PRISM.nc

# ===== WRF setting =====
#WRF_archived_root=/expanse/lustre/scratch/t2hsu/temp_project/PROCESSED_CW3E_WRF_RUNS/0.08deg
WRF_archived_root=/expanse/lustre/scratch/t2hsu/temp_project/CW3E_WRF_RUNS/0.08deg
WRF_params=(
#    Baseline01 "BLANK" 
    Perturb1-1  PAT00_AMP1.0
#    Perturb1-1  PAT00_AMP-1.0
)

exp_beg_time=$verification_date_beg
wrfout_data_interval=$(( 3600 * 24 ))
frames_per_wrfout_file=1
#ens_ids="0-16,18,21,22,24-30"
ens_ids="0-30"
#ens_ids="0-5"







