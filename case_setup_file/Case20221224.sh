#!/bin/bash


#WRF_archived_root=/expanse/lustre/scratch/t2hsu/temp_project/CW3E_WRF_RUNS/0.08deg
WRF_archived_root=/expanse/lustre/scratch/t2hsu/temp_project/PROCESSED_CW3E_WRF_RUNS/0.08deg
label=exp_20221224

verification_date_beg="2022-12-24T00:00:00"
verification_days=10
mask_file=gendata/mask.nc

regrid_file=gendata/regrid_idx.nc


WRF_params=(
    $label  "Perturb1"   "PAT00_AMP-1.0"
    $label  "Perturb1"   "PAT00_AMP1.0"
    $label  "Baseline01" "BLANK"
)

#ens_ids="0-30"
ens_ids="1-30"

