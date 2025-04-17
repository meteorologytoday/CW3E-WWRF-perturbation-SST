#!/bin/bash


label=2023010700

verification_date_beg="2023-01-07T00:00:00"
verification_days=10
mask_file=gendata/mask.nc

regrid_file=$gendata_dir/regrid_idx.nc

# ===== WRF setting =====
WRF_archived_root=./data/WestWRF200member
#WRF_archived_root=/expanse/lustre/scratch/t2hsu/temp_project/CW3E_WRF_RUNS/0.08deg
WRF_params=(
    $label  "gefs" "0-79"
)
naming_style="CW3E-WestWRF"
wrfout_prefix="wrfcf_d01_"
wrfout_suffix=".nc"
exp_beg_time=$verification_date_beg
wrfout_data_interval=$(( 3600 * 3 ))
frames_per_wrfout_file=1
#ens_ids="0-16,18,21,22,24-30"

#ens_ids="2-6,8,15-20,22-30"
#ens_ids="2-6,22-30"
#ens_ids="2-6"

