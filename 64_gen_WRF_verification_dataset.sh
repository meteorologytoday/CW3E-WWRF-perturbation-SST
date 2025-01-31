#!/bin/bash

source 98_trapkill.sh
    
nproc=1
    
archived_root=/expanse/lustre/scratch/t2hsu/temp_project/CW3E_WRF_RUNS/0.16deg
output_dir=gendata/verification

mkdir -p $output_dir

params=(
    Baseline01 "BLANK" 
    Perturb1-1  PAT00_AMP-1.0
)

exp_beg_time="2023-01-01T00:00:00"
wrfout_data_interval=$(( 3600 * 24 ))
frames_per_wrfout_file=1
ens_size=31

nparams=2
for (( i=0 ; i < $(( ${#params[@]} / $nparams )) ; i++ )); do

    expname="${params[$(( i * $nparams + 0 ))]}"
    subgroup="${params[$(( i * $nparams + 1 ))]}"

    echo ":: expname = $expname"
    echo ":: subgroup = $subgroup"
 
    input_root=$archived_root/exp_${expname}/runs

    output_file=$output_dir/verification_WRF_${expname}_${subgroup}.nc

    python3 src/gen_WRF_verification_data.py             \
        --expname $expname                               \
        --subgroup $subgroup                             \
        --region SWUS                                    \
        --input-root $input_root                         \
        --exp-beg-time $exp_beg_time                     \
        --wrfout-data-interval $wrfout_data_interval     \
        --frames-per-wrfout-file $frames_per_wrfout_file \
        --wrfout-suffix "_temp"                          \
        --ens-size 31                                    \
        --time-beg 0                                     \
        --lead-days 10                                   \
        --mask gendata/mask_SWUS.nc                      \
        --output $output_file

    proc_cnt=$(( $proc_cnt + 1))

    if (( $proc_cnt >= $nproc )) ; then
        echo "Max proc reached: $nproc"
        wait
        proc_cnt=0
    fi

done

wait

echo "Done."
