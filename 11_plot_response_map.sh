#!/bin/bash

source 00_setup.sh
    
output_fig_dir=$fig_dir/diff

nproc=1

source 98_trapkill.sh

archive_root=$data_dir/WRF_RUNS/0.16deg/runs

input_dirs=(
    $archive_root/PRELIMINARY_EOF1_AMP0.5/output/wrfout
)

input_dirs_base=(
    $archive_root/PRELIMINARY_EOF1_AMP-0.5/output/wrfout
)






exp_beg_time="2023-01-01T00:00:00"
wrfout_data_interval=10800
frames_per_wrfout_file=1


mkdir -p $output_dir



output_dir=$fig_dir/response_map
output=$output_dir/diff.png
python3 ./src/plot_response_map.py               \
    --input-dirs "${input_dirs[@]}"              \
    --input-dirs-base "${input_dirs_base[@]}"    \
    --exp-beg-time $exp_beg_time                 \
    --wrfout-data-interval $wrfout_data_interval \
    --frames-per-wrfout-file $frames_per_wrfout_file \
    --wrfout-suffix "_temp" \
    --time-rng $(( 24 * 4 )) $(( 24 * 5 )) \
    --output $output \
    --lat-rng 0 65 \
    --lon-rng 160 280 \
    --varnames SST PH500 \
    --no-display

if [ ] ; then
        proc_cnt=$(( $proc_cnt + 1))
        
        if (( $proc_cnt >= $nproc )) ; then
            echo "Max proc reached: $nproc"
            wait
            proc_cnt=0
        fi
fi

wait
echo "Done."
