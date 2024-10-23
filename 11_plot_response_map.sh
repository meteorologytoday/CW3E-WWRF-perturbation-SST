#!/bin/bash

source 00_setup.sh
source 98_trapkill.sh
    
output_fig_dir=$fig_dir/diff

nproc=2

archive_root=$data_dir/WRF_RUNS/0.16deg/runs

input_dirs=(
    $archive_root/PRELIMINARY_EOF1_AMP-0.5/output/wrfout
    $archive_root/PRELIMINARY_EOF1_AMP-0.3/output/wrfout
    $archive_root/PRELIMINARY_EOF1_AMP0.3/output/wrfout
    $archive_root/PRELIMINARY_EOF1_AMP0.5/output/wrfout
)

input_dirs_base=(
    $archive_root/PRELIMINARY_EOF1_AMP0.0/output/wrfout
)






exp_beg_time="2023-01-01T00:00:00"
wrfout_data_interval=10800
frames_per_wrfout_file=1


mkdir -p $output_dir



for day_beg in $( seq 0 16 ) ; do

    day_end=$(( $day_beg + 1 ))
    output_dir=$fig_dir/response_map
    output=$output_dir/$( printf "diff_day%02d-%02d.png" $day_beg $day_end )

    if [ -f "$output" ]; then

        echo "File already exists: $output"
        echo "Skip."

    else

        echo "Produce output: $output"
        python3 ./src/plot_response_map.py               \
            --input-dirs "${input_dirs[@]}"              \
            --input-dirs-base "${input_dirs_base[@]}"    \
            --exp-beg-time $exp_beg_time                 \
            --wrfout-data-interval $wrfout_data_interval \
            --frames-per-wrfout-file $frames_per_wrfout_file \
            --wrfout-suffix "_temp" \
            --time-rng $(( 24 * $day_beg )) $(( 24 * $day_end )) \
            --output $output \
            --lat-rng 0 65 \
            --lon-rng 160 280 \
            --varnames SST_NOLND PH500 PH850 TTL_RAIN \
            --no-display & 


        proc_cnt=$(( $proc_cnt + 1))
        
        if (( $proc_cnt >= $nproc )) ; then
            echo "Max proc reached: $nproc"
            wait
            proc_cnt=0
        fi

    fi
done

wait
echo "Done."
