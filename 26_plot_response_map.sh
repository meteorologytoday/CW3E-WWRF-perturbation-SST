#!/bin/bash

source 00_setup.sh
source 98_trapkill.sh
    
output_fig_dir=$fig_dir/diff

nproc=2

for batchname in method2; do

    archive_root=$data_dir/WRF_RUNS/0.16deg/runs_${batchname}

    for epsilon in -50 50 100 150; do
    #for epsilon in 50; do
        epsilon=$(  python3 -c "print('%.2f' % ($epsilon/100) )" )
        input_dirs="$input_dirs $archive_root/PRELIMINARY_eta1.00_epsilon${epsilon}/output/wrfout"
        epsilons="$epsilons $epsilon"
    done

    input_dirs_base=(
        $archive_root/PRELIMINARY_eta1.00_epsilon0.00/output/wrfout
    )

    exp_beg_time="2023-01-01T00:00:00"
    wrfout_data_interval=10800
    frames_per_wrfout_file=1

    days_to_avg=1

    for day_beg in $( seq 15 -1 0 ) ; do

        day_end=$(( $day_beg + $days_to_avg ))
        output_dir=$fig_dir/response_map/$batchname
        output=$output_dir/$( printf "diff_day%02d-%02d.png" $day_beg $day_end )


        if [ -f "$output" ]; then

            echo "File already exists: $output"
            echo "Skip."

        else

            mkdir -p $output_dir

            echo "Produce output: $output"
            python3 ./src/plot_response_map.py               \
                --input-dirs $input_dirs                     \
                --input-dirs-base "${input_dirs_base[@]}"    \
                --exp-beg-time $exp_beg_time                 \
                --wrfout-data-interval $wrfout_data_interval \
                --frames-per-wrfout-file $frames_per_wrfout_file \
                --wrfout-suffix "_temp" \
                --time-rng $(( 24 * $day_beg )) $(( 24 * $day_end )) \
                --output $output \
                --lat-rng 0 65 \
                --lon-rng 160 275 \
                --varnames SST_NOLND PH500 PH850 TTL_RAIN.FILTER-LP \
                --no-display & 



            proc_cnt=$(( $proc_cnt + 1))
            
            if (( $proc_cnt >= $nproc )) ; then
                echo "Max proc reached: $nproc"
                wait
                proc_cnt=0
            fi

        fi
    done
done

wait
echo "Done."
