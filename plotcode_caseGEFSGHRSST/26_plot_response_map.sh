#!/bin/bash

source 00_setup.sh
source 98_trapkill.sh


output_fig_dir=$fig_dir/diff

nproc=4
hours_to_avg=6

for pat in 0 ; do

    archive_root=$data_dir/WRF_RUNS/0.16deg/runs_${batchname}

    #for amp_int in $( seq -10 10 ) ; do
    for amp_int in -10 -5 5 10 ; do
    #for amp_int in 10 ; do
        amp=$(  python3 -c "print('%.1f' % ( $amp_int / 10), )" )
        input_dirs="$input_dirs $archive_root/PRELIMINARY_PAT${pat}_AMP${amp}/output/wrfout"
        amps="$amps $amp"
    done

    input_dirs_base=(
        $archive_root/PRELIMINARY_PAT0_AMP0.0/output/wrfout
    )

    exp_beg_time="2023-01-01T00:00:00"
    wrfout_data_interval=10800
    frames_per_wrfout_file=1
    
    for hour_beg in $( seq $(( 15 * 24 )) -6 0 ) ; do

        hour_end=$(( $hour_beg + $hours_to_avg ))
        output_dir=$fig_dir/response_map/$batchname
        output=$output_dir/$( printf "diff_hours%03d-%03d.png" $hour_beg $hour_end )


        if [ -f "$output" ]; then

            echo "File already exists: $output"
            echo "Skip."

        else

            mkdir -p $output_dir

            echo "Produce output: $output"
            python3 $src_dir/plot_response_map.py               \
                --input-dirs $input_dirs                     \
                --input-dirs-base "${input_dirs_base[@]}"    \
                --exp-beg-time $exp_beg_time                 \
                --wrfout-data-interval $wrfout_data_interval \
                --frames-per-wrfout-file $frames_per_wrfout_file \
                --wrfout-suffix "_temp" \
                --time-rng $hour_beg $hour_end \
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
