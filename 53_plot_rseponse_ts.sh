#!/bin/bash

source 00_setup.sh
source 98_trapkill.sh
    
nproc=1
exp_beg_time="2023-01-01T00:00:00"
wrfout_data_interval=$(( 3600 * 6 ))
frames_per_wrfout_file=1

batches=(
    OISST_OSTIA_sstskinON
    OISST_OSTIA_sstskinOFF
)

params=(
    "LA"       33.3 34.3 241.2 243.0
)
nparams=5


for batchname in "${batches[@]}"; do

    archive_root=$data_dir/WRF_RUNS/0.16deg/runs_${batchname}

    input_dirs=""
    amp=""
    for amp in $( seq -14 2 14 ); do
    #for amp in 2 ; do
        amp=$(  python3 -c "print('%.1f' % ($amp/10) )" )
        input_dirs="$input_dirs $archive_root/PRELIMINARY_PAT0_AMP${amp}/output/wrfout"
        amps="$amps $amp"
    done

    for (( i=0 ; i < $(( ${#params[@]} / $nparams )) ; i++ )); do

        box_label="${params[$(( i * $nparams + 0 ))]}"
        lat_beg="${params[$(( i * $nparams + 1 ))]}"
        lat_end="${params[$(( i * $nparams + 2 ))]}"
        lon_beg="${params[$(( i * $nparams + 3 ))]}"
        lon_end="${params[$(( i * $nparams + 4 ))]}"

        echo ":: box_label = $box_label"
        echo ":: lat_beg = $lat_beg"
        echo ":: lat_end = $lat_end"
        echo ":: lon_beg = $lon_beg"
        echo ":: lon_end = $lon_end"

       
        day_beg=0
        day_end=$(( $day_beg + 17 ))

        box_rng_str=$( python3 src/pretty_latlon.py --func box --lat-rng $lat_beg $lat_end --lon-rng $lon_beg $lon_end )
        time_str=$( printf "day%02d-%02d" $day_beg $day_end )

        output_dir=$fig_dir/response_ts/$batchname
        output=$output_dir/ts_${box_label}_${box_rng_str}_${time_str}.png

        mkdir -p $output_dir

        if [ -f "$output" ]; then

            echo "File already exists: $output"
            echo "Skip."

        else

            echo "Produce output: $output"
            python3 ./src/plot_box_timeseries.py               \
                --input-dirs ${input_dirs[@]}              \
                --exp-beg-time $exp_beg_time                 \
                --wrfout-data-interval $wrfout_data_interval \
                --frames-per-wrfout-file $frames_per_wrfout_file \
                --wrfout-suffix "_temp" \
                --time-rng $(( 24 * $day_beg )) $(( 24 * $day_end )) \
                --output $output \
                --lat-rng $lat_beg $lat_end \
                --lon-rng $lon_beg $lon_end \
                --no-legend \
                --varnames SST_NOLND PH500 PH850 IVT TTL_RAIN \
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
