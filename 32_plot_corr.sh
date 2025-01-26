#!/bin/bash

source 00_setup.sh
source 98_trapkill.sh
    
nproc=2

input_dir=gendata/corr_remote/method2/

input_fmt="$input_dir/scatter_OPENOCN_33.0N-37.0N_142.0W-138.0W_day%02d-%02d.nc"

days_to_avg=1

for day_beg in $( seq 17 -1 0 ) ; do

    day_end=$(( $day_beg + $days_to_avg ))
    output_dir=$fig_dir/corr_map/$batchname
    output=$output_dir/$( printf "corr_map_day%02d-%02d.png" $day_beg $day_end )
    
    title=$( printf "Day %02d - %02d" $day_beg $day_end )
    
    input=$( printf "$input_fmt" $day_beg $day_end ) 

    if [ -f "$output" ]; then

        echo "File already exists: $output"
        echo "Skip."

    else

        mkdir -p $output_dir

        echo "Produce output: $output"
        python3 ./src/plot_corr_map.py      \
            --input $input                  \
            --output $output                \
            --title  "$title"               \
            --lat-rng 0 65                  \
            --lon-rng 160 275               \
            --no-display                    \
            --sig-threshold 0.6             \
            --ncol 2                        \
            --varnames          corr_PH500_SST \
                                corr_PH850_SST \
                                corr_PRECIP_SST \
                                corr_TTL_RAIN_SST \
            &




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
