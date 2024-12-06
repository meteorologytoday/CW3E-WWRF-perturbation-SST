#!/bin/bash

source 00_setup.sh
source 98_trapkill.sh
    
nproc=1

input_dir=gendata/corr_remote/$batchname

input_fmt="$input_dir/scatter_OPENOCN_33.0N-37.0N_142.0W-138.0W_hours%03d-%03d.nc"

hours_to_avg=6

for hour_beg in $( seq $(( 15 * 24 )) -6 0 ) ; do

    hour_end=$(( $hour_beg + $hours_to_avg ))
    output_dir=$fig_dir/corr_map/$batchname
    output=$output_dir/$( printf "corr_map_hours%03d-%03d.png" $hour_beg $hour_end )
    
    title=$( printf "Hour %d - %d" $hour_beg $hour_end )
    
    input=$( printf "$input_fmt" $hour_beg $hour_end ) 

    if [ -f "$output" ]; then

        echo "File already exists: $output"
        echo "Skip."

    else

        mkdir -p $output_dir

        echo "Produce output: $output"
        python3 $src_dir/plot_corr_map.py      \
            --input $input                  \
            --output $output                \
            --title  "$title"               \
            --lat-rng 0 65                  \
            --lon-rng 160 275               \
            --sig-threshold      0.45        \
            --no-display                    \
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
