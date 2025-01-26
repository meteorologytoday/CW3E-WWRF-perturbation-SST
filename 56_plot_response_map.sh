#!/bin/bash

source 00_setup.sh
source 98_trapkill.sh
    
nproc=1


batches=(
    OISST_OSTIA_sstskinON
    OISST_OSTIA_sstskinOFF
)

params=(
    "OPENOCN"  33.00 37.00 218.00 222.00
)


for batchname in "${batches[@]}"; do

    archive_root=$data_dir/WRF_RUNS/0.16deg/runs_${batchname}

    input_dirs=""
    amp=""
    for amp in -14 -10 10 14; do
        amp=$(  python3 -c "print('%.1f' % ($amp/10) )" )
        input_dirs="$input_dirs $archive_root/PRELIMINARY_PAT0_AMP${amp}/output/wrfout"
        amps="$amps $amp"
    done
    input_dirs_base=(
        $archive_root/PRELIMINARY_PAT0_AMP0.0/output/wrfout
    )


    exp_beg_time="2023-01-01T00:00:00"
    wrfout_data_interval=$(( 3600 * 3 ))
    frames_per_wrfout_file=1

    nparams=5
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

        for hr_beg in $( seq 360 -12 0 ) ; do
       
            hr_end=$(( $hr_beg + 12 ))

            box_rng_str=$( python3 src/pretty_latlon.py --func box --lat-rng $lat_beg $lat_end --lon-rng $lon_beg $lon_end )
            time_str=$( printf "hr%03d-%03d" $hr_beg $hr_end )
            
            output_dir=$fig_dir/response_map/$batchname
            output=$output_dir/diff_${time_str}.png
            
            mkdir -p $output_dir
            
            if [ -f "$output" ]; then
                
                echo "File already exists: $output"
                echo "Skip."
                
            else
                
                
                echo "Produce output: $output"
                python3 ./src/plot_response_map.py                   \
                    --input-dirs ${input_dirs[@]}                    \
                    --input-dirs-base "${input_dirs_base[@]}"        \
                    --exp-beg-time $exp_beg_time                     \
                    --wrfout-data-interval $wrfout_data_interval     \
                    --frames-per-wrfout-file $frames_per_wrfout_file \
                    --wrfout-suffix "_temp"                          \
                    --time-rng $hr_beg $hr_end                       \
                    --output $output                                 \
                    --lat-rng 0 65                                   \
                    --lon-rng 160 280                                \
                    --varnames SST_NOLND PH500 PH850 IVT TTL_RAIN_LP \
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

done
wait
echo "Done."
