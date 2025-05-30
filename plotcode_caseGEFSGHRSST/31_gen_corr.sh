#!/bin/bash

source 00_setup.sh
source 98_trapkill.sh
 
params=(
    "OPENOCN"  33.00 37.00 218.00 222.00
)
   
nproc=1

input_dirs=""
amps=""
for pat in 0 ; do
for amp_int in $( seq -10 10 ) ; do
#for amp_int in -10 0 10 ; do
#for amp_int in 10 ; do
    amp=$(  python3 -c "print('%.1f' % ( $amp_int / 10), )" )
    input_dirs="$input_dirs $archive_root/PRELIMINARY_PAT${pat}_AMP${amp}/output/wrfout"
    amps="$amps $amp"
done
done

hours_to_avg=6

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

    for hour_beg in $( seq $(( 15 * 24 )) -6 0 ) ; do
   
        hour_end=$(( $hour_beg + hours_to_avg ))

        box_rng_str=$( python3 ${src_dir}/pretty_latlon.py --func box --lat-rng $lat_beg $lat_end --lon-rng $lon_beg $lon_end )
        time_str=$( printf "hours%03d-%03d" $hour_beg $hour_end )

        output_dir=$gendata_dir/corr_remote/$batchname/
        output=$output_dir/scatter_${box_label}_${box_rng_str}_${time_str}.nc

        mkdir -p $output_dir

        if [ -f "$output" ]; then

            echo "File already exists: $output"
            echo "Skip."

        else

            echo "Produce output: $output"
            python3 ${src_dir}/gen_correlation.py                   \
                --input-dirs $input_dirs                       \
                --corr-type remote                             \
                --corr-varnames-1 SST_NOLND PH850 PH500 PRECIP TTL_RAIN  \
                --corr-varname-2 SST                           \
                --exp-beg-time $exp_beg_time                   \
                --wrfout-data-interval $wrfout_data_interval   \
                --frames-per-wrfout-file $frames_per_wrfout_file \
                --wrfout-suffix "_temp" \
                --time-rng $hour_beg $hour_end \
                --output $output \
                --corr-box-lat-rng $lat_beg $lat_end \
                --corr-box-lon-rng $lon_beg $lon_end &


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
