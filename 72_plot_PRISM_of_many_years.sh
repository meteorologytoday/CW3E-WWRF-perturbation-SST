#!/bin/bash

source 98_trapkill.sh
source 00_setup.sh
    

nproc=2


mkdir -p $output_dir



half_window_size=0

for year in $( seq 2023 2023 ) ; do
    params="$params $(( $year-1 ))-11-01 ${year}-01-10"
done

IFS=" "
params=($params)

output_dir=$fig_dir/PRISM_map/half_window_size-${half_window_size}

mkdir -p $output_dir

nparams=2

for (( i=0 ; i < $(( ${#params[@]} / $nparams )) ; i++ )); do

    beg_date="${params[$(( i * $nparams + 0 ))]}"
    end_date="${params[$(( i * $nparams + 1 ))]}"

    echo ":: beg_date = $beg_date"
    echo ":: end_date = $end_date"

    IFS=" "
    date_list=( $( python3 -c "import pandas as pd; dts=pd.date_range('${beg_date}', '${end_date}', freq='D', inclusive='both'); print(' '.join([dt.strftime('%Y-%m-%d') for dt in dts]))" ) )
    
   
    for d in ${date_list[@]} ; do

        output_file=$output_dir/"precip_${d}.png"

        if [ -f "${output_file}" ] ; then
            echo "File $output_file already exists. Skip."
        else

            python3 ./src/plot_PRISM_rainfall.py             \
                --date-selection-mode date-center            \
                --stat mean                                  \
                --date-center $d                             \
                --half-window-size $half_window_size         \
                --output $output_file                        \
                --precip-levs 0 5 10 15 20 25 30 35 40 45 50 \
                --lon-rng $(( 360 - 125 )) $(( 360 - 114 ))  \
                --lat-rng 31.5 42.5                          \
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

echo "All Done."
