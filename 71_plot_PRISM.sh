#!/bin/bash

source 00_setup.sh
    

output_dir=$fig_dir/PRISM

mkdir -p $output_dir

params=(
    2023-01-01 2023-01-05
    2023-01-01 2023-01-10
    2023-01-01 2023-01-15
    2023-01-01 2023-01-20
)

nparams=2

for (( i=0 ; i < $(( ${#params[@]} / $nparams )) ; i++ )); do

    beg_date="${params[$(( i * $nparams + 0 ))]}"
    end_date="${params[$(( i * $nparams + 1 ))]}"

    echo ":: beg_date = $beg_date"
    echo ":: end_date = $end_date"

    output_file=$output_dir/"${beg_date}_to_${end_date}.png"
 
    python3 ./src/plot_PRISM_rainfall.py            \
        --date-rng $beg_date $end_date              \
        --output $output_file                       \
        --lon-rng $(( 360 - 125 )) $(( 360 - 114 )) \
        --lat-rng 31.5 42.5                         \
        --no-display 

done

