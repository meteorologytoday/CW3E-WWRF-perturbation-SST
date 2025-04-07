#!/bin/bash

source 00_setup.sh


input_dir=$gendata_dir/PRISM_historical
output_dir=$fig_dir/PRISM_historical


beg_year=2005
end_year=2025

for year in $( seq $beg_year $end_year ) ; do
    for region in CA ; do
        output_dir=$fig_dir/PRISM_historical_$region
        mkdir -p $output_dir
        #output_file=$output_dir/PRISM_historical_region-${region}_year-${beg_year}-${end_year}.png
        output_file=$output_dir/PRISM_historical_region-${region}_year-${year}.png

        python3 ./src/plot_PRISM_rainfall_timeseries.py \
            --wateryears ${year}-${year}        \
            --plot-pentad-rng 54 $(( 54 + 36 ))         \
            --output $output_file                       \
            --region $region                            \
            --input-dir $input_dir                      \
            --no-display 
    done
done
