#!/bin/bash

source 000_setup.sh
source 999_trapkill.sh
    
nproc=10

output_root=$fig_dir/sensitivity
 
params=(
    "case_IWV"  "SO3_gendata/correlation/dx0.5/20230107_southCal/sensitivity_PAT00_AMP0.0_sens-2023-01-09_00-IWV_target-2023-01-12_00-TTL_RAIN.nc"
    "case_IVT"  "SO3_gendata/correlation/dx0.5/20230107_southCal/sensitivity_PAT00_AMP0.0_sens-2023-01-09_00-IVT_target-2023-01-12_00-TTL_RAIN.nc"
    "case_IWV_WestWRF"  "SO3_gendata/correlation/dx0.5/20230107_southCal/sensitivity_WestWRF-gefs_sens-2023-01-09_00-IWV_target-2023-01-12_00-TTL_RAIN.nc" 
    "case_IVT_WestWRF"  "SO3_gendata/correlation/dx0.5/20230107_southCal/sensitivity_WestWRF-gefs_sens-2023-01-09_00-IWV_target-2023-01-12_00-TTL_RAIN.nc" 
)

mkdir -p $output_root

nparams=2
for (( i=0 ; i < $(( ${#params[@]} / $nparams )) ; i++ )); do
    
    casename="${params[$(( i * $nparams + 0 ))]}"
    input_file="${params[$(( i * $nparams + 1 ))]}"
    
    output_dir=$output_root/$casename

    mkdir -p $output_dir
    
    python3 src/plot_sensitivity.py \
        --title $casename           \
        --input-file ${input_file}  \
        --output-root $output_dir   \
        --lat-rng 10 70 \
        --lon-rng 180 270 &

    proc_cnt=$(( $proc_cnt + 1))

    if (( $proc_cnt >= $nproc )) ; then
        echo "Max proc reached: $nproc"
        wait
        proc_cnt=0
    fi

done

wait
echo "Done."
