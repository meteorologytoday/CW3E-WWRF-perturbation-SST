#!/bin/bash

source 000_setup.sh
source 999_trapkill.sh
    
nproc=1

output_root=$fig_dir/sensitivity_fullanalysis
 
params=(
    "case_IWV_20230107"  "SO3_gendata/correlation/dx0.5/20230107_southCal/sensitivity_PAT00_AMP0.0_sens-2023-01-09_00-IWV_target-2023-01-12_00-TTL_RAIN.nc" 30
    "case_IWV_20230112"  "SO3_gendata/correlation/dx0.5/20230112_southCal/sensitivity_CTL_sens-2023-01-12_06-IWV_target-2023-01-17_00-TTL_RAIN.nc" 30
#    "WestWRF_IWV"  "SO3_gendata/correlation/dx0.5/20230107_southCal/sensitivity_WestWRF-gefs_sens-2023-01-09_00-IWV_target-2023-01-12_00-TTL_RAIN.nc" 30
)

mkdir -p $output_root

nparams=3
for (( i=0 ; i < $(( ${#params[@]} / $nparams )) ; i++ )); do
    
    casename="${params[$(( i * $nparams + 0 ))]}"
    input_file="${params[$(( i * $nparams + 1 ))]}"
    modes="${params[$(( i * $nparams + 2 ))]}"

    python3 src/plot_sensitivity_analysis.py \
        --input-file ${input_file}  \
        --output-root $output_root   \
        --output-label $casename    \
        --regions CA city_LA city_SF \
        --modes $modes \
        --first-modes 10 \
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
