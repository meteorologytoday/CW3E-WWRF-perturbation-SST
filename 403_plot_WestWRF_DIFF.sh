#!/bin/bash

source 000_setup.sh
source 999_trapkill.sh

#varnames=( SST TTL_RAIN PSFC IVT IWV )
#varnames=( SST TTL_RAIN PSFC IWV IVT TOmA )
varnames=( SST IWV W-700 PBLH )

nproc=5

params=(
#    2023-01-07T00:00:00  "exp_20230107/PAT01_AMP-1.0/0-30,exp_20230107/PAT00_AMP0.0/0-30,exp_20230107/PAT01_AMP1.0/0-30,exp_20230107/PAT01_AMP2.0/0-30,exp_20230107/PAT01_AMP4.0/0-30" 1
    2023-01-07T00:00:00  "exp_20230107/PAT01_AMP-1.0/0-30,exp_20230107/PAT00_AMP0.0/0-30,exp_20230107/PAT01_AMP1.0/0-30" 1
#    2023-01-07T00:00:00  "exp_20230107/PAT00_AMP2.0/0-30,exp_20230107/PAT00_AMP1.0/0-30,exp_20230107/PAT00_AMP0.0/0-30,exp_20230107/PAT00_AMP-1.0/0-30" 2
)

nparams=3
for (( i=0 ; i < $(( ${#params[@]} / $nparams )) ; i++ )); do
    
    exp_beg_time="${params[$(( i * $nparams  + 0 ))]}"
    expblobs_str="${params[$(( i * $nparams  + 1 ))]}"
    ref="${params[$(( i * $nparams  + 2 ))]}"
    
    IFS=,; read -ra expblobs <<<"$expblobs_str"
    
    echo ":: ${expblobs[@]}"
    
    input_root=$gendata_dir/regrid_data/dx0.5
    output_root=$fig_dir/ens_compare
    mkdir -p $output_root

    time_beg=$(( 0 ))
    time_end=$(( 24 * 5 + 6 ))
    time_stride=6
    python3 ./src/plot_ensemble_diff_stat.py \
        --input-root $input_root             \
        --output-root $output_root           \
        --expblobs ${expblobs[@]} \
        --ref-expblob $ref                   \
        --varnames ${varnames[@]}            \
        --exp-beg-time $exp_beg_time         \
        --time-beg $time_beg \
        --time-end $time_end \
        --time-stride $time_stride \
        --lat-rng 10 70                      \
        --lon-rng $(( 360 - 200 )) $(( 360 - 105 ))           \
        --pval 0.1 \
        --extension png                                   \
        --nproc $nproc  &

    proc_cnt=$(( $proc_cnt + 1))

    if (( $proc_cnt >= $nproc )) ; then
        echo "Max proc reached: $nproc"
        wait
        proc_cnt=0
    fi

done

wait
echo "Done."
