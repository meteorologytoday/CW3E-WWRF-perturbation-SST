#!/bin/bash

source 000_setup.sh
source 999_trapkill.sh

varnames=( QVAPOR W  )

nproc=5
    
time_stride=6

params=(
    2023-01-07T00:00:00  "exp_20230107/PAT01_AMP-1.0/0-30,exp_20230107/PAT00_AMP0.0/0-30,exp_20230107/PAT01_AMP1.0/0-30" 1  meridional 205 208 20 40 24 $(( 48 + 6 )) 
#    2023-01-07T00:00:00  "exp_20230107/PAT01_AMP-1.0/0-30,exp_20230107/PAT00_AMP0.0/0-30,exp_20230107/PAT01_AMP1.0/0-30" 1  meridional 219 221 20 40 42 $(( 48 + 12 )) 
)

nparams=10
for (( i=0 ; i < $(( ${#params[@]} / $nparams )) ; i++ )); do
    
    exp_beg_time="${params[$(( i * $nparams  + 0 ))]}"
    expblobs_str="${params[$(( i * $nparams  + 1 ))]}"
    ref="${params[$(( i * $nparams  + 2 ))]}"
    cxs_dir="${params[$(( i * $nparams  + 3 ))]}"
    cxs_loc_rng_beg="${params[$(( i * $nparams  + 4 ))]}"
    cxs_loc_rng_end="${params[$(( i * $nparams  + 5 ))]}"
    cxs_rng_beg="${params[$(( i * $nparams  + 6 ))]}"
    cxs_rng_end="${params[$(( i * $nparams  + 7 ))]}"
    time_beg="${params[$(( i * $nparams  + 8 ))]}"
    time_end="${params[$(( i * $nparams  + 9 ))]}"
    
    IFS=,; read -ra expblobs <<<"$expblobs_str"
    
    echo ":: ${expblobs[@]}"
    
    input_root=$gendata_dir/regrid_data/dx0.5
    output_root=$fig_dir/ens_compare_cxs
    mkdir -p $output_root


    python3 ./src/plot_ensemble_diff_stat_cross-section.py \
        --input-root $input_root             \
        --output-root $output_root           \
        --expblobs ${expblobs[@]}            \
        --ref-expblob $ref                   \
        --varnames ${varnames[@]}            \
        --exp-beg-time $exp_beg_time         \
        --time-beg $time_beg                 \
        --time-end $time_end                 \
        --time-stride $time_stride           \
        --cxs-dir $cxs_dir                   \
        --cxs-loc-rng $cxs_loc_rng_beg $cxs_loc_rng_end  \
        --cxs-rng $cxs_rng_beg $cxs_rng_end  \
        --pval 0.1                           \
        --extension png                      \
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
