#!/bin/bash

source 00_setup.sh

#varnames=( SST TTL_RAIN PSFC IVT IWV )
varnames=( SST PSFC )

nproc=2

params=(
#    2023-01-07T00:00:00  2023010700   gefs  "0-30"  exp_20230107 PAT00_AMP0.0  "0-30"  exp_20230107 PAT00_AMP-1.0  "0-30"
#    2023-01-07T00:00:00  2023010700   gefs  "31-61"  exp_20230107 PAT00_AMP0.0  "0-30"  exp_20230107 PAT00_AMP-1.0  "0-30"
    2023-01-07T00:00:00  2023010700   gefs  "62-79"  exp_20230107 PAT00_AMP0.0  "0-30"  exp_20230107 PAT00_AMP-1.0  "0-30"
)

nparams=9
for (( i=0 ; i < $(( ${#params[@]} / $nparams )) ; i++ )); do
    
    exp_beg_time="${params[$(( i * $nparams  + 0 ))]}"
    
    expname1="${params[$(( i * $nparams  + 1 ))]}"
    group1="${params[$(( i * $nparams  + 2 ))]}"
    ens_rng1="${params[$(( i * $nparams  + 3 ))]}"

    expname2="${params[$(( i * $nparams  + 4 ))]}"
    group2="${params[$(( i * $nparams  + 5 ))]}"
    ens_rng2="${params[$(( i * $nparams  + 6 ))]}"

    expname3="${params[$(( i * $nparams  + 7 ))]}"
    group3="${params[$(( i * $nparams  + 8 ))]}"
    ens_rng3="${params[$(( i * $nparams  + 6 ))]}"



    echo ":: ${expname1}::${group1}::${ens_rng1}"
    echo ":: ${expname2}::${group2}::${ens_rng2}"
    echo ":: ${expname3}::${group3}::${ens_rng3}"
    
    input_root=$gendata_dir/regrid_data
    output_root=$fig_dir/ens_compare
    mkdir -p $output_root

    time_beg=$(( 24 * 0 ))
    time_end=$(( 24 * 6 + 6 ))
    time_stride=6
    python3 ./src/plot_ensemble_diff_stat.py \
        --input-root $input_root             \
        --output-root $output_root           \
        --expnames $expname1 $expname2 $expname3  \
        --groups $group1 $group2 $group3          \
        --ens-rngs $ens_rng1 $ens_rng2 $ens_rng3  \
        --varnames ${varnames[@]}            \
        --exp-beg-time $exp_beg_time         \
        --time-beg $time_beg \
        --time-end $time_end \
        --time-stride $time_stride \
        --lat-rng 10 70                      \
        --lon-rng $(( 360 - 179 )) $(( 360 - 105 ))           \
        --pval 0.1 \
        --extension .${ens_rng1}.png                                   \
        --nproc $nproc 

done
