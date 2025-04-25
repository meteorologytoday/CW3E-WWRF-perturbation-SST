#!/bin/bash

source 00_setup.sh
        
#varnames=( SST TTL_RAIN PSFC IVT )
#varnames=(  IWV WND::850 IVT TTL_RAIN SST )
varnames=(  SST TTL_RAIN  )
#varnames=(  PH::850 PSFC SST )

nproc=10

source 60_verification_setup.sh    

if [ "$expname" = "exp_20230101" ] ; then

    params=(
        PAT00_AMP-1.0 PAT00_AMP1.0
    )

else

    params=(
        PAT00_AMP1.0 PAT00_AMP-1.0
    )

fi

ens_rng=0-30

nparams=2
for (( i=0 ; i < $(( ${#params[@]} / $nparams )) ; i++ )); do
    
    group1="${params[$(( i * $nparams  + 0 ))]}"
    group2="${params[$(( i * $nparams  + 1 ))]}"
    
    echo ":: group1  = $group1"
    echo ":: group2  = $group2"
    
    input_root=$wrf_regrid_dir
    output_root=$fig_dir/ens_compare
    mkdir -p $output_root

    time_beg=$(( 24 * 5 ))
    time_end=$(( 24 * 5 + 6 ))
    time_stride=6
    python3 ./src/plot_ensemble_diff_stat.py \
        --input-root $input_root             \
        --output-root $output_root           \
        --expnames $expname $expname         \
        --groups $group1 $group2             \
        --ens-rngs $ens_rng $ens_rng         \
        --varnames ${varnames[@]}            \
        --exp-beg-time $exp_beg_time         \
        --time-beg $time_beg \
        --time-end $time_end \
        --time-stride $time_stride \
        --lat-rng 30  50                      \
        --lon-rng $(( 360 - 130 )) $(( 360 - 110 ))           \
        --pval 0.1 \
        --quantiles 0 0.25 0.75 1.0 \
        --extension png                                   \
        --plot-quantile \
        --nproc $nproc 

done
