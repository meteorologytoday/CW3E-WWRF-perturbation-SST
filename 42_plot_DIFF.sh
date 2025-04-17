#!/bin/bash

source 00_setup.sh
        
#varnames=( SST TTL_RAIN PSFC IVT )
#varnames=(  IWV WND::850 IVT TTL_RAIN SST )
varnames=(  TTL_RAIN  )
#varnames=(  PH::850 PSFC SST )

nproc=1

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

nparams=2
for (( i=0 ; i < $(( ${#params[@]} / $nparams )) ; i++ )); do
    
    group1="${params[$(( i * $nparams  + 0 ))]}"
    group2="${params[$(( i * $nparams  + 1 ))]}"
    
    echo ":: group1  = $group1"
    echo ":: group2  = $group2"
    
    input_root=$output_wrf_regrid_dir
    output_root=$fig_dir/ens_compare
    mkdir -p $output_root

    time_beg=0
    time_end=246
    time_stride=6
    python3 ./src/plot_ensemble_diff_stat.py \
        --input-root $input_root             \
        --output-root $output_root           \
        --expnames $expname $expname         \
        --groups $group1 $group2             \
        --ens-ids $ens_ids                   \
        --varnames ${varnames[@]}            \
        --exp-beg-time $exp_beg_time         \
        --time-beg $time_beg \
        --time-end $time_end \
        --time-stride $time_stride \
        --lat-rng 10 70                      \
        --lon-rng $(( 360 - 179 )) $(( 360 - 105 ))           \
        --pval 0.1 \
        --extension png                                   \
        --plot-quartile \
        --nproc $nproc 

done
