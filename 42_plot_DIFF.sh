#!/bin/bash

source 00_setup.sh
        
varnames=( SST TTL_RAIN PSFC IVT )

nproc=10

source 60_verification_setup.sh    

if [ "$expname" = "exp_20230101" ] ; then

    params=(
        Perturb1-1  PAT00_AMP-1.0    Perturb1-1  PAT00_AMP1.0
    )
else

    params=(
        Perturb1  PAT00_AMP1.0    Perturb1  PAT00_AMP-1.0
    )

fi

nparams=4
for (( i=0 ; i < $(( ${#params[@]} / $nparams )) ; i++ )); do
    
    group1="${params[$(( i * $nparams  + 0 ))]}"
    subgroup1="${params[$(( i * $nparams + 1 ))]}"
    
    group2="${params[$(( i * $nparams  + 2 ))]}"
    subgroup2="${params[$(( i * $nparams + 3 ))]}"
    
    echo ":: group1  = $group1"
    echo ":: subgroup1 = $subgroup1"
    
    echo ":: group2  = $group2"
    echo ":: subgroup2 = $subgroup2"
    
    day_beg=0
    day_end=9

    input_root=$output_ens_stat_dir
    output_root=$fig_dir/ens_compare
    mkdir -p $output_root

    time_beg=0
    time_end=240
    time_stride=12
    python3 ./src/plot_ensemble_comparison.py    \
        --input-root $input_root             \
        --output-root $output_root           \
        --expnames $expname $expname         \
        --groups $group1 $group2             \
        --subgroup $subgroup1 $subgroup2     \
        --varnames ${varnames[@]}            \
        --exp-beg-time $exp_beg_time         \
        --time-beg $time_beg \
        --time-end $time_end \
        --time-stride $time_stride \
        --lat-rng 10 70                      \
        --lon-rng $(( 360 - 179 )) $(( 360 - 105 ))           \
        --pval 0.1 \
        --extension png                                   \
        --nproc $nproc 

done
