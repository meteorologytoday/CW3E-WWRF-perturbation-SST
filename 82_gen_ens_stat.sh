#!/bin/bash

nproc=5

source 000_setup.sh
source 999_trapkill.sh
    
input_root=$regriddata_dir/dx0.5
output_root=$gendata_dir/stat


varnames="IVT,TTL_RAIN,PSFC,T2,U10,V10"
varnames="TOmA"
varnames="W"
params=(

    "2023-01-07" "exp_20230107/PAT01_AMP2.0/0-30"           "20230107" "PAT01_AMP2.0"    $varnames
    "2023-01-07" "exp_20230107/PAT01_AMP4.0/0-30"           "20230107" "PAT01_AMP4.0"    $varnames
    "2023-01-07" "exp_20230107/PAT01_AMP1.0/0-30"            "20230107" "PAT01_AMP1.0"   $varnames
    "2023-01-07" "exp_20230107/PAT01_AMP-1.0/0-30"           "20230107" "PAT01_AMP-1.0"  $varnames


    "2023-01-07" "exp_20230107/PAT00_AMP0.0/0-30"            "20230107" "PAT00_AMP0.0"   $varnames
    "2023-01-07" "exp_20230107/PAT00_AMP-1.0/0-30"           "20230107" "PAT00_AMP-1.0"  $varnames
    "2023-01-07" "exp_20230107/PAT00_AMP1.0/0-30"            "20230107" "PAT00_AMP1.0"   $varnames
    "2023-01-07" "exp_20230107/PAT00_AMP2.0/0-30"            "20230107" "PAT00_AMP2.0"   $varnames


#    "2023-01-07" "2023010700/gefs/0-79"                      "20230107"  "WESTWRF-GEFS"     "TTL_RAIN"
#    "2023-01-07" "2023010700/gefs/0-79|2023010700/ecm/0-119" "20230107"  "WESTWRF-ALL"      "TTL_RAIN"
#    "2023-01-07" "exp_20230107/PAT00_AMP1.0/0-30"            "20230107" "AMP1"  "TTL_RAIN"
#    "2023-01-07" "exp_20230107/PAT00_AMP-1.0/0-30"           "20230107" "AMP-1" "TTL_RAIN"
)

nparams=5
for (( i=0 ; i < $(( ${#params[@]} / $nparams )) ; i++ )); do

    exp_beg_time="${params[$(( i * $nparams + 0 ))]}"
    expblob="${params[$(( i * $nparams + 1 ))]}"
    output_dir="${params[$(( i * $nparams + 2 ))]}"
    output_prefix="${params[$(( i * $nparams + 3 ))]}"
    varnames="${params[$(( i * $nparams + 4 ))]}"

    echo ":: exp_beg_time      = $exp_beg_time"
    echo ":: expblob           = $expblob"
    echo ":: output_dir        = $output_dir"
    echo ":: output_prefix     = $output_prefix"
    echo ":: varnames          = $varnames"
    
    IFS="," read -ra varnames <<< "$varnames" 


    python3 src/gen_ensemble_analysis.py  \
        --input-root $input_root          \
        --nproc $nproc                    \
        --expblob $expblob                \
        --varnames ${varnames[@]}         \
        --exp-beg-time $exp_beg_time      \
        --time-beg 0                      \
        --time-end 120                    \
        --time-stride 6                          \
        --output-dir    $output_root/$output_dir \
        --output-prefix $output_prefix    

done


echo "Done."
