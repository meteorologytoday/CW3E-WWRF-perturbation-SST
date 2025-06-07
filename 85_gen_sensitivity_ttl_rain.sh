#!/bin/bash

nproc=1

source 000_setup.sh
source 999_trapkill.sh
    
input_root=$regriddata_dir
output_root=$gendata_dir/correlation


target_varnames=(
    "TTL_RAIN"
)

sens_varnames=(
#    "Q2"
    "TTL_RAIN"
    "T2"
)

params=(
#    "2023-01-11" "2023-01-12" "exp_20230107/PAT00_AMP0.0/0-30" "20230107_southCal" "PAT00_AMP0.0"   34.0 36.0 239 241
    "2023-01-12" "2023-01-12" "exp_20230107/PAT00_AMP0.0/0-30" "20230107_southCal" "PAT00_AMP0.0"   34.0 36.0 239 241

)

nparams=9
for (( i=0 ; i < $(( ${#params[@]} / $nparams )) ; i++ )); do
    
    sens_time="${params[$(( i * $nparams + 0 ))]}"
    target_time="${params[$(( i * $nparams + 1 ))]}"
    expblob="${params[$(( i * $nparams + 2 ))]}"
    output_dir="${params[$(( i * $nparams + 3 ))]}"
    output_prefix="${params[$(( i * $nparams + 4 ))]}"
 
    target_lat_beg="${params[$(( i * $nparams + 5 ))]}"
    target_lat_end="${params[$(( i * $nparams + 6 ))]}"
    target_lon_beg="${params[$(( i * $nparams + 7 ))]}"
    target_lon_end="${params[$(( i * $nparams + 8 ))]}"

    echo ":: sens_time         = $sens_time"
    echo ":: target_time       = $target_time"
    echo ":: expblob           = $expblob"
    echo ":: output_dir        = $output_dir"
    echo ":: output_prefix     = $output_prefix"
    
    for target_varname in "${target_varnames[@]}"; do
    for sens_varname in "${sens_varnames[@]}"; do
 
        python3 src/gen_sensitivity_analysis.py  \
            --input-root $input_root          \
            --expblob $expblob                \
            --target-varname $target_varname  \
            --target-time    $target_time     \
            --sens-varname   $sens_varname    \
            --sens-time      $sens_time       \
            --target-lat-rng $target_lat_beg $target_lat_end  \
            --target-lon-rng $target_lon_beg $target_lon_end  \
            --output-dir    $output_root/$output_dir          \
            --output-prefix $output_prefix    

    done
    done
done

echo "Done."
