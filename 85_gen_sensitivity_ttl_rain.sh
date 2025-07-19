#!/bin/bash

nproc=1

source 000_setup.sh
source 999_trapkill.sh



dx=0.5 
input_root=$regriddata_dir/dx${dx}
output_root=$gendata_dir/correlation/dx${dx}

region_mask_file=$gendata_dir/region_mask/mask.nc
#sens_regrid_file=$gendata_dir/regrid_idx/dx0.5/regrid_idx_sensdx6.0_mygrid2sengrid_ocn.nc
sens_regrid_file=$gendata_dir/regrid_idx/dx0.5/regrid_idx_sensdx1.0_sensdy1.0_mygrid2sengrid_ocn.nc
#sens_regrid_file=$gendata_dir/regrid_idx/dx0.5/regrid_idx_sensdx2.0_sensdy2.0_mygrid2sengrid_ocn.nc

target_varnames=(
    "TTL_RAIN"
)

sens_varnames=(
#    "IVT"
    "IWV"
#    "TTL_RAIN"
)

params=(
    "2023-01-09T00" "2023-01-12" "exp_20230107/PAT00_AMP0.0/0-30" "20230107_southCal" "PAT00_AMP0.0"
#    "2023-01-09T00" "2023-01-12" "exp_20230107/PAT00_AMP0.0/0-30" "20230107_southCal" "PAT00_AMP0.0"
#    "2023-01-09T00" "2023-01-12" "2023010700/gefs/0-79" "20230107_southCal" "WestWRF-gefs"
#    "2023-01-12" "2023-01-12" "exp_20230107/PAT00_AMP0.0/0-30" "20230107_southCal" "PAT00_AMP0.0"   34.0 36.0 239 241

)

nparams=5
for (( i=0 ; i < $(( ${#params[@]} / $nparams )) ; i++ )); do
    
    sens_time="${params[$(( i * $nparams + 0 ))]}"
    target_time="${params[$(( i * $nparams + 1 ))]}"
    expblob="${params[$(( i * $nparams + 2 ))]}"
    output_dir="${params[$(( i * $nparams + 3 ))]}"
    output_prefix="${params[$(( i * $nparams + 4 ))]}"
 
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
            --sens-regrid-file $sens_regrid_file \
            --sens-time      $sens_time       \
            --target-mask-type "mask_file"    \
            --target-mask-file $region_mask_file \
            --target-mask-file-regions "CA" "south_CA" "north_CA" "city_LA" "city_SF"    \
            --output-dir    $output_root/$output_dir          \
            --output-prefix $output_prefix    

            #--target-mask-file-regions "CA" "south_CA" "north_CA" "city_LA" "city_SF"    \



    done
    done
done

echo "Done."
