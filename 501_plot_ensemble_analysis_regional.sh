#!/bin/bash

source 000_setup.sh
source 999_trapkill.sh
    
nproc=10

output_root=$fig_dir/ensemble_analysis_regional
 
input_root_20230107=$gendata_dir/stat_regional/20230107

function case20230107 {
    echo $gendata_dir/stat_regional/20230107/PAT${1}_AMP${2}_2023-01-12.nc
}


params=(
    "20230112_constSST" "$(case20230107 01 -1.0),$(case20230107 00 0.0),$(case20230107 01 1.0),$(case20230107 01 2.0),$(case20230107 01 4.0)" "CA,city_SD,city_SF,city_LA,Dam_Oroville" "TTL_RAIN" "SST perturbation" "-1,0,1,2,4"
    "20230112_PAT00" "$(case20230107 00 -1.0),$(case20230107 00 0.0),$(case20230107 00 1.0),$(case20230107 00 2.0)" "CA,city_SD,city_SF,city_LA,Dam_Oroville" "TTL_RAIN" "SST perturbation" "-1,0,1,2"
)

mkdir -p $output_root

nparams=6
for (( i=0 ; i < $(( ${#params[@]} / $nparams )) ; i++ )); do
    
    casename="${params[$(( i * $nparams + 0 ))]}"
    input_files="${params[$(( i * $nparams + 1 ))]}"
    regions="${params[$(( i * $nparams + 2 ))]}"
    varnames="${params[$(( i * $nparams + 3 ))]}"
    xlabel="${params[$(( i * $nparams + 4 ))]}"
    xvals="${params[$(( i * $nparams + 5 ))]}"
    
    IFS=',' read -ra input_files <<< "$input_files"
    IFS=',' read -ra regions <<< "$regions"
    IFS=',' read -ra varnames <<< "$varnames"
    IFS=',' read -ra xvals <<< "$xvals"
   
    for region in ${regions[@]} ; do
    for varname in ${varnames[@]} ; do
        
        output_file=$output_root/${casename}-${region}-${varname}.png
        
        if [ -f "$output" ]; then
            
            echo "File already exists: $output"
            echo "Skip."
            
        else
        
            python3 src/plot_ensemble_analysis_regional.py \
                --input-files ${input_files[@]} \
                --varname $varname \
                --region $region \
                --xvals ${xvals[@]} \
                --xlabel "$xlabel" \
                --output $output_file \
                --no-display &
        
            proc_cnt=$(( $proc_cnt + 1))

            if (( $proc_cnt >= $nproc )) ; then
                echo "Max proc reached: $nproc"
                wait
                proc_cnt=0
            fi

        fi
    done
    done
done

wait
echo "Done."
