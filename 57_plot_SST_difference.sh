#!/bin/bash

source 00_setup.sh
source 98_trapkill.sh
    
nproc=5

archive_root=/home/t2hsu/projects/project-SST-spectrum/data/cropped

cropped_label=NPAC_0.1
params=(
    oisst GHRSST-Mean
#    oisst OSTIA_UKMO
)

nparams=2
for (( i=0 ; i < $(( ${#params[@]} / $nparams )) ; i++ )); do

    dataset1="${params[$(( i * $nparams + 0 ))]}"
    dataset2="${params[$(( i * $nparams + 1 ))]}"

    for y in 2016; do
    for p in 70 71 72; do # $( 0 17 ); do

        yp=$( printf "%04dP%02d" $y $p )
        
        echo "Doing $dataset1 - $dataset2 . yp = $yp"
        
        input_file1=$archive_root/$cropped_label/sst/$dataset1/${dataset1}_cropped_sst_${yp}.nc
        input_file2=$archive_root/$cropped_label/sst/$dataset2/${dataset2}_cropped_sst_${yp}.nc

        diff_label="${dataset1}_minus_${dataset2}"
        output_dir=$fig_dir/sst_difference/$diff_label
        output=$output_dir/diff_${diff_label}_${yp}.png

        mkdir -p $output_dir
        
        if [ -f "$output" ]; then
            
            echo "File already exists: $output"
            echo "Skip."
            
        else
            
            
            echo "Produce output: $output"
            python3 ./src/plot_SST_difference.py                 \
                --input-files $input_file1 $input_file2          \
                --labels $dataset1 $dataset2                     \
                --output $output                                 \
                --extra-title "${yp}"                            \
                --lat-rng 0 65                                   \
                --lon-rng 160 280                                \
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
