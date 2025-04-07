#!/bin/bash

source 00_setup.sh
source 98_trapkill.sh
    
nproc=3

archive_root=/home/t2hsu/projects/project-SST-spectrum/data/cropped

cropped_label=NPAC_0.1
params=(
    oisst      2022P72 OSTIA_UKMO  2022P72
    oisst      2023P01 OSTIA_UKMO  2023P01
    oisst      2022P72 oisst       2023P01
    OSTIA_UKMO 2022P72 OSTIA_UKMO  2023P01

)

nparams=4
for (( i=0 ; i < $(( ${#params[@]} / $nparams )) ; i++ )); do

    dataset1="${params[$(( i * $nparams + 0 ))]}"
    pentadstamp1="${params[$(( i * $nparams + 1 ))]}"
    dataset2="${params[$(( i * $nparams + 2 ))]}"
    pentadstamp2="${params[$(( i * $nparams + 3 ))]}"

    echo "Doing ${dataset1}_${pentadstamp1} - ${dataset2}_${pentadstamp2}"

    input_file1=$archive_root/$cropped_label/sst/$dataset1/${dataset1}_cropped_sst_${pentadstamp1}.nc
    input_file2=$archive_root/$cropped_label/sst/$dataset2/${dataset2}_cropped_sst_${pentadstamp2}.nc

    diff_label="${dataset1}-${pentadstamp1}_minus_${dataset2}-${pentadstamp2}"
    output_dir=$fig_dir/sst_difference
    output=$output_dir/diff_${diff_label}.png

    mkdir -p $output_dir
    
    if [ -f "$output" ]; then
        
        echo "File already exists: $output"
        echo "Skip."
        
    else
        
        
        echo "Produce output: $output"
        python3 ./src/plot_SST_difference.py                 \
            --input-files $input_file1 $input_file2          \
            --labels ${dataset1}_${pentadstamp1} ${dataset2}_${pentadstamp2}  \
            --output $output                                 \
            --extra-title "${yp}"                            \
            --lat-rng 0 65                                   \
            --lon-rng 160 280                                \
            --no-display  &


        proc_cnt=$(( $proc_cnt + 1))

        if (( $proc_cnt >= $nproc )) ; then
            echo "Max proc reached: $nproc"
            wait
            proc_cnt=0
        fi
    fi
done

wait
echo "Done."
