#!/bin/bash

if [ -n "${___SETUP___}"  ]; then
    echo "___TRAKILL___ already set, skip it."
else

    py=python3
    sh=bash

    src_dir=src
    data_dir=./data
    data_sim_dir=$data_dir/sim_data
    data_SQ15_dir=./data/data_SQ15
    fig_dir=figures
    fig_static_dir=figures_static
    finalfig_dir=final_figures

    gendata_dir=./SO3_gendata
    regriddata_dir=$gendata_dir/regrid_data

    mkdir -p $fig_dir


    source tools.sh

    ___SETUP___=TRUE
fi
