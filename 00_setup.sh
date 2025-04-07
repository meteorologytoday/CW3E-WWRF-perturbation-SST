#!/bin/bash
py=python3
sh=bash

src_dir=src
data_dir=./data
data_sim_dir=$data_dir/sim_data
data_SQ15_dir=./data/data_SQ15
fig_dir=figures
fig_static_dir=figures_static
finalfig_dir=final_figures

gendata_dir=./gendata
preavg_dir=$gendata_dir/preavg

mkdir -p $fig_dir


source tools.sh
