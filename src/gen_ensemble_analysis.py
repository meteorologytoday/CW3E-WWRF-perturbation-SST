from multiprocessing import Pool
import multiprocessing

import traceback
import xarray as xr
import pandas as pd
import numpy as np
import argparse
import tool_fig_config
import wrf_load_helper 
import wrf_preprocess
import wrf_varname_info
import datetime
import os
from pathlib import Path

import CRPS_tools
import PRISM_tools
import cmocean
from scipy import sparse
import WRF_ens_tools

import re

def doJob(details, detect_phase=False):

    # phase \in ['detect', 'work']
    result = dict(details = details, status="UNKNOWN", need_work=False, detect_phase=detect_phase)

    try:
        
        varname     = details["varname"]
        input_root  = Path(details["input_root"])
        expblob     = details["expblob"]
        
        target_time     = pd.Timestamp(details["target_time"])
        
        output_dir     = Path(details["output_dir"]) 
        output_prefix   = details["output_prefix"] 
        
        output_file = output_dir / ("%s_%s.nc" % (
            output_prefix,
            target_time.strftime("%Y-%m-%d"),
        ))
        
        # Detecting
        result["output_file"] = output_file
        
        # First round is just to decide which files
        # to be processed to enhance parallel job 
        # distribution. I use variable `phase` to label
        # this stage.
        if detect_phase is True:

            result['need_work'] = not output_file.exists()

            if result['need_work']:
                result['status'] = 'OK'
            else:           
                result['status'] = 'OK'
                result['need_work'] = False

            return result
        
        print("##### Doing time: ", target_time)
        
        output_file.parent.mkdir(parents=True, exist_ok=True)
        expsets = WRF_ens_tools.parseExpblob(expblob)
       
        ds = [] 
        ens_cnt = 0
        for expname, group, ens_rng in expsets:

            print("Expname/group : %s/%s" % (expname, group))            
            print("Ensemble ids: %s" % ( ",".join(["%d" % i for i in ens_rng] ) ) )
            
            print("Load %s - %s" % (expname, group,)) 
            _ds = WRF_ens_tools.loadGroup(expname, group, ens_rng, varname, target_time, root=input_root)
            _ds = _ds.assign_coords({"ens" : np.arange(_ds.dims["ens"]) + ens_cnt})
            ds.append(_ds)

            ens_cnt += _ds.dims["ens"]

        da = xr.merge(ds).transpose("ens", "time", "lat", "lon")[varname]
        
        merge_data = []
        merge_data.append(da.std(dim="ens").expand_dims(dim={"stat": ["std",]}, axis=1).rename(varname))
        merge_data.append(da.mean(dim="ens").expand_dims(dim={"stat": ["mean",]}, axis=1).rename(varname))
        merge_data.append(da.count(dim="ens").expand_dims(dim={"stat": ["count",]}, axis=1).rename(varname))

        ds_output = xr.merge(merge_data)
        
        print("Saving output: ", output_file)
        ds_output.to_netcdf(output_file, unlimited_dims="time")

        if output_file.exists():
            print("File `%s` is generated." % (str(output_file),))


        result['status'] = 'OK'

    except Exception as e:

        result['status'] = 'ERROR'
        #traceback.print_stack()
        traceback.print_exc()
        print(e)


    return result

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--input-root', type=str, help='Input directories.', required=True)
    parser.add_argument('--expblob', type=str, help='Input directories.', required=True)
    parser.add_argument('--varnames', type=str, nargs="+", help='Variables processed. If only a particular layer is selected, use `-` to separate the specified layer. Example: PH-850', required=True)
    parser.add_argument('--exp-beg-time', type=str, help='analysis beg time', required=True)
    parser.add_argument('--time-beg',    type=int, help='Beginning of time. Hours` after `--exp-beg-time`', required=True)
    parser.add_argument('--time-end',    type=int, help='Ending of time. Hours after `--exp-beg-time`', required=True)
    parser.add_argument('--time-stride', type=int, help='Each plot differ by this time interval. Hours', required=True)
    parser.add_argument('--output-dir', type=str, help='Output filename in nc file.', required=True)
    parser.add_argument('--output-prefix', type=str, help='analysis beg time', required=True)
    parser.add_argument('--nproc', type=int, help="Number of processors", default=1)

    args = parser.parse_args()

    print(args)
 
    N = int((args.time_end - args.time_beg) / args.time_stride) + 1
    
    failed_dates = []
    input_args = []
   
    
    for varname in args.varnames:
        
        for i in range(N):
            
            target_time = pd.Timestamp(args.exp_beg_time) + pd.Timedelta(hours=args.time_beg + i * args.time_stride) 
            details = dict(
                varname     = varname,
                input_root  = args.input_root,
                expblob     = args.expblob,
                target_time = target_time,
                output_dir  = args.output_dir,
                output_prefix = args.output_prefix,
            )

            print("[Detect] Checking time=", target_time,)
            
            result = doJob(details, detect_phase=True)
            
            if not result['need_work']:
                print("File `%s` already exist. Skip it." % (str(result['output_files']),))
                continue
            
            input_args.append((details, False))

    with Pool(processes=args.nproc) as pool:

        results = pool.starmap(doJob, input_args)

        for i, result in enumerate(results):
            if result['status'] != 'OK':
                print(result)
                print('!!! Failed to generate output file %s.' % (str(result['output_file']),))
                failed_dates.append(result['details'])


    print("Tasks finished.")

    print("Failed output files: ")
    for i, failed_detail in enumerate(failed_dates):
        print("%d : " % (i+1), failed_detail)

    print("Done.")    

