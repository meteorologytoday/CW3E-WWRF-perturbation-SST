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

import PRISM_tools
import cmocean
from scipy import sparse
import WRF_ens_tools

import re

def doJob(details, detect_phase=False):

    # phase \in ['detect', 'work']
    result = dict(details = details, status="UNKNOWN", need_work=False, detect_phase=detect_phase)

    try:
        

        # This is the target variable, such as total rainfall
        target_dt = details["target_dt"]
        target_varname = details["target_varname"]
 
        # This lat-lon range applies to target
        target_lat_rng     = details["target_lat_rng"]
        target_lon_rng     = details["target_lon_rng"]
       
        # This is the sensitivity variable, such as initial moisture
        sens_dt = details["sens_dt"]
        sens_varname = details["sens_varname"]

        input_root  = Path(details["input_root"])
        expblob     = details["expblob"]
        
        output_dir     = Path(details["output_dir"]) 
        output_prefix   = details["output_prefix"] 
        
        output_file = output_dir / ("sensitivity_%s_%s_%s.nc" % (
            output_prefix,
            varname,
            target_dt.strftime("%Y-%m-%d_%H"),
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

        target_da = WRF_ens_tools.loadExpblob(expblob, target_varname, target_time, root=input_root).isel(time=0) 
        sens_da   = WRF_ens_tools.loadExpblob(expblob, sens_varname,   sens_time,   root=input_root).isel(time=0)
        
        # average target_da

        # do correlation       

        box_rng = (
            (lat >= target_lat_rng[0]) &
            (lat <  target_lat_rng[1]) &
            (lon >= target_lon_rng[0]) &
            (lon <  target_lon_rng[1])
        )

        target_da = target_da.where(box_rng).mean(
            dim=["lat", "lon"], skipna=True,
        )

        da_corr = xr.corr(target_da, sens_da, dim="ens",)
       
        print("correlation dataarray = ", da_corr) 
        print("Saving output: ", output_file)
        da_corr.to_netcdf(output_file)

        if output_file.exists():
            print("File `%s` is generated." % (str(output_file),))


        result['status'] = 'OK'

    except Exception as e:

        traceback.print_exc()
        result['status'] = 'ERROR'
        #traceback.print_stack()

        print(e)


    return result

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--input-root', type=str, help='Input directories.', required=True)
    parser.add_argument('--expblob', type=str, help='Input directories.', required=True)
    parser.add_argument('--target-varname', type=str, help='analysis beg time', required=True)
    parser.add_argument('--target-time', type=str, help='analysis beg time', required=True)
    parser.add_argument('--sens-varname', type=str, help='analysis beg time', required=True)
    parser.add_argument('--sens-time', type=str, help='analysis beg time', required=True)

    parser.add_argument('--target-lat-rng', type=float, nargs=2, help="Latitude range for box. It is used if `--corr-type` is 'remote'. ", default=[-90.0, 90.0])
    parser.add_argument('--target-lon-rng', type=float, nargs=2, help="Longitudinal range for box. It is used if `--corr-type` is 'remote'.", default=[0.0, 360.0])
 
    parser.add_argument('--output-dir', type=str, help='Output filename in nc file.', required=True)
    parser.add_argument('--output-prefix', type=str, help='analysis beg time', required=True)
    parser.add_argument('--nproc', type=int, help="Number of processors", default=1)

    args = parser.parse_args()

    print(args)
 
    details = dict(
        target_varname = target_varname,
        sens_varname   = sens_varname,
        target_dt      = pd.Timestamp(args.target_time),
        sens_dt        = pd.Timestamp(args.sens_time),
        input_root     = args.input_root,
        expblob        = args.expblob,
        output_dir     = args.output_dir,
        output_prefix  = args.output_prefix,
        target_lat_rng = args.target_lat_rng,
        target_lon_rng = args.target_lon_rng,
    )

    result = doJob(details, detect_phase=True)
    
    if not result['need_work']:
        print("File `%s` already exist. Skip it." % (str(result['output_file']),))
    else:
        doJob(details, detect_phase=False)
    
    print("Done.")    

