from multiprocessing import Pool
import multiprocessing

import traceback
import xarray as xr
import pandas as pd
import numpy as np
import argparse
import wrf_load_helper 
import datetime
from pathlib import Path

from scipy import sparse
import WRF_ens_tools
import regrid_tools
import re


def doJob(details, detect_phase=False):

    # phase \in ['detect', 'work']
    result = dict(details = details, status="UNKNOWN", need_work=False, detect_phase=detect_phase)

    try:
        
        varname_level   = details["varname_level"]
        varname         = details["varname"]
        level           = details["level"]
        input_WRF_root  = Path(details["input_WRF_root"])
        expname         = details["expname"]
        group           = details["group"]
        ens_id          = details["ens_id"]
        target_hour     = details["target_hour"]
        exp_beg_time    = details["exp_beg_time"]
        wrfout_data_interval = details["wrfout_data_interval"]
        frames_per_wrfout_file = details["frames_per_wrfout_file"]
        wrfout_prefix   = details["wrfout_prefix"]
        wrfout_suffix   = details["wrfout_suffix"]
        regrid_file     = Path(details["regrid_file"])
        output_root     = Path(details["output_root"]) 
        input_style     = details["input_style"]

        exp_beg_time = pd.Timestamp(exp_beg_time)
        wrfout_data_interval = pd.Timedelta(seconds=wrfout_data_interval)
        target_time = exp_beg_time + pd.Timedelta(hours=target_hour)

        # Detecting
        output_file = WRF_ens_tools.genEnsFilename(expname, group, ens_id, varname_level, target_time, root=output_root)
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

        output_file.parent.mkdir(parents=True, exist_ok=True)

        ds_regrid = xr.open_dataset(regrid_file)
        WRF_lat_idx = ds_regrid["lat_idx"].to_numpy()
        WRF_lon_idx = ds_regrid["lon_idx"].to_numpy()

        lat_regrid = ds_regrid["lat_regrid"].to_numpy()   
        lon_regrid = ds_regrid["lon_regrid"].to_numpy()   
    
        WRF_avg_info   = regrid_tools.constructAvgMtx(WRF_lat_idx,   WRF_lon_idx,   len(lat_regrid), len(lon_regrid))


        wsm = wrf_load_helper.WRFSimMetadata(
            start_datetime  = exp_beg_time,
            data_interval   = wrfout_data_interval,
            frames_per_file = frames_per_wrfout_file,
        )

        print("##### Doing time: ", target_time)
 
        # Constructing what to load from WRF   
        input_WRF_root = Path(input_WRF_root)

        input_WRF_dir = WRF_ens_tools.genWRFEnsRelPathDir(
            expname = expname,
            group = group,
            ens_id = ens_id,
            root = input_WRF_root,
            filename_style = input_style, 
        )
            
        if input_style == "default":
            time_fmt=wrf_load_helper.wrfout_time_fmt
        elif input_style == "CW3E-WestWRF":
            time_fmt="%Y-%m-%d_%H_%M_%S"
        
        # Load WRF  
        print("Loading wrf dir: %s, ens_id=%d" % (input_WRF_dir, ens_id))

        ds = wrf_load_helper.loadWRFDataFromDir(
            wsm, 
            input_WRF_dir,
            beg_time = target_time,
            end_time = target_time,
            prefix=args.wrfout_prefix,
            suffix=args.wrfout_suffix,
            avg=None,
            verbose=False,
            inclusive="both",
            drop_latlon_time_dependency = True,
            time_fmt=time_fmt,
        )

        if level is not None:
            ds = ds.sel(pressure=[level,])

        if varname == "WND":
            da = ((ds["U"]**2 + ds["V"]**2)**0.5).rename("WND")

        elif varname == "TTL_RAIN":
            if input_style == "default":
                da = ds["RAINNC"] + ds["RAINC"]
            elif input_style == "CW3E-WestWRF":
                da = ds["precip"]
            da = da.rename("TTL_RAIN")
        elif varname == "PSFC":
            if input_style == "default":
                da = ds[varname]
            elif input_style == "CW3E-WestWRF":
                da = ds["slp"].rename("PSFC") * 1e2

        elif varname == "TOmA":
            da = (ds["SST"] - ds["T2"]).rename("TOmA")
        else:
            da = ds[varname]
        
        has_level = "pressure" in da.coords
        
        

        _data = da.isel(time=0).to_numpy()
        
        regrid_data = regrid_tools.regrid(WRF_avg_info, _data)

        regrid_data = np.expand_dims(regrid_data, axis=0)  # recover time dimension
        regrid_data = np.expand_dims(regrid_data, axis=0)  # add ens dimension
        
        print("Making output dataset...")

        output_coords = dict(
            ens = (["ens"], [ens_id,]),
            time = (["time"], [target_time,]),
            lat = (["lat"], lat_regrid),
            lon = (["lon"], lon_regrid),
        )

        if has_level:
            output_dims = ["ens", "time", "pressure", "lat", "lon"]
            output_coords["pressure"] = (["pressure",], da.coords["pressure"].to_numpy())
        else:
            output_dims = ["ens", "time", "lat", "lon"]
        
        
        da = xr.DataArray(
            name = varname_level,
            data = regrid_data,
            dims = output_dims,
            coords = output_coords,
            attrs = dict(
            )
        )
        
        print("Saving output: ", output_file)
        da.to_netcdf(output_file, unlimited_dims="time", encoding={'time':{'units':'hours since 1970-01-01'}})

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
    parser.add_argument('--input-WRF-root', type=str, help='Input directories.', required=True)
    parser.add_argument('--expname', type=str, help='Input directories.', required=True)
    parser.add_argument('--group', type=str, help='Input directories.', required=True)
    parser.add_argument('--ens-ids', type=str, help="Ens ids. Comma separated and can use range like 1-3,5,23-25", required=True)
    
    parser.add_argument('--varnames', type=str, nargs="+", help='Variables processed. If only a particular layer is selected, use `-` to separate the specified layer. Example: PH-850', required=True)


    parser.add_argument('--no-display', action="store_true")
    parser.add_argument('--output-time-range', type=int, nargs=2, help="Time beg after --exp-beg-time", required=True)
    parser.add_argument('--exp-beg-time', type=str, help='analysis beg time', required=True)
    parser.add_argument('--wrfout-data-interval', type=int, help='Time interval between each adjacent record in wrfout files in seconds.', required=True)
    parser.add_argument('--frames-per-wrfout-file', type=int, help='Number of frames in each wrfout file.', required=True)
    parser.add_argument('--wrfout-prefix', type=str, default="wrfout_d01_")
    parser.add_argument('--wrfout-suffix', type=str, default="")
    parser.add_argument('--regrid-file', type=str, help="The regrid file.", required=True)
    parser.add_argument('--input-style', type=str, default="default")
    
    parser.add_argument('--output-root', type=str, help='Output filename in nc file.', required=True)
    parser.add_argument('--nproc', type=int, help="Number of processors", default=1)

    args = parser.parse_args()

    print(args)
 
    if args.output_time_range[1] < args.output_time_range[0]:
        raise Exception("Error: negative output time range. Please check `--output-time-range`.")
   
    ens_ids = WRF_ens_tools.parseRanges(args.ens_ids)
    print("Ensemble ids: %s => %s" % (args.ens_ids, ",".join(["%d" % i for i in ens_ids] ) ))

    stepping_hour = int(args.wrfout_data_interval / 3600)
    print("Stepping hour: %d" % (stepping_hour,))

    

    failed_dates = []
    input_args = []

    for long_varname in args.varnames:
        
        varname, level = WRF_ens_tools.parseVarname(long_varname)
        
        print("Full varname: %s" % (long_varname,))
        print("Parsed varname: %s" % (varname,))
        print("Parsed level: %s" % ( "%d" % level if level is not None else "N/A") )
        
        for target_hour in range(args.output_time_range[0], args.output_time_range[1]+1, stepping_hour):
            for ens_id in ens_ids:  
                details = dict(
                    varname_level = long_varname,
                    varname = varname,
                    level   = level,
                    input_WRF_root = args.input_WRF_root,
                    expname = args.expname,
                    group = args.group,
                    ens_id = ens_id,
                    target_hour = target_hour,
                    exp_beg_time = args.exp_beg_time,
                    wrfout_data_interval = args.wrfout_data_interval,
                    frames_per_wrfout_file = args.frames_per_wrfout_file,
                    wrfout_prefix = args.wrfout_prefix,
                    wrfout_suffix = args.wrfout_suffix,
                    regrid_file = args.regrid_file,
                    output_root = args.output_root,
                    input_style = args.input_style,
                )

                print("[Detect] Checking %s (hour=%d)" % (varname, target_hour,))
                
                result = doJob(details, detect_phase=True)
                
                if not result['need_work']:
                    print("File `%s` already exist. Skip it." % (result['output_file'],))
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
