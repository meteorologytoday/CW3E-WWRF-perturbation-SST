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

def parseVarname(varname_long):
    
    m = re.match(r'(?P<varname>\w+)(::(?P<level>[0-9]+))?', varname_long)

    varname = None
    level = None
    if m:
        
        d = m.groupdict()
        print("GROUPDICT: ", d)
        varname = d["varname"]
        level = d["level"]
        if level is not None:
            level = int(level)
        
    else:
        raise Exception("Varname %s cannot be parsed. ") 

    return varname, level

def parse_ranges(input_str):
    numbers = []
    for part in input_str.split(','):
        part = part.strip()
        if '-' in part:
            start, end = map(int, part.split('-'))
            numbers.extend(range(start, end + 1))
        else:
            numbers.append(int(part))
    return numbers


def constructAvgMtx(lat_idx, lon_idx, nbox_lat, nbox_lon):
     
    Ny, Nx = lat_idx.shape
    original_total_grids = Ny * Nx
    total_boxes = nbox_lat * nbox_lon
    
    #print("Ny, Nx = ", lat_idx.shape)
   
    # This is the numbering of each regridded boxes 
    regrid_row_idx = np.arange(total_boxes).reshape((nbox_lat, nbox_lon)) 

    # This is the numbering of the original grids
    original_grid_idx = np.arange(original_total_grids).reshape((Ny, Nx))
   
    #print("shape of regrid_row_idx: ", regrid_row_idx.shape) 



    row_idxes = []
    col_idxes = []
    for i in range(Nx):
        for j in range(Ny):
 
            _lon_idx = lon_idx[j, i]
            _lat_idx = lat_idx[j, i]
          
            if _lon_idx >= 0 and _lat_idx >= 0:
 
                _row_idx = regrid_row_idx[ _lat_idx, _lon_idx]
                _col_idx = original_grid_idx[j, i]

                row_idxes.append(_row_idx)
                col_idxes.append(_col_idx)
                 
    vals = np.ones((len(row_idxes), ))
    

    avg_mtx = sparse.coo_array((vals, (row_idxes, col_idxes)), shape=(total_boxes, original_total_grids), dtype=np.float32)
    
    wgt = avg_mtx.sum(axis=1)

    mask = np.zeros((nbox_lat, nbox_lon), dtype=np.int32)
    mask[np.reshape(wgt, (nbox_lat, nbox_lon)) != 0] = 1
    
    wgt_mtx = sparse.dia_array( ([wgt**(-1),], [0,]), shape=(total_boxes, total_boxes))
    avg_mtx = wgt_mtx @ avg_mtx 

    regrid_info = dict(
        avg_mtx = avg_mtx,
        shape_original = (Ny, Nx),
        shape_regrid = (nbox_lat, nbox_lon),
        mask = mask,
    )

    return regrid_info

def regrid(regrid_info, arr):
   
    if len(arr.shape) == 3:
        
        print("The input array is three dimension. Treating the first dimenion as the time")
        result = [
            regrid(regrid_info, arr[i, :, :])
            for i in range(arr.shape[0])
        ]

        result = np.stack(result, axis=0)
        
        return result
        
        
    flattened_arr = np.array(arr).flatten()

    if len(flattened_arr) != regrid_info["shape_original"][0] * regrid_info["shape_original"][1]:
        print(regrid_info["shape_original"], "; ", arr.shape)
        raise Exception("Dimension of input array does not match avg_info.")
    
    result = regrid_info["avg_mtx"] @ np.array(arr).flatten()
    result = np.reshape(result, regrid_info["shape_regrid"])
    result[regrid_info['mask'] == 0] = np.nan

    return result 



def doJob(details, detect_phase=False):

    # phase \in ['detect', 'work']
    result = dict(details = details, status="UNKNOWN", need_work=False, detect_phase=detect_phase)

    try:
        
        varname         = details["varname"]
        level           = details["level"]
        input_WRF_root  = Path(details["input_WRF_root"])
        expname         = details["expname"]
        group           = details["group"]
        ens_ids         = details["ens_ids"]
        target_hour     = details["target_hour"]
        exp_beg_time    = details["exp_beg_time"]
        wrfout_data_interval = details["wrfout_data_interval"]
        frames_per_wrfout_file = details["frames_per_wrfout_file"]
        wrfout_suffix   = details["wrfout_suffix"]
        regrid_file     = Path(details["regrid_file"])
        output_root      = Path(details["output_root"]) 

        exp_beg_time = pd.Timestamp(exp_beg_time)
        wrfout_data_interval = pd.Timedelta(seconds=wrfout_data_interval)
        target_time = exp_beg_time + pd.Timedelta(hours=target_hour)

        # Detecting
        output_file = WRF_ens_tools.genEnsStatFilename(expname, group, varname, target_time, root=output_root, level=level)
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
        WRF_lat_idx = ds_regrid["WRF_lat_idx"].to_numpy()
        WRF_lon_idx = ds_regrid["WRF_lon_idx"].to_numpy()

        lat_regrid = ds_regrid["lat_regrid"].to_numpy()   
        lon_regrid = ds_regrid["lon_regrid"].to_numpy()   
    
        WRF_avg_info   = constructAvgMtx(WRF_lat_idx,   WRF_lon_idx,   len(lat_regrid), len(lon_regrid))


        wsm = wrf_load_helper.WRFSimMetadata(
            start_datetime  = exp_beg_time,
            data_interval   = wrfout_data_interval,
            frames_per_file = frames_per_wrfout_file,
        )

        output_data = dict(mu=[], sig=[])
        
        print("##### Doing time: ", target_time)
 
        # Constructing what to load from WRF   
        ens_id_to_idx_mapping = dict() 
        input_WRF_root = Path(input_WRF_root)
        input_WRF_dirs = []

        for i, ens_id in enumerate(ens_ids):
            ens_id_to_idx_mapping[ens_id] = i
            input_WRF_dir = WRF_ens_tools.genWRFEnsRelPathDir(
                expname = expname,
                group = group,
                ens_id = ens_id,
                root = input_WRF_root 
            )
            
            input_WRF_dirs.append(input_WRF_dir)

        # Load WRF  
        data_WRF = [] 

        for i, input_WRF_dir in enumerate(input_WRF_dirs):
             
            ens_id = ens_ids[i]
            print("Loading the %d-th wrf dir: %s, ens_id=%d" % (i, input_WRF_dir, ens_id))

            ds = wrf_load_helper.loadWRFDataFromDir(
                wsm, 
                input_WRF_dir,
                beg_time = target_time,
                end_time = target_time,
                suffix=args.wrfout_suffix,
                avg=None,
                verbose=False,
                inclusive="both",
                drop_latlon_time_dependency = True,
            )

            if level is not None:
                ds = ds.sel(pressure=level)

            if varname == "WND":
                #U_U = ds["U"].to_numpy()
                #V_V = ds["V"].to_numpy()
                
                #U = xr.zeros_like(ds["PH"])
                #V = xr.zeros_like(ds["PH"])
                
                #U.data[:, :, :] = (U_U[:, :, 1:] + U_U[:, :, :-1]) / 2
                #V.data[:, :, :] = (V_V[:, :, 1:] + V_V[:, :, :-1]) / 2
                
                
                da = ((ds["U"]**2 + ds["V"]**2)**0.5).rename("WND")

            elif varname == "TTL_RAIN":
                da = ds["RAINNC"] + ds["RAINC"]
                da = da.rename("TTL_RAIN")
            else:
                da = ds[varname]
            
            
            _data = da.to_numpy()
            #print(_data.shape)

            regrid_data = regrid(WRF_avg_info, _data)
            data_WRF.append(regrid_data)
        #print(data_WRF)
        data_WRF = np.stack(data_WRF, axis=0)

        print("Making output dataset...")
        da = xr.DataArray(
            data = data_WRF,
            dims = ["ens", "time", "lat", "lon"],
            coords = dict(
                ens = (["ens"], ens_ids),
                time = (["time"], [target_time,]),
                lat = (["lat"], lat_regrid),
                lon = (["lon"], lon_regrid),
            ),
            attrs = dict(
                ens_N = len(ens_ids),
                ens_ids = ens_ids,
            )
        )
        
        merge_data = []
        merge_data.append(da.std(dim="ens").expand_dims(dim={"stat": ["std",]}, axis=1).rename(varname))
        merge_data.append(da.mean(dim="ens").expand_dims(dim={"stat": ["mean",]}, axis=1).rename(varname))
        merge_data.append(da.count(dim="ens").expand_dims(dim={"stat": ["count",]}, axis=1).rename(varname))

        ds_WRF = xr.merge(merge_data)
        #print(ds_WRF)
        print("Saving output: ", output_file)
        ds_WRF.to_netcdf(output_file, unlimited_dims="time")

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
    parser.add_argument('--wrfout-suffix', type=str, default="")
    parser.add_argument('--regrid-file', type=str, help="The regrid file.", required=True)
    
    parser.add_argument('--output-root', type=str, help='Output filename in nc file.', required=True)
    parser.add_argument('--nproc', type=int, help="Number of processors", default=1)

    args = parser.parse_args()

    print(args)
 
    if args.output_time_range[1] < args.output_time_range[0]:
        raise Exception("Error: negative output time range. Please check `--output-time-range`.")
   
    ens_ids = parse_ranges(args.ens_ids)
    print("Ensemble ids: %s => %s" % (args.ens_ids, ",".join(["%d" % i for i in ens_ids] ) ))

    stepping_hour = int(args.wrfout_data_interval / 3600)
    print("Stepping hour: %d" % (stepping_hour,))

    

    failed_dates = []
    input_args = []

    for long_varname in args.varnames:
        
        varname, level = parseVarname(long_varname)
        
        print("Full varname: %s" % (long_varname,))
        print("Parsed varname: %s" % (varname,))
        print("Parsed level: %s" % ( "%d" % level if level is not None else "N/A") )
        
        for target_hour in range(args.output_time_range[0], args.output_time_range[1]+1, stepping_hour):
        
            details = dict(
                varname = varname,
                level   = level,
                input_WRF_root = args.input_WRF_root,
                expname = args.expname,
                group = args.group,
                ens_ids = ens_ids,
                target_hour = target_hour,
                exp_beg_time = args.exp_beg_time,
                wrfout_data_interval = args.wrfout_data_interval,
                frames_per_wrfout_file = args.frames_per_wrfout_file,
                wrfout_suffix = args.wrfout_suffix,
                regrid_file = args.regrid_file,
                output_root = args.output_root,
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
