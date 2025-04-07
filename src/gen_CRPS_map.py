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


if __name__ == "__main__":

    """
    ds = xr.open_dataset("/home/t2hsu/temp_project/PROCESSED_CW3E_WRF_RUNS/0.08deg/exp_Baseline01/runs/Baseline01_ens00/output/wrfout/wrfout_d01_2023-01-10_06:00:00_temp").isel(Time=0)
    TTL = (ds["RAINC"] + ds["RAINNC"]).to_numpy()
    
    #ds_PRISM = PRISM_tools.loadDatasetWithTime("2023-01-05").isel(time=0)
    #TTL = ds_PRISM["total_precipitation"].to_numpy()
    
    ds_regrid = xr.open_dataset("gendata/regrid_idx.nc")
   
    print(ds_regrid) 
    #lat_idx = ds_regrid["PRISM_lat_idx"].to_numpy()
    #lon_idx = ds_regrid["PRISM_lon_idx"].to_numpy()
    lat_idx = ds_regrid["WRF_lat_idx"].to_numpy()
    lon_idx = ds_regrid["WRF_lon_idx"].to_numpy()


    lat_regrid_bnds = ds_regrid["lat_regrid_bnd"].to_numpy()   
    lon_regrid_bnds = ds_regrid["lon_regrid_bnd"].to_numpy()   

    lat_regrid = (lat_regrid_bnds[1:] + lat_regrid_bnds[:-1])/2
    lon_regrid = (lon_regrid_bnds[1:] + lon_regrid_bnds[:-1])/2
    
    regrid_shape = (len(lat_regrid), len(lon_regrid))
    
    avg_info = constructAvgMtx(lat_idx, lon_idx, len(lat_regrid), len(lon_regrid))

    print("Doing regrid")    
    TTL_regrid = regrid(avg_info, TTL)
    print("done")

    print(TTL_regrid.shape)
    new_ds = xr.Dataset(
        data_vars = dict(
            TTL_original = (["south_north", "west_east"], TTL),
            TTL_regrid   = (["lat_regrid", "lon_regrid"], TTL_regrid),
        ),
        coords = dict(
            lat_regrid = (["lat_regrid",], lat_regrid), 
            lon_regrid = (["lon_regrid",], lon_regrid), 
        ),
    )    

    #new_ds = new_ds.isel(south_north=slice(None, None, -1))
    new_ds.to_netcdf("regridded_TTL.nc") 
    """

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--input-WRF-root', type=str, help='Input directories.', required=True)
    parser.add_argument('--expname', type=str, help='Input directories.', required=True)
    parser.add_argument('--group', type=str, help='Input directories.', required=True)
    parser.add_argument('--subgroup', type=str, help='Input directories.', default="BLANK")
    parser.add_argument('--ens-ids', type=str, help="Ens ids. Comma separated and can use range like 1-3,5,23-25", required=True)

    parser.add_argument('--no-display', action="store_true")
    parser.add_argument('--time-beg', type=int, help="Time beg after --exp-beg-time", required=True)
    parser.add_argument('--lead-days', type=int, help="How many lead days" , required=True)
    parser.add_argument('--exp-beg-time', type=str, help='analysis beg time', required=True)
    parser.add_argument('--wrfout-data-interval', type=int, help='Time interval between each adjacent record in wrfout files in seconds.', required=True)
    parser.add_argument('--frames-per-wrfout-file', type=int, help='Number of frames in each wrfout file.', required=True)
    parser.add_argument('--wrfout-suffix', type=str, default="")
    parser.add_argument('--regrid-file', type=str, help="The regrid file.", required=True)
    
    #parser.add_argument('--clim-year-range', type=int, nargs=2, help='Output filename in nc file.', default=[1981, 2021])
    parser.add_argument('--clim-year-range', type=int, nargs=2, help='Output filename in nc file.', default=[1991, 2021])
    
    parser.add_argument('--output', type=str, help='Output filename in nc file.', required=True)

    args = parser.parse_args()

    print(args)
    
    clim_years = list(range(args.clim_year_range[0], args.clim_year_range[1]+1))

    ens_ids = parse_ranges(args.ens_ids)
    print("Ensemble ids: %s => %s" % (args.ens_ids, ",".join(["%d" % i for i in ens_ids] ) ))

    ens_N = len(ens_ids)

    # Constructing what to load from WRF   
    ens_id_to_idx_mapping = dict() 
    input_WRF_root = Path(args.input_WRF_root)
    input_WRF_dirs = []

    for i, ens_id in enumerate(ens_ids):
        ens_id_to_idx_mapping[ens_id] = i
        input_WRF_dir = WRF_ens_tools.genWRFEnsRelPathDir(
            group = args.group,
            subgroup = args.subgroup,
            ens_id = ens_id,
            style = "v1",
            root = input_WRF_root 
        )
        
        input_WRF_dirs.append(input_WRF_dir)


    ds_regrid = xr.open_dataset("gendata/regrid_idx.nc")
   
    PRISM_lat_idx = ds_regrid["PRISM_lat_idx"].to_numpy()
    PRISM_lon_idx = ds_regrid["PRISM_lon_idx"].to_numpy()
    
    WRF_lat_idx = ds_regrid["WRF_lat_idx"].to_numpy()
    WRF_lon_idx = ds_regrid["WRF_lon_idx"].to_numpy()

    lat_regrid = ds_regrid["lat_regrid"].to_numpy()   
    lon_regrid = ds_regrid["lon_regrid"].to_numpy()   

    WRF_avg_info   = constructAvgMtx(WRF_lat_idx,   WRF_lon_idx,   len(lat_regrid), len(lon_regrid))
    PRISM_avg_info = constructAvgMtx(PRISM_lat_idx, PRISM_lon_idx, len(lat_regrid), len(lon_regrid))

    exp_beg_time = pd.Timestamp(args.exp_beg_time)
    wrfout_data_interval = pd.Timedelta(seconds=args.wrfout_data_interval)
    time_beg = exp_beg_time
    time_end = time_beg + pd.Timedelta(days=args.lead_days-1)

    if args.lead_days <= 0:
        raise Exception("Error: `--lead-days` has to be a positive integer.")

    wsm = wrf_load_helper.WRFSimMetadata(
        start_datetime  = exp_beg_time,
        data_interval   = wrfout_data_interval,
        frames_per_file = args.frames_per_wrfout_file,
    )

    data = dict(CRPS=[], mu=[], sig=[], obs=[], clim_CRPS=[], clim_mu=[], clim_sig=[]) 
    times = pd.date_range(time_beg, time_end, freq="D")
    for dt in times:
 
        _beg_time = dt
        _end_time = dt + pd.Timedelta(days=1)
      
        print("##### Doing time: ", _beg_time, " to ", _end_time) 
        # Load PRISM
        PRISM_ds = PRISM_tools.loadDatasetWithTime(dt).isel(time=0)
    
        # Regrid
        data_PRISM = regrid(PRISM_avg_info, PRISM_ds["total_precipitation"].to_numpy())
   
        # PRISM clim
        PRISM_clim = []
        for idx_y, year in enumerate(clim_years):
                
            center_dt = pd.Timestamp("%04d-%s" % (year, dt.strftime("%m-%d"))) 

            for day_shift in range(-5, 6):
                
                dt_used = center_dt + pd.Timedelta(days=day_shift)
                print("Loading clim files: ", dt_used)
                _ds = PRISM_tools.loadDatasetWithTime(dt_used).isel(time=0)
                PRISM_clim.append(regrid(PRISM_avg_info, _ds["total_precipitation"].to_numpy()))

        PRISM_clim = np.stack(PRISM_clim, axis=0)
        print("Shape of PRISM_clim = ", PRISM_clim.shape)

        clim_mu = np.mean(PRISM_clim, axis=0)
        clim_sig = np.std(PRISM_clim, axis=0, ddof=1)
        clim_sig[clim_sig==0] = np.nan

        data["clim_mu"].append(clim_mu)
        data["clim_sig"].append(clim_sig)
        data["clim_CRPS"].append(CRPS_tools.CRPS_gaussian(clim_mu, clim_sig, data_PRISM))

        # Load WRF  
        data_WRF = [] 


        for i, input_WRF_dir in enumerate(input_WRF_dirs):
             
            print("Loading the %d-th wrf dir: %s" % (i, input_WRF_dir,))
            try:
                ds = wrf_load_helper.loadWRFDataFromDir(
                    wsm, 
                    input_WRF_dir,
                    beg_time = _beg_time,
                    end_time = _end_time,
                    suffix=args.wrfout_suffix,
                    avg=None,
                    verbose=False,
                    inclusive="both",
                    drop_latlon_time_dependency = True,
                ).sel(time=[_beg_time, _end_time,])
               
                TTL_PRECIP = (ds["RAINC"] + ds["RAINNC"]).to_numpy()
                TTL_PRECIP = TTL_PRECIP[1, :, :] - TTL_PRECIP[0, :, :]
                TTL_PRECIP_regrid = regrid(WRF_avg_info, TTL_PRECIP)

                data_WRF.append(TTL_PRECIP_regrid) 

            except Exception as e:
                
                traceback.print_exc()
                print("Loading error. Skip this.")

        data_WRF = np.stack(data_WRF, axis=0)
        
        # Compute CRPS
        mu  = np.mean(data_WRF, axis=0) 
        sig = np.std(data_WRF, axis=0, ddof=1)
        sig[sig==0] = np.nan
        data["CRPS"].append(CRPS_tools.CRPS_gaussian(mu, sig, data_PRISM))
        data["mu"].append(mu)
        data["sig"].append(sig)
        data["obs"].append(data_PRISM)



    print("Stacking outcome...")
    CRPS_data = np.stack(data["CRPS"], axis=0) 
    mu_data = np.stack(data["mu"], axis=0) 
    sig_data = np.stack(data["sig"], axis=0) 
    obs_data = np.stack(data["obs"], axis=0) 
    clim_mu_data = np.stack(data["clim_mu"], axis=0)
    clim_sig_data = np.stack(data["clim_sig"], axis=0)
    clim_CRPS_data = np.stack(data["clim_CRPS"], axis=0)

    print("Making output dataset...")
    ds_output = xr.Dataset(
        data_vars = dict(
            CRPS = (["time", "lat", "lon"], CRPS_data),
            mu = (["time", "lat", "lon"], mu_data),
            sig = (["time", "lat", "lon"], sig_data),
            obs = (["time", "lat", "lon"], obs_data),
            clim_mu   = (["time", "lat", "lon"], clim_mu_data),
            clim_sig  = (["time", "lat", "lon"], clim_sig_data),
            clim_CRPS = (["time", "lat", "lon"], clim_CRPS_data),
        ),
        coords = dict(
            time = (["time"], times),
            lat = (["lat"], lat_regrid),
            lon = (["lon"], lon_regrid),
        ),
        attrs = dict(
            ens_N = ens_N,
        )
    )

    print("Saving output: ", args.output)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    ds_output.to_netcdf(args.output)

