import traceback
import xarray as xr
import pandas as pd
import numpy as np
import argparse

import wrf_load_helper 
import wrf_preprocess
import wrf_varname_info
from pathlib import Path

import PRISM_tools
from scipy import sparse
import regrid_tools

def doJob(details, detect_phase=False):
    
    # phase \in ['detect', 'work']
    result = dict(details = details, status="UNKNOWN", need_work=False, detect_phase=detect_phase)
    
    try:
        
        time_beg = pd.Timestamp(details["time_beg"])
        time_end = pd.Timestamp(details["time_end"])
        clim_years = details["clim_years"]
        regrid_file = details["regrid_file"]
        output_root = Path(details["output_root"]) 
        
        # Detecting
        output_file = output_root / ("PRISM_%s_%s.nc" % (
            time_beg.strftime("%Y-%m-%d"),
            time_end.strftime("%Y-%m-%d"),
        ))
        
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

        print("Loading regrid_file: ", regrid_file)        
        ds_regrid = xr.open_dataset(regrid_file)

        PRISM_lat_idx = ds_regrid["PRISM_lat_idx"].to_numpy()
        PRISM_lon_idx = ds_regrid["PRISM_lon_idx"].to_numpy()
        
        lat_regrid = ds_regrid["lat_regrid"].to_numpy()   
        lon_regrid = ds_regrid["lon_regrid"].to_numpy()   
        
        PRISM_avg_info = regrid_tools.constructAvgMtx(PRISM_lat_idx, PRISM_lon_idx, len(lat_regrid), len(lon_regrid))

        days_spanned = (time_end - time_beg) / pd.Timedelta(days=1) 
        
        if days_spanned <= 0:
            raise Exception("Error: `--lead-days` has to be positive.")
        
        if days_spanned % 1 != 0:
            raise Exception("Error: `--lead-days` has to be an integer.")
            

        PRISM_obs   = []
        PRISM_clim  = []

        # PRISM obs
        for lead_day in range(int(days_spanned)):
     
            dt =  time_beg + pd.Timedelta(days=lead_day)
            print("Loading for obs: ", dt) 
             
            # Load PRISM
            PRISM_ds = PRISM_tools.loadDatasetWithTime(dt).isel(time=0)
        
            # Regrid
            PRISM_obs.append(regrid_tools.regrid(PRISM_avg_info, PRISM_ds["total_precipitation"].to_numpy()))
       
        # Compute acc rainfall 
        PRISM_obs = np.stack(PRISM_obs, axis=0).sum(axis=0, keepdims=True)
        
        # PRISM clim
        for idx_y, year in enumerate(clim_years):

            tmp = []
            for lead_day in range(int(days_spanned)):
            
                dt = time_beg + pd.Timedelta(days=lead_day)
                clim_dt = pd.Timestamp("%04d-%s" % (year, dt.strftime("%m-%d"))) 
                print("Loading for clim: ", clim_dt) 
               
                _ds = PRISM_tools.loadDatasetWithTime(clim_dt).isel(time=0)
                tmp.append(regrid_tools.regrid(PRISM_avg_info, _ds["total_precipitation"].to_numpy()))
            
            PRISM_clim.append(np.stack(tmp, axis=0).sum(axis=0))
        
        
        PRISM_clim = np.stack(PRISM_clim, axis=0)        
        print("Shape of PRISM_clim = ", PRISM_clim.shape)
        
        clim_mu  = np.mean(PRISM_clim, axis=0, keepdims=True)
        clim_sig = np.std(PRISM_clim, axis=0, ddof=1, keepdims=True)
        clim_sig[clim_sig==0] = np.nan
        
        data_vars = dict(
            obs      = ( ["time", "lat", "lon"] , PRISM_obs),
            clim_mu  = ( ["time", "lat", "lon"] , clim_mu),
            clim_sig = ( ["time", "lat", "lon"] , clim_sig),
        )
       
        print(data_vars["obs"][1].shape)
        print(data_vars["clim_mu"][1].shape)
        print(data_vars["clim_sig"][1].shape)
 
        print("Making output dataset...")
        ds_output = xr.Dataset(
            data_vars = data_vars,
            coords = dict(
                time = (["time"], [time_beg,]),
                lat = (["lat"], lat_regrid),
                lon = (["lon"], lon_regrid),
            ),
#            attrs = dict(
#                clim_years = len(clim_years),
#            )
        )
        
        print("Saving output: ", output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        ds_output.to_netcdf(output_file)
 
        result['status'] = 'OK'

    except Exception as e:

        result['status'] = 'ERROR'
        traceback.print_exc()
        print(e)


    return result


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--time-beg', type=str, help='Begin accumulation', required=True)
    parser.add_argument('--time-end', type=str, help='End accumulation', required=True)
    parser.add_argument('--regrid-file', type=str, help="The regrid file.", required=True)
    parser.add_argument('--clim-year-range', type=int, nargs=2, help='Output filename in nc file.', default=[1991, 2021])
    parser.add_argument('--output-root', type=str, help='Output dir.', required=True)

    args = parser.parse_args()

    print(args)
    
    clim_years = list(range(args.clim_year_range[0], args.clim_year_range[1]+1))
    details = dict(
        time_beg = args.time_beg,
        time_end = args.time_end,
        output_root = args.output_root,
        regrid_file = args.regrid_file,
        clim_years = clim_years,
    )

    doJob(details)

    print("Done.")    
