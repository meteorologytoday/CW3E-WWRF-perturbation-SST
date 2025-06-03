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
import regrid_tools

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


def doJob(details, detect_phase=False):

    # phase \in ['detect', 'work']
    result = dict(details = details, status="UNKNOWN", need_work=False, detect_phase=detect_phase)

    try:
        
        varname         = details["varname"]
        input_WRF_root  = Path(details["input_WRF_root"])
        expsubsets         = details["expsubsets"]
        exp_beg_time    = pd.Timestamp(details["exp_beg_time"])
        target_time     = pd.Timestamp(details["target_time"])
        output_root     = Path(details["output_root"]) 

        # Detecting
        
        output_file = output_root / ("%s.png" % target_time.strftime("%Y-%m-%dT%H:%M:%S"))

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

        # Load WRF       
        data = []
        for expsubset in expsubsets:
            print("Loading expsubset: ", expsubset)
            data.append(WRF_ens_tools.loadGroup(expsubset[0], expsubset[1], expsubset[2], "TTL_RAIN", target_time, root=input_WRF_root))
        
       
        PRISM_lat_idx = ds_regrid["PRISM_lat_idx"].to_numpy()
        PRISM_lon_idx = ds_regrid["PRISM_lon_idx"].to_numpy()
        
        WRF_lat_idx = ds_regrid["WRF_lat_idx"].to_numpy()
        WRF_lon_idx = ds_regrid["WRF_lon_idx"].to_numpy()

        lat_regrid = ds_regrid["lat_regrid"].to_numpy()   
        lon_regrid = ds_regrid["lon_regrid"].to_numpy()   
        
        PRISM_avg_info = constructAvgMtx(PRISM_lat_idx, PRISM_lon_idx, len(lat_regrid), len(lon_regrid))
       

        lead_days = (target_time - exp_beg_time) / pd.Timedelta(days=1) 
        
        if lead_days <= 0:
            raise Exception("Error: `--lead-days` has to be positive.")
        
        if lead_days % 1 != 0:
            raise Exception("Error: `--lead-days` has to be an integer.")
            

        PRISM_ds = PRISM_tools.loadDatasetWithTime(dt).isel(time=0)
        PRISM_clim = PRISM_tools.loadClimDatasetWithTime(beg_dt, end_dt=None, inclusive="both")


        
        PRISM_ds = loadDatasetWithTime(beg_dt, end_dt=None, inclusive="both"):


        data = dict(CRPS=[], mu=[], sig=[], obs=[], clim_CRPS=[], clim_mu=[], clim_sig=[]) 
        times = pd.date_range(time_beg, time_end, freq="D")




        for lead_day in range(lead_days):
     
            _beg_time = exp_beg_time
            _end_time = _beg_time + pd.Timedelta(days=lead_day)
             
            print("##### Doing time: ", _beg_time, " to ", _end_time)
             
            # Load PRISM

       
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






 
        result['status'] = 'OK'

    except Exception as e:

        result['status'] = 'ERROR'
        #traceback.print_stack()
        traceback.print_exc()
        print(e)


    return result


def work():

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




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--input-files', type=str, nargs="+", help='Input stat files.', required=True)
    parser.add_argument('--ttl-rain-time-rng', type=str, nargs=2, help='analysis beg time',    required=True)
    parser.add_argument('--no-display', action="store_true")

    parser.add_argument('--regrid-file', type=str, help="The regrid file.", required=True)
    parser.add_argument('--clim-year-range', type=int, nargs=2, help='Output filename in nc file.', default=[1991, 2021])
    parser.add_argument('--output', type=str, help='Output filename in nc file.', required=True)

    args = parser.parse_args()
    
    print(args)
    
    clim_years = list(range(args.clim_year_range[0], args.clim_year_range[1]+1))
   
    sim_data = [ xr.open_dataset(input_file) for input_file in args.input_files ]
 
    PRISM_ds = PRISM_tools.loadDatasetWithTime(dt).isel(time=0)
    PRISM_clim = PRISM_tools.loadClimDatasetWithTime(beg_dt, end_dt=None, inclusive="both")

    print("Loading regrid_file: ", regrid_file)        
    ds_regrid = xr.open_dataset(regrid_file)
    PRISM_lat_idx = ds_regrid["PRISM_lat_idx"].to_numpy()
    PRISM_lon_idx = ds_regrid["PRISM_lon_idx"].to_numpy()
    lat_regrid = ds_regrid["lat_regrid"].to_numpy()   
    lon_regrid = ds_regrid["lon_regrid"].to_numpy()   
    
    PRISM_avg_info = regrid_tools.constructAvgMtx(PRISM_lat_idx, PRISM_lon_idx, len(lat_regrid), len(lon_regrid))

    lead_days = (target_time - exp_beg_time) / pd.Timedelta(days=1) 
    
    if lead_days <= 0:
        raise Exception("Error: `--lead-days` has to be positive.")
    
    if lead_days % 1 != 0:
        raise Exception("Error: `--lead-days` has to be an integer.")
        

    PRISM_ds = PRISM_tools.loadDatasetWithTime(dt).isel(time=0)
    PRISM_clim = PRISM_tools.loadClimDatasetWithTime(beg_dt, end_dt=None, inclusive="both")


    
    PRISM_ds = loadDatasetWithTime(beg_dt, end_dt=None, inclusive="both"):


    data = dict(CRPS=[], mu=[], sig=[], obs=[], clim_CRPS=[], clim_mu=[], clim_sig=[]) 
    times = pd.date_range(time_beg, time_end, freq="D")




    for lead_day in range(lead_days):
 
        _beg_time = exp_beg_time
        _end_time = _beg_time + pd.Timedelta(days=lead_day)
         
        print("##### Doing time: ", _beg_time, " to ", _end_time)
         
        # Load PRISM

   
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


