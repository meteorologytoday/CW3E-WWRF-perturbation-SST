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

import PRISM_tools
import cmocean
from scipy import sparse

def constructAvgMtx(lat_idx, lon_idx, nbox_lat, nbox_lon):
     
    Ny, Nx = lat_idx.shape
    original_total_grids = Ny * Nx
    total_boxes = nbox_lat * nbox_lon
    
    print("Ny, Nx = ", lat_idx.shape)
   
    # This is the numbering of each regridded boxes 
    regrid_row_idx = np.arange(total_boxes).reshape((nbox_lat, nbox_lon)) 

    # This is the numbering of the original grids
    original_grid_idx = np.arange(original_total_grids).reshape((Ny, Nx))
   
    print("shape of regrid_row_idx: ", regrid_row_idx.shape) 



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
    
    wgt_mtx = sparse.dia_array( ([wgt**(-1),], [0,]), shape=(total_boxes, total_boxes))
    avg_mtx = wgt_mtx @ avg_mtx 

    regrid_info = dict(
        avg_mtx = avg_mtx,
        shape_original = (Ny, Nx),
        shape_regrid = (nbox_lat, nbox_lon),
    )

    return regrid_info

def regrid(regrid_info, arr):
    
    flattened_arr = np.array(arr).flatten()

    if len(flattened_arr) != regrid_info["shape_original"][0] * regrid_info["shape_original"][1]:
        raise Exception("Dimension of input array does not match avg_info.")
    
    result = regrid_info["avg_mtx"] @ np.array(arr).flatten()
    result = np.reshape(result, regrid_info["shape_regrid"])

    return result 




def getCRPS(

    # WRF
    list_of_casedirs,
    filename,
    varname,
    
    # Observation
    da_obs,
):
    

    lat = da_obs.coords["lat"]
    lon = da_obs.coords["lon"]
    
    
    # Load WRF files
    # Interpolate them into observation grids


    # Compute means and standard deviations
    # Compute CRPS
    
    # output file or return dataset
    
    
    





def getWRFVariable(ds, *varnames):

    merge_data = []
    for varname in varnames:

        info = wrf_varname_info.wrf_varname_info[varname] if "varname" in wrf_varname_info.wrf_varname_info else dict()
        selector = info["selector"] if "selector" in info else None
        wrf_varname = info["wrf_varname"] if "wrf_varname" in info else varname

        da = ds[wrf_varname]

        if selector is not None:
            da      = da.isel(**selector)

        merge_data.append(da)

    return xr.merge(merge_data)

if __name__ == "__main__":

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
    parser.add_argument('--input-WRF-dirs', type=str, nargs="+", help='Input directories.', required=True)
    parser.add_argument('--no-display', action="store_true")
    parser.add_argument('--time-rng', type=str, nargs=2, help="Time range", required=True)
    parser.add_argument('--exp-beg-time', type=str, help='analysis beg time', required=True)
    parser.add_argument('--wrfout-data-interval', type=int, help='Time interval between each adjacent record in wrfout files in seconds.', required=True)
    parser.add_argument('--frames-per-wrfout-file', type=int, help='Number of frames in each wrfout file.', required=True)
    parser.add_argument('--wrfout-suffix', type=str, default="")
    parser.add_argument('--regrid-file', type=str, help="The regrid file.", required=True)
    
    parser.add_argument('--output', type=str, help='Output filename in nc file.', required=True)

    args = parser.parse_args()

    print(args)

    exp_beg_time = pd.Timestamp(args.exp_beg_time)
    wrfout_data_interval = pd.Timedelta(seconds=args.wrfout_data_interval)
    time_beg = pd.Timestamp(args.time_rng[0])
    time_end = pd.Timestamp(args.time_rng[1])

    wsm = wrf_load_helper.WRFSimMetadata(
        start_datetime  = exp_beg_time,
        data_interval   = wrfout_data_interval,
        frames_per_file = args.frames_per_wrfout_file,
    )
   
    ds_regrid = xr.open_dataset(xr.regrid_file)
    nlat_box = ds_regrid.attrs["nlat_box"]
    nlon_box = ds_regrid.attrs["nlon_box"]
   

    for dt in pd.date_range(time_beg, time_end, freq="D"):
        
        # Load PRISM
        PRISM_ds = PRISM_tools.loadDatasetWithTime(dt)
        
        # Regrid
        
        
        
        
    
        # Load WRF  
        data = [] 
        merge_data = []
        for i in range(len(args.input_dirs)):
            
            input_dir      = args.input_dirs[i] 
            print("Loading the %d-th wrf dir: %s" % (i, input_dir,))
            try:
                ds = wrf_load_helper.loadWRFDataFromDir(
                    wsm, 
                    input_dir,
                    beg_time = time_beg,
                    end_time = time_end,
                    suffix=args.wrfout_suffix,
                    avg=None,
                    verbose=False,
                    inclusive="both",
                    drop_latlon_time_dependency = True,
                )
                
                ds = xr.merge([
                    ds,
                    wrf_preprocess.genAnalysis(ds, wsm.data_interval, varnames=corr_varnames),
                ])
                
                
                ds = getWRFVariable(ds, *corr_varnames)

                avg_time_dims = []
                for dimname in ["time", "time_mid",]:
                    if dimname in ds.dims:
                        avg_time_dims.append(dimname) 

                ds = ds.mean(dim=avg_time_dims)
                ds = ds.expand_dims(dim={"ens": [i,]}, axis=0)
                
                data_to_merge = []
                if args.corr_type == "remote":
     
                    lat = ds.coords["XLAT"]
                    lon = ds.coords["XLONG"] % 360.0
                     
                    box_rng = (
                        (lat >= args.corr_box_lat_rng[0]) &
                        (lat <  args.corr_box_lat_rng[1]) &
                        (lon >= args.corr_box_lon_rng[0]) &
                        (lon <  args.corr_box_lon_rng[1])
                    )

                    avg_dims = []
                    for dimname in ["south_north", "south_north_stag", "west_east_stag", "west_east"]:
                        if dimname in ds.dims:
                            avg_dims.append(dimname) 
                     
                    da = ds[args.corr_varname_2].where(box_rng).mean(
                        dim=avg_dims, skipna=True,
                    )

                    data_to_merge.append(da)

                for varname in args.corr_varnames_1:
                    data_to_merge.append(ds[varname])

     
                ds = xr.merge(data_to_merge)

                merge_data.append(ds)
                
            except Exception as e:
                traceback.print_exc()
                print("Loading error. Skip this.")

        ds = xr.merge(merge_data)
        
        # Compute CPRS
        
        

    # Output data   
    print("Obtained data: ")
    print(ds)
    
    if args.output != "":
        print("Saving output: ", args.output) 
        corr_ds.to_netcdf(args.output)

    """
