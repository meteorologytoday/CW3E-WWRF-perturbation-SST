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

import cmocean


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
    
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--input-dirs', type=str, nargs="+", help='Input directories.', required=True)
    parser.add_argument('--corr-type', type=str, help='Type of correlation. (1) Local correlation. (2) Correlation with a particular box. ', required=True, choices=["local", "remote"])
    parser.add_argument('--no-display', action="store_true")

    parser.add_argument('--time-rng', type=int, nargs=2, help="Time range in hours after --exp-beg-time", required=True)
    parser.add_argument('--exp-beg-time', type=str, help='analysis beg time', required=True)
    parser.add_argument('--wrfout-data-interval', type=int, help='Time interval between each adjacent record in wrfout files in seconds.', required=True)
    parser.add_argument('--frames-per-wrfout-file', type=int, help='Number of frames in each wrfout file.', required=True)
    parser.add_argument('--wrfout-suffix', type=str, default="")

    parser.add_argument('--corr-varnames-1', type=str, nargs="+", help="The first  varname to correlate. Multiple varnames can be supplied.", required=True)
    parser.add_argument('--corr-varname-2', type=str, help="The second varname to correlate. If `--corr-type` is 'remote', this refers to the varname in the box to be correlated.", required=True)

    parser.add_argument('--corr-box-lat-rng', type=float, nargs=2, help="Latitude range for box. It is used if `--corr-type` is 'remote'. ", default=[-90.0, 90.0])
    parser.add_argument('--corr-box-lon-rng', type=float, nargs=2, help="Longitudinal range for box. It is used if `--corr-type` is 'remote'.", default=[0.0, 360.0])
    
    parser.add_argument('--output', type=str, help='Output filename in nc file.', required=True)

    args = parser.parse_args()

    print(args)

    if args.corr_type == "local":
        print("I will do local correlation. ")
    elif args.corr_type == "remote":
        print("I will do remote correlation. ")

    corr_varnames = [*args.corr_varnames_1, args.corr_varname_2]

    exp_beg_time = pd.Timestamp(args.exp_beg_time)
    wrfout_data_interval = pd.Timedelta(seconds=args.wrfout_data_interval)
    time_beg = exp_beg_time + pd.Timedelta(hours=args.time_rng[0])
    time_end = exp_beg_time + pd.Timedelta(hours=args.time_rng[1])

    wsm = wrf_load_helper.WRFSimMetadata(
        start_datetime  = exp_beg_time,
        data_interval   = wrfout_data_interval,
        frames_per_file = args.frames_per_wrfout_file,
    )
    
    # Loading     
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
       
    print("Obtained data: ")
    print(ds)
    
    print("Now do correlation.")
    corr_merge = []
    for corr_varname_1 in args.corr_varnames_1:
        da1 = ds[corr_varname_1]
        da2 = ds[args.corr_varname_2]
        da_corr = xr.corr(da1, da2, dim="ens").expand_dims(dim={'time' : [time_beg,]}, axis=0).rename("corr_%s_%s" % (corr_varname_1, args.corr_varname_2,))
        corr_merge.append(da_corr)

    corr_ds = xr.merge(corr_merge)

    if args.output != "":
        print("Saving output: ", args.output) 
        corr_ds.to_netcdf(args.output)


