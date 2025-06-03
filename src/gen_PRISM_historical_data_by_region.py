import traceback
import xarray as xr
import pandas as pd
import numpy as np
import argparse
import os
from pathlib import Path
import PRISM_tools

def getAvgData(dt, wgt):

    PRISM_filename = "/expanse/lustre/scratch/t2hsu/temp_project/data/PRISM_stable_4kmD2/PRISM_ppt_stable_4kmD2-{ymd:s}.nc".format(
       ymd = dt.strftime("%Y-%m-%d"),
    )

    ds = xr.open_dataset(PRISM_filename).isel(time=0)
    
    data = ds[verification_varname].to_numpy()
    data[np.isnan(data)] = 0.0

    avg_data = np.sum(data * wgt) / np.sum(wgt)

    return avg_data








if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--date-beg', type=str, help='Verification date range.', required=True)
    parser.add_argument('--test-days', type=int, help='Verification date range.', default=10)
    parser.add_argument('--region-file', type=str, help='Region file.', required=True)
    parser.add_argument('--output', type=str, help='Output filename in nc file.', required=True)
    parser.add_argument('--year-range', type=int, nargs=2, help='Output filename in nc file.', default=[1981, 2021])

    args = parser.parse_args()

    print(args)

    beg_date = pd.Timestamp(args.date_beg)
    end_date = pd.Timedelta(days=args.test_days) + beg_date

    ds_region = xr.open_dataset(args.region_file)    

    years = list(range(args.year_range[0], args.year_range[1]+1))
    
    verification_varnames = ["total_precipitation", ]
    

    output_data_clim = np.zeros((args.test_days, ds_region.dims["region"], len(verification_varnames), len(years)))
    output_data_obs  = np.zeros((args.test_days, ds_region.dims["region"], len(verification_varnames),))
    dts = pd.date_range(beg_date, end_date, freq="D", inclusive="left")
        
    #print("latitude: ", ds_region["latitude"].to_numpy())


    for idx_r, region in enumerate(ds_region.coords["region"]):
        print("Doing region: ", region)
        

        mask = ds_region["mask_PRISM"].sel(region=region).to_numpy()
        wgt  = np.cos(ds_region["lat_PRISM"].to_numpy() * np.pi/180.0)[:, None]
       
        wgt_valid = wgt * mask 
    

        print("SUM OF WGT_VALID: ", np.sum(wgt_valid))

        for idx_vv, verification_varname in enumerate(verification_varnames):
        
            print("Doing variable: ", verification_varname)

            for idx_d in range(args.test_days):
            
                print("Doing test day: ", idx_d)
               
                dt = beg_date + pd.Timedelta(days = idx_d)

                output_data_obs[idx_d, idx_r, idx_vv] = getAvgData(dt, wgt_valid)

                # Clim
                for idx_y, year in enumerate(years):
                    dt_clim = pd.Timestamp("%04d-%s" % (year, dt.strftime("%m-%d")))
                    output_data_clim[idx_d, idx_r, idx_vv, idx_y] = getAvgData(dt_clim, wgt_valid)

    output_ds = xr.Dataset(
        
        data_vars = dict(
            clim_data=(["time", "region", "variable", "year"], output_data_clim),
            obs_data=(["time", "region", "variable",], output_data_obs),
        ),
        
        coords=dict(
            time = dts,
            year = years,
            region = ds_region.coords["region"],
            variable = verification_varnames,
            reference_time="1970-01-01",
        ),
    )
            
    
    print("Output file: ", args.output)
    output_ds.to_netcdf(args.output)
         
    

