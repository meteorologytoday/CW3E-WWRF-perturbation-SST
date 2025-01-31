import ERA5_tools

import traceback
import xarray as xr
import pandas as pd
import numpy as np
import argparse
import os
from pathlib import Path

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--date-beg', type=str, help='Verification date range.', required=True)
    parser.add_argument('--test-days', type=int, help='Verification date range.', default=10)
    parser.add_argument('--region-file', type=str, help='Region file.', required=True)
    parser.add_argument('--output', type=str, help='Output filename in nc file.', required=True)

    args = parser.parse_args()

    print(args)

    beg_date = pd.Timestamp(args.date_beg)
    end_date = pd.Timedelta(days=args.test_days) + beg_date

    ds_region = xr.open_dataset(args.region_file)    
    
    verification_varnames = ["total_precipitation", "convective_precipitation", "large_scale_precipitation"]
    

    output_data = np.zeros((args.test_days, ds_region.dims["region"], len(verification_varnames)))
    dts = pd.date_range(beg_date, end_date, freq="D", inclusive="left")
        
    print("latitude: ", ds_region["latitude"].to_numpy())


    for idx_r, region in enumerate(ds_region.coords["region"]):
        print("Doing region: ", region)
        

        mask = ds_region["mask_ERA5"].sel(region=region).to_numpy()
        wgt  = np.cos(ds_region["latitude"].to_numpy() * np.pi/180.0)[:, None]
       
        #wwget, _ = np.meshgrid(wgt, range(ds_region.coords["longitude"]), indexing='ij')
        valid_idx = mask == 1
        wgt_valid = wgt * mask 
        
        for idx_vv, verification_varname in enumerate(verification_varnames):
        
            print("Doing variable: ", verification_varname)

            shortname = ERA5_tools.mapping_longname_shortname[verification_varname]
     
            for idx_d in range(args.test_days):
            
                print("Doing test day: ", idx_d)
               
                dt = beg_date + pd.Timedelta(days = idx_d)
                era5_ds = ERA5_tools.open_dataset(verification_varname, dt, "24hr")
                
                era5_data = era5_ds[shortname].to_numpy()
                avg_data = np.sum(era5_data * wgt_valid) / np.sum(wgt_valid)

               
                 
                output_data[idx_d, idx_r, idx_vv] = avg_data



    output_ds = xr.Dataset(
        
        data_vars = dict(
            obs_data=(["time", "region", "variable"], output_data),
        ),
        
        coords=dict(
            time = dts,
            region = ds_region.coords["region"],
            variable = verification_varnames,
            reference_time="1970-01-01",
        ),
    )
            
    
    print("Output file: ", args.output)
    output_ds.to_netcdf(args.output)
         
    

