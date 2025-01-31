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
    parser.add_argument('--year-range', type=int, nargs=2, help='Output filename in nc file.', default=[1981, 2014])

    args = parser.parse_args()

    print(args)

    beg_date = pd.Timestamp(args.date_beg)
    end_date = pd.Timedelta(days=args.test_days) + beg_date

    ds_region = xr.open_dataset(args.region_file)    

    years = list(range(args.year_range[0], args.year_range[1]+1))
    
    verification_varnames = ["total_precipitation", ]
    

    output_data = np.zeros((args.test_days, ds_region.dims["region"], len(verification_varnames), len(years)))
    dts = pd.date_range(beg_date, end_date, freq="D", inclusive="left")
        
    #print("latitude: ", ds_region["latitude"].to_numpy())


    for idx_r, region in enumerate(ds_region.coords["region"]):
        print("Doing region: ", region)
        

        mask = ds_region["mask_PRISM"].sel(region=region).to_numpy()
        wgt  = np.cos(ds_region["lat_PRISM"].to_numpy() * np.pi/180.0)[:, None]
       
        valid_idx = mask == 1
        wgt_valid = wgt * mask 
    

        print("SUM OF WGT_VALID: ", np.sum(wgt_valid))

        for idx_vv, verification_varname in enumerate(verification_varnames):
        
            print("Doing variable: ", verification_varname)

            for idx_d in range(args.test_days):
            
                print("Doing test day: ", idx_d)
               
                dt = beg_date + pd.Timedelta(days = idx_d)


                for idx_y, year in enumerate(years):

                    PRISM_filename = "/expanse/lustre/scratch/t2hsu/temp_project/data/PRISM/PRISM-ppt-{y:04d}-{md:s}.nc".format(
                       y=year,
                       md = dt.strftime("%m-%d"),
                    )
 
                    ds = xr.open_dataset(PRISM_filename).isel(time=0)
                    
                    data = ds[verification_varname].to_numpy()
                    data[np.isnan(data)] = 0.0
                    avg_data = np.sum(data * wgt_valid) / np.sum(wgt_valid)
            
                    output_data[idx_d, idx_r, idx_vv, idx_y] = avg_data



    output_ds = xr.Dataset(
        
        data_vars = dict(
            obs_data=(["time", "region", "variable", "year"], output_data),
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
         
    

