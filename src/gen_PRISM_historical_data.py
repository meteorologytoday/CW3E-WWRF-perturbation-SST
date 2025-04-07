import traceback
import xarray as xr
import pandas as pd
import numpy as np
import argparse
import os
from pathlib import Path
import PRISM_tools

def getAvgData(dts, wgt):

    PRISM_filenames = [ PRISM_tools.getPRISMFilename(dt) for dt in dts]


    ds = xr.open_mfdataset(PRISM_filenames).mean(dim="time")
   
    data = ds[verification_varname].to_numpy()

    #print(data[np.isfinite(data)])

    data[np.isnan(data)] = 0.0

    #print(data.shape, "; ", wgt.shape)


    avg_data = np.sum(data * wgt) / np.sum(wgt)

    return avg_data








if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--year-rng', type=int, nargs=2, help='Verification date range.', required=True)
    parser.add_argument('--valid-months', type=int, nargs="+", help='Verification date range.', default=[1,2,3,4,5,6,7,8,9,10,11,12])
    parser.add_argument('--half-window-size', type=int, help='Verification date range.', required=True)
    parser.add_argument('--region-file', type=str, help='Region file.', required=True)
    parser.add_argument('--output-dir', type=str, help='Output filename in nc file.', required=True)

    args = parser.parse_args()

    print(args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)


    ds_region = xr.open_dataset(args.region_file)    

    verification_varnames = ["total_precipitation", ]

    ref_time = "1970-01-01"
    ref_dt   = pd.Timestamp(ref_time)

    for year in range(args.year_rng[0], args.year_rng[1]+1):

        print(f"Doing year {year:04d}")

        beg_date = pd.Timestamp(year=year, month=1, day=1)
        end_date = pd.Timestamp(year=year+1, month=1, day=1)
        dts = list(pd.date_range(beg_date, end_date, freq="D", inclusive="left"))

        time_vec = [ (dt - ref_dt)/pd.Timedelta(days=1) for dt in dts ]


        output_data_obs = np.zeros((len(dts), ds_region.dims["region"], len(verification_varnames),))
        output_data_obs[:] = np.nan
        
        output_file = output_dir / f"PRISM_stat_year{year:04d}.nc"

        if output_file.exists():
            print("Output file %s already exists. Skip." % (str(output_file),))
            continue

        for idx_d, dt in enumerate(dts):
            
            try: 
                if dt.month not in args.valid_months:
                    continue

                 
                for idx_r, region in enumerate(ds_region.coords["region"]):
                    
                    #print("Doing region: ", region)
        
                    mask = ds_region["mask_PRISM"].sel(region=region).to_numpy()
                    wgt  = np.cos(ds_region["lat_PRISM"].to_numpy() * np.pi/180.0)[:, None]
                   
                    wgt_valid = wgt * mask 
                
                    for idx_vv, verification_varname in enumerate(verification_varnames):
                    
                        half_window = pd.Timedelta(days=args.half_window_size) 
                        selected_dts = pd.date_range(dt - half_window, dt + half_window, freq="D", inclusive="both")

                        output_data_obs[idx_d, idx_r, idx_vv] = getAvgData(selected_dts, wgt_valid)

            except Exception as e:
                
                print(e)
                print("Something's wrong with date : ", dt)

        output_ds = xr.Dataset(
            
            data_vars = dict(
                obs_data=(["time", "region", "variable",], output_data_obs),
            ),
            
            coords=dict(
                time = ( ["time",], time_vec, {'units': 'days since 1970-01-01 00:00:00'}),
                region = ds_region.coords["region"],
                variable = verification_varnames,
            ),
        )
                
        print("Output file: ", output_file)
        output_ds.to_netcdf(output_file, unlimited_dims="time")
        
            

