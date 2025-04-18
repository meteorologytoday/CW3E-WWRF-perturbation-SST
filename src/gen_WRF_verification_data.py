import traceback
import xarray as xr
import pandas as pd
import numpy as np
import argparse
import tool_fig_config
import wrf_load_helper 
import datetime
import os
import wrf_preprocess
import cmocean
from pathlib import Path

import multiprocessing
from multiprocessing import Pool 

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


mapping = {
    "diff_TTL_RAIN" : "total_precipitation",
    "diff_RAINC"    : "convective_precipitation",
    "diff_RAINNC"   : "large_scale_precipitation",
}


def work(details):

    print("Enter")
    input_dir = details["input_dir"]
    mask_file = details["mask_file"]
    mask_varname = details["mask_varname"]
    beg_time  = pd.Timestamp(details["beg_time"])
    lead_days = details["lead_days"]
    end_time  = beg_time + pd.Timedelta(days=lead_days)
    wsm = details["wsm"] 
    regions = details["regions"]

    sel_time = [beg_time + pd.Timedelta(days=i) for i in range(lead_days)]

    result = dict(
        status="UNKNOWN",
        details = details,
    )
    
    print("ready")

    try:


        print("loading mask file: ", mask_file)
        ds_mask = xr.open_dataset(mask_file)
        
        print("loading wrf file")
        ds = wrf_load_helper.loadWRFDataFromDir(
            wsm, 
            input_dir,
            beg_time = beg_time,
            end_time = end_time,
            suffix=args.wrfout_suffix,
            avg=None,
            verbose=False,
            inclusive="both",
        ).sel(time=sel_time)
        

        if regions is not None:
            ds_mask = ds_mask.sel(region=regions)
        
        
        TTL_RAIN = (ds["RAINC"] + ds["RAINNC"]).rename("TTL_RAIN")

        print("Merg")
        ds = xr.merge([ds, TTL_RAIN])
        
        #print(ds.coords["time"].to_numpy())
        verification_data_array = []
        
        print("computing...")
        # Compute daily accumulative rainfall
        for acc_varname in ["RAINC", "RAINNC", "TTL_RAIN"]:
               
            new_varname = "diff_%s" % (acc_varname,)
            da = ds[acc_varname]
            diff = (da.shift(time=-1) - da).rename(new_varname).drop_vars("XTIME", errors="ignore")
            
            verification_data_array.append(diff)
        
    
        for i, da in enumerate(verification_data_array):
            verification_data_array[i] = da.expand_dims(dim={"region": ds_mask.coords["region"]}, axis=1)
            
        verification_ds = xr.merge(verification_data_array)

        #print(verification_ds)
        verification_ds_avg = verification_ds.weighted(ds_mask["wgt_WRF"] * ds_mask["mask_WRF"]).mean(dim=["south_north", "west_east"]).compute()

        return_dict={
            varname : verification_ds_avg[varname].to_numpy() for varname in list(verification_ds_avg.keys())
        }
            
            


        result["status"] = "OK"
        result["data"] = verification_ds_avg

        ds.close()
        ds_mask.close()

        print("Done %s" % (input_dir,)) 
       
    except Exception as e:
        
        print("Loading error. Put None.")
        result["status"] = "ERROR"
        traceback.print_exc()


    print("Return....")
    return result





if __name__ == "__main__":
   
    multiprocessing.freeze_support()
 
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--input-root', type=str, help='Input directories.', required=True)
    parser.add_argument('--expname', type=str, help='Input directories.', required=True)
    parser.add_argument('--group', type=str, help='Input directories.', required=True)
    parser.add_argument('--subgroup', type=str, help='Input directories.', default="BLANK")

    parser.add_argument('--ens-ids', type=str, help="Ens ids. Comma separated and can use range like 1-3,5,23-25", required=True)
    parser.add_argument('--time-beg', type=int, help="Time beg after --exp-beg-time", required=True)
    parser.add_argument('--lead-days', type=int, help="How many lead days" , required=True)
    parser.add_argument('--exp-beg-time', type=str, help='analysis beg time', required=True)
    parser.add_argument('--wrfout-data-interval', type=int, help='Time interval between each adjacent record in wrfout files in seconds.', required=True)
    parser.add_argument('--frames-per-wrfout-file', type=int, help='Number of frames in each wrfout file.', required=True)
    parser.add_argument('--wrfout-suffix', type=str, default="")
    parser.add_argument('--mask', type=str, help="Mask file", required=True)
    parser.add_argument('--regions', type=str, nargs="*", help='Region name.', default=None)
    parser.add_argument('--nproc', type=int, help="Time range in hours after --exp-beg-time", default=1)
    
    parser.add_argument('--output', type=str, help='Output netCDF file.', required=True)

    args = parser.parse_args()

    print(args)

    ens_ids = parse_ranges(args.ens_ids)
    print("Ensemble ids: %s => %s" % (args.ens_ids, ",".join(["%d" % i for i in ens_ids] ) ))
    verification_varnames = ["diff_RAINC", "diff_RAINNC", "diff_TTL_RAIN"]

    exp_beg_time = pd.Timestamp(args.exp_beg_time)
    wrfout_data_interval = pd.Timedelta(seconds=args.wrfout_data_interval)
    time_beg = exp_beg_time + pd.Timedelta(hours=args.time_beg)

    wsm = wrf_load_helper.WRFSimMetadata(
        start_datetime  = exp_beg_time,
        data_interval   = wrfout_data_interval,
        frames_per_file = args.frames_per_wrfout_file,
    )
    
    # Loading   
    data = []
    ens_id_to_idx_mapping = dict() 
    
    input_root = Path(args.input_root)
    input_args = []
    if args.subgroup == "BLANK":
        extra_casename = f"{args.group:s}_"
        subgroup_dir = ""

    else:
        extra_casename = f"{args.group:s}_{args.subgroup:s}_"
        subgroup_dir = args.subgroup

    

    for i, ens_id in enumerate(ens_ids):
      
        ens_id_to_idx_mapping[ens_id] = i
        input_dir = input_root / args.group / "runs" / subgroup_dir / f"{extra_casename:s}ens{ens_id:02d}" / "output" / "wrfout"
        
        input_args.append((dict(
            ens_id       = ens_id,
            input_dir    = input_dir,
            mask_file    = args.mask,
            mask_varname = "mask_WRF",
            beg_time     = time_beg,
            lead_days    = args.lead_days,
            wsm          = wsm,
            regions      = args.regions,
        ),))

  
    regions = None 
    if args.regions is None:
        with xr.open_dataset(args.mask) as ds_test:
            regions = ds_test.coords["region"].to_numpy().astype(str)

    else:
        regions = args.regions

    output_data = np.zeros((args.lead_days, len(regions),  len(verification_varnames), len(ens_ids), ))
    
    print("There are %d works." % (len(input_args),))

    failed_ens = []

    with Pool(processes=args.nproc) as pool:
        results = pool.starmap(work, input_args)

        for i, result in enumerate(results):

            print("Getting the %d-th result" % (i,))
            
            if result['status'] == 'OK':
               
                ens_id = result['details']['ens_id'] 
                ds = result['data']

                for k, varname in enumerate(verification_varnames):

                    factor = 1.0
                    if varname in ["diff_RAINNC", "diff_RAINC", "diff_TTL_RAIN"]:
                        factor = 1e-3
                    
                    idx = ens_id_to_idx_mapping[ens_id]
                    

                    output_data[:, :, k, idx] = ds[varname].to_numpy() * factor
                
            else:
                print('!!! Failed to generate output of date %s.' % (result['details']['ens_id']))
                failed_ens.append(result['details']['ens_id'])

            

    new_varnames = [ mapping[varname] if varname in mapping else varname for varname in verification_varnames ]

    output_ds = xr.Dataset(
        
        data_vars = dict(
            data=(["time", "region", "variable", "ens_id"], output_data),
        ),
        
        coords=dict(
            time = [time_beg + pd.Timedelta(days=i) for i in range(args.lead_days) ],
            variable = new_varnames,
            region = regions,
            ens_id = ens_ids,
        ),
    )
    
    print("Output file: ", args.output) 
    output_ds.to_netcdf(args.output) 

    print("Tasks finished.")

    print("Failed ens: ")
    for i, ens_id in enumerate(failed_ens):
        print("%d : ens_id = %d" % (i+1, ens_id,))


    print("Done.")


           
