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
import WRF_ens_tools

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

def doJob(details, detect_phase=False):

    # phase \in ['detect', 'work']
    result = dict(details = details, status="UNKNOWN", need_work=False, detect_phase=detect_phase)

    try:

         
        varname     = details["varname"]
        input_root  = Path(details["input_root"])
        expblob     = details["expblob"]
        mask_file   = Path(details["mask_file"])
        regions     = details["regions"] 
        target_time     = pd.Timestamp(details["target_time"])
        
        output_dir     = Path(details["output_dir"]) 
        output_prefix   = details["output_prefix"] 
        
        output_file = output_dir / ("%s_%s.nc" % (
            output_prefix,
            target_time.strftime("%Y-%m-%d"),
        ))
        
        # Detecting
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

        # Loading mask
        print("loading mask file: ", mask_file)
        ds_mask = xr.open_dataset(mask_file)
        print(ds_mask)
        
        ds_mask = ds_mask.rename({"lat_regridded" : "lat", "lon_regridded" : "lon"})
        if regions is not None:
            ds_mask = ds_mask.sel(region=regions)
        
        print("##### Doing time: ", target_time)
        
        output_file.parent.mkdir(parents=True, exist_ok=True)
        expsets = WRF_ens_tools.parseExpblob(expblob)
       
        ds = [] 
        ens_cnt = 0
        for expname, group, ens_rng in expsets:

            print("Expname/group : %s/%s" % (expname, group))            
            print("Ensemble ids: %s" % ( ",".join(["%d" % i for i in ens_rng] ) ) )
            
            print("Load %s - %s" % (expname, group,)) 
            _ds = WRF_ens_tools.loadGroup(expname, group, ens_rng, varname, target_time, root=input_root)
            _ds = _ds.assign_coords({"ens" : np.arange(_ds.dims["ens"]) + ens_cnt})
            ds.append(_ds)

            ens_cnt += _ds.dims["ens"]

        da = xr.merge(ds).transpose("ens", "time", "lat", "lon")[varname]
        da_weighted = da.weighted(ds_mask["wgt_regridded"] * ds_mask["mask_regridded"])
        
        da_mean  = da_weighted.mean(dim=["lat", "lon", "ens"]).expand_dims(dim={"stat": ["mean"]}, axis=1).rename(varname)
        da_std   = da_weighted.std(dim=["lat", "lon", "ens"]).expand_dims(dim={"stat": ["std"]}, axis=1).rename(varname)
        da_count = xr.zeros_like(da).isel(lon=0, lat=0).count(dim="ens").expand_dims(dim={"stat": ["count"]}, axis=1).rename(varname)
        
        ds_output = xr.merge([da_mean, da_std, da_count])
        #da_avg = da.weighted(ds_mask["wgt_regridded"] * ds_mask["mask_regridded"]).mean(dim=["lat", "lon", "ens"]).compute()

        #return_dict={
        #    varname : ds_avg[varname].to_numpy() for varname in list(ds_avg.keys())
        #}
        
        #merge_data = []
        #merge_data.append(da.std(dim="ens").expand_dims(dim={"stat": ["std",]}, axis=1).rename(varname))
        #merge_data.append(da.mean(dim="ens").expand_dims(dim={"stat": ["mean",]}, axis=1).rename(varname))
        #merge_data.append(da.count(dim="ens").expand_dims(dim={"stat": ["count",]}, axis=1).rename(varname))

        #ds_output = xr.merge(merge_data)
        
        print("Saving output: ", output_file)
        ds_output.to_netcdf(output_file, unlimited_dims="time")

        if output_file.exists():
            print("File `%s` is generated." % (str(output_file),))


        result['status'] = 'OK'

    except Exception as e:

        traceback.print_exc()
        result['status'] = 'ERROR'
        #traceback.print_stack()

        print(e)


    return result


def work(details):

    print("Enter")


    input_dir = details["input_dir"]
    mask_file = details["mask_file"]
    mask_varname = details["mask_varname"]
    
        
    varname     = details["varname"]
    expblob     = details["expblob"]



    beg_time  = pd.Timestamp(details["beg_time"])
    lead_days = details["lead_days"]
    end_time  = beg_time + pd.Timedelta(days=lead_days)
    wsm = details["wsm"] 
    regions = details["regions"]

    output_prefix   = details["output_prefix"] 
    
    output_file = output_dir / ("%s_%s.nc" % (
        output_prefix,
        target_time.strftime("%Y-%m-%d"),
    ))
 
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

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--input-root', type=str, help='Input directories.', required=True)
    parser.add_argument('--expblob', type=str, help='Input directories.', required=True)
    parser.add_argument('--varnames', type=str, nargs="+", help='Variables processed. If only a particular layer is selected, use `-` to separate the specified layer. Example: PH-850', required=True)
    parser.add_argument('--exp-beg-time', type=str, help='analysis beg time', required=True)
    parser.add_argument('--time-beg',    type=int, help='Beginning of time. Hours` after `--exp-beg-time`', required=True)
    parser.add_argument('--time-end',    type=int, help='Ending of time. Hours after `--exp-beg-time`', required=True)
    parser.add_argument('--time-stride', type=int, help='Each plot differ by this time interval. Hours', required=True)
    parser.add_argument('--output-dir', type=str, help='Output filename in nc file.', required=True)
    parser.add_argument('--output-prefix', type=str, help='analysis beg time', required=True)
    parser.add_argument('--nproc', type=int, help="Number of processors", default=1)
    parser.add_argument('--mask', type=str, help="Mask file", required=True)
    parser.add_argument('--regions', type=str, nargs="*", help='Region name.', default=None)

    args = parser.parse_args()

    print(args)
 
    N = int((args.time_end - args.time_beg) / args.time_stride) + 1
    
    failed_dates = []
    input_args = []
   
    
    for varname in args.varnames:
        
        for i in range(N):
            
            target_time = pd.Timestamp(args.exp_beg_time) + pd.Timedelta(hours=args.time_beg + i * args.time_stride) 
            details = dict(
                varname     = varname,
                input_root  = args.input_root,
                expblob     = args.expblob,
                target_time = target_time,
                output_dir  = args.output_dir,
                output_prefix = args.output_prefix,
                mask_file     = args.mask,
                regions = args.regions,
            )

            print("[Detect] Checking time=", target_time,)
            
            result = doJob(details, detect_phase=True)
            
            if not result['need_work']:
                print("File `%s` already exist. Skip it." % (str(result['output_file']),))
                continue
            
            input_args.append((details, False))

    with Pool(processes=args.nproc) as pool:

        results = pool.starmap(doJob, input_args)

        for i, result in enumerate(results):
            if result['status'] != 'OK':
                print(result)
                print('!!! Failed to generate output file %s.' % (str(result['output_file']),))
                failed_dates.append(result['details'])


    print("Tasks finished.")

    print("Failed output files: ")
    for i, failed_detail in enumerate(failed_dates):
        print("%d : " % (i+1), failed_detail)

    print("Done.")    



"""
if __name__ == "__main__":
   
    multiprocessing.freeze_support()
 
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--input-root', type=str, help='Input directories.', required=True)
    parser.add_argument('--expblob', type=str, help='Input directories.', required=True)
    parser.add_argument('--exp-beg-time', type=str, help='analysis beg time', required=True)
    parser.add_argument('--time-beg',    type=int, help='Beginning of time. Hours` after `--exp-beg-time`', required=True)
    parser.add_argument('--time-end',    type=int, help='Ending of time. Hours after `--exp-beg-time`', required=True)
    parser.add_argument('--time-stride', type=int, help='Each plot differ by this time interval. Hours', required=True)
    parser.add_argument('--mask', type=str, help="Mask file", required=True)
    parser.add_argument('--regions', type=str, nargs="*", help='Region name.', default=None)
    parser.add_argument('--varnames', type=str, nargs="+", help='Variables processed. If only a particular layer is selected, use `-` to separate the specified layer. Example: PH-850', required=True)
    parser.add_argument('--nproc', type=int, help="Time range in hours after --exp-beg-time", default=1)
    
    parser.add_argument('--output', type=str, help='Output netCDF file.', required=True)

    args = parser.parse_args()

    print(args)
    
    N = int((args.time_end - args.time_beg) / args.time_stride) + 1

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


"""           
