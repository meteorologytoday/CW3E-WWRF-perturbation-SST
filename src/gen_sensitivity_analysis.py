from multiprocessing import Pool
import multiprocessing
import time
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

import PRISM_tools
import cmocean
from scipy import sparse
import WRF_ens_tools
import regrid_tools

import re
import matrix_helper

def doJob(details, detect_phase=False):

    # phase \in ['detect', 'work']
    result = dict(details = details, status="UNKNOWN", need_work=False, detect_phase=detect_phase)

    try:
       
 

        # This is the target variable, such as total rainfall
        target_dt = details["target_dt"]
        target_varname = details["target_varname"]


        
        target_mask_type   = details["target_mask_type"]
        target_mask_file   = details["target_mask_file"]
        target_mask_file_regions = np.array(details["target_mask_file_regions"])

 
        # This lat-lon range applies to target
        target_lat_rng     = details["target_lat_rng"]
        target_lon_rng     = details["target_lon_rng"]
       
        # This is the sensitivity variable, such as initial moisture
        sens_dt = details["sens_dt"]
        sens_varname = details["sens_varname"]
        sens_regrid_file = details["sens_regrid_file"]

        input_root  = Path(details["input_root"])
        expblob     = details["expblob"]
        
        output_dir     = Path(details["output_dir"]) 
        output_prefix   = details["output_prefix"] 
        
        output_file = output_dir / ("sensitivity_%s_sens-%s-%s_target-%s-%s.nc" % (
            output_prefix,
            sens_dt.strftime("%Y-%m-%d_%H"),
            sens_varname,
            target_dt.strftime("%Y-%m-%d_%H"),
            target_varname,
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
        
        print("##### Doing time: ", target_dt)
        
        output_file.parent.mkdir(parents=True, exist_ok=True)
    
        # Load regrid file for sensitivity
        do_sens_regrid = sens_regrid_file is not None
        
        if do_sens_regrid:
            sens_avg_info = regrid_tools.constructAvgMtxFromFile(sens_regrid_file)

        print("Loading sens_da")
        sens_da   = WRF_ens_tools.loadExpblob(expblob, sens_varname,   sens_dt,   root=input_root).isel(time=0)

        if do_sens_regrid:
            regridded_sens_data = regrid_tools.regrid(sens_avg_info, sens_da.to_numpy())
            print("regridded_sens_data : ", regridded_sens_data)
            regridded_sens_da = xr.DataArray(
                name = "regridded_sens_da",
                data = regridded_sens_data,
                dims = ("ens", "lat", "lon"),
                coords = dict(
                    lat = sens_avg_info["lat_regrid"],
                    lon = sens_avg_info["lon_regrid"],
                    ens = sens_da.coords["ens"],
                )
            )


            ds_regridded_bnd = xr.Dataset(
                data_vars = dict(
                    lat_regrid_bnd = (["lat_regrid_bnd", ], sens_avg_info["lat_regrid_bnd"]),
                    lon_regrid_bnd = (["lon_regrid_bnd", ], sens_avg_info["lon_regrid_bnd"]),
                ),
            )
        else:
            regridded_sens_da = sens_da
 


        print("Loading target_da")
        target_da = WRF_ens_tools.loadExpblob(expblob, target_varname, target_dt, root=input_root).isel(time=0) 
        # average target_da
        if target_mask_type == "latlon_rng":
            
            lat = target_da.coords["lat"]
            lon = target_da.coords["lon"]
            box_rng = (
                (lat >= target_lat_rng[0]) &
                (lat <  target_lat_rng[1]) &
                (lon >= target_lon_rng[0]) &
                (lon <  target_lon_rng[1])
            )

            target_da = target_da.where(box_rng).mean(
                dim=["lat", "lon"], skipna=True,
            ).expand_dims(dim="region", axis=0)
            
        elif target_mask_type == "mask_file":
            print("Using mask_file: ", target_mask_file)
            mask_da = xr.open_dataset(target_mask_file)["mask_regridded"].rename(
                {
                    'lat_regridded': 'lat',
                    'lon_regridded': 'lon',
                }
            )
            if target_mask_file_regions is not None:
                #print(mask_da)
                mask_da = mask_da.sel(region=target_mask_file_regions)
                #print(mask_da)
            
                
            target_da = target_da.expand_dims(dim={"region": mask_da.coords["region"]}, axis=0)
            target_da = target_da.where(mask_da == 1).mean(dim=["lat", "lon"], skipna=True)

            #for region in mask_da.coords["region"]:
            #    print("region = ", str(region.values))
            #    _mask = mask_da.sel(region=region)
            #    avg_da = target_da.where(_mask == 1).mean(dim=["lat", "lon"], skipna=True).expand_dims(dim={"region": [region,]}, axis=0)
            #    print("avg_da = ", avg_da)

            #    _tmp.append(avg_da)
            
            #target_da = xr.merge(_tmp).rename("target_da")
            #print(target_da)


        #
        # Mathematical basis:
        #
        # y = G.T * f
        #
        # where y = response, f = forcing
        #
        # for example, y is the rainfall, f is the moisture,
        # G is the green's function. Each column vector of G
        # is a delta response map to forcing
        #
        # Then, we have
        # y * fT = G.T * (f * fT)
        # or 
        # (f * fT) * G = f * yT
        # Therefore, 
        #
        # G = (f * yT) / ( f * fT )
        #
        coords = regridded_sens_da.coords
        Ne = len(coords["ens"])
        Nx = len(coords["lon"])
        Ny = len(coords["lat"])

        # First construct reduced matrix
        # the mask here are used to remove nan points
        mask = np.isfinite(regridded_sens_da.isel(ens=0).transpose("lat", "lon").to_numpy().flatten())
        invalid_mask = np.isnan(regridded_sens_da.isel(ens=0).transpose("lat", "lon").to_numpy())
        
        valid_pts = np.sum(mask)
        invalid_pts = np.sum(invalid_mask)

        print("Valid points: %d / %d (%.2f%%)" % (
            valid_pts,
            valid_pts + invalid_pts,
            valid_pts / ( valid_pts  + invalid_pts ) * 100.0,
        )) 

        reduce_mtx = matrix_helper.constructSubspaceWith(mask)

        # construct f and y
        f_full = regridded_sens_da.transpose("ens", "lat", "lon").to_numpy().reshape(
            (Ne, Ny * Nx)
        ).transpose()
        f_full[np.isnan(f_full)] = 0.0 # need to remove nan otherwise the multiplication gets nan
        f = reduce_mtx @ f_full
            
        # remove ensemble mean
        f = f - f.mean(axis=1, keepdims=True)
        print("f.shape = ", f.shape)

        ds_output = []

        if do_sens_regrid:
            ds_output.append(ds_regridded_bnd)


        for region in target_da.coords["region"]:


        
            region_str = str(region.values)
            print("Doing region:", region_str)
            #print(target_da)
            _target_da = target_da.sel(region=region)

            y = _target_da.to_numpy().reshape((1, -1))
            
            # remove ensemble mean

            print("y.shape = ", y.shape)
            y = y - y.mean(axis=1, keepdims=True)
            
            # compute forcing correlation
            ffT = f @ f.T
           
            #print("f*f^T = ", ffT) 
            # comptue left hand side
            fyT = f @ y.T
            fyT_full = reduce_mtx.T @ fyT
            
            # compute Green's function
            computing_start_time = time.perf_counter()
            inverse_exists = True
            G_full = None
            try:
                

                print("ffT.shape = ", ffT.shape)
                print("fyT.shape = ", fyT.shape)
                
                r = np.linalg.matrix_rank(ffT)
                print("Precomputed rank: %d" % (r,)) 
                
                print("Solving (ffT) G = fyT")
                # G = greens function (column vectors are responses)
                #G = np.linalg.solve(ffT.copy(), fyT.copy())
                G = np.linalg.pinv(ffT.copy()) @ fyT
                #G = np.linalg.inv(ffT.copy()) @ fyT
                
                G_full = reduce_mtx.T * G
                G_full = G_full.reshape( ( Ny, Nx ) )
                G_full[invalid_mask] = np.nan
                
            except Exception as e:
                inverse_exists = False
                print("Error occurs when solving matrix")
                print(str(e))

            
            if G_full is None: # no inverse
                G_full = np.zeros((Ny, Nx))
                G_full[:] = np.nan
            

            computing_end_time = time.perf_counter()
            computing_elapsed_time = computing_end_time - computing_start_time
            print(f"Solving matrix takes {computing_elapsed_time} seconds.")
 
            """
            G_map = xr.DataArray(
                name = "GreenFunc",
                data = G_full.reshape( (1, Ny, Nx ) ),
                dims = ("region", "lat", "lon"),
                coords = dict(
                    region = [region, ],
                    lat = coords["lat"],
                    lon = coords["lon"],
                ),
            )
            """
        
            ds_output.append(xr.Dataset(
                data_vars = dict(
                    GreenFunc = (["region", "lat", "lon"], G_full.reshape( (1, Ny, Nx ) )),
                    fyT = (["region", "lat", "lon"], fyT_full.reshape( (1, Ny, Nx) )),
                    ffT = (["region", "allpts1", "allpts2"], ffT.reshape((1, *ffT.shape))),
                ),
                coords = dict(
                    region = np.array([region_str, ]),
                    lat = (["lat"], coords["lat"].to_numpy()),
                    lon = (["lon"], coords["lon"].to_numpy()),
                ),
            ))

        ds_output = xr.merge(ds_output)#.set_coords(["lat", "lon"])
        
        print("Result: ", ds_output)
        print("Saving output: ", output_file)
        ds_output.to_netcdf(output_file)

        if output_file.exists():
            print("File `%s` is generated." % (str(output_file),))

        result['status'] = 'OK'

    except Exception as e:

        traceback.print_exc()
        result['status'] = 'ERROR'
        #traceback.print_stack()

        print(e)


    return result

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--input-root', type=str, help='Input directories.', required=True)
    parser.add_argument('--expblob', type=str, help='Input directories.', required=True)
    parser.add_argument('--target-varname', type=str, help='analysis beg time', required=True)
    parser.add_argument('--target-time', type=str, help='analysis beg time', required=True)
    parser.add_argument('--sens-regrid-file', type=str, help='Sensitivity\'s regrid file', default=None)
    parser.add_argument('--sens-varname', type=str, help='analysis beg time', required=True)
    parser.add_argument('--sens-time', type=str, help='analysis beg time', required=True)
    
    parser.add_argument('--target-mask-type', type=str, help='analysis beg time', required=True, choices=["latlon_rng", "mask_file"])
    parser.add_argument('--target-mask-file', type=str, help='analysis beg time', default=None)
    parser.add_argument('--target-mask-file-regions', type=str, nargs="+", help='Regions to be diagnosed. If none is provided, then means all regions will be applied', default=None)
    parser.add_argument('--target-lat-rng', type=float, nargs=2, help="Latitude range for box. It is used if `--target-mask-type` is 'latlon_rng'. ", default=[-90.0, 90.0])
    parser.add_argument('--target-lon-rng', type=float, nargs=2, help="Longitudinal range for box. It is used if `--target-mask-type` is 'latlon_rng'. ", default=[0.0, 360.0])
 
    parser.add_argument('--output-dir', type=str, help='Output filename in nc file.', required=True)
    parser.add_argument('--output-prefix', type=str, help='analysis beg time', required=True)
    parser.add_argument('--nproc', type=int, help="Number of processors", default=1)

    args = parser.parse_args()

    print(args)
 
    details = dict(

        target_varname = args.target_varname,
        sens_regrid_file = args.sens_regrid_file,
        sens_varname   = args.sens_varname,
        target_dt      = pd.Timestamp(args.target_time),
        sens_dt        = pd.Timestamp(args.sens_time),
        input_root     = args.input_root,
        expblob        = args.expblob,
        output_dir     = args.output_dir,
        output_prefix  = args.output_prefix,

        target_mask_type = args.target_mask_type,
        target_mask_file = args.target_mask_file,
        target_mask_file_regions = args.target_mask_file_regions,
        target_lat_rng = args.target_lat_rng,
        target_lon_rng = args.target_lon_rng,
    )

    result = doJob(details, detect_phase=True)
    
    if not result['need_work']:
        print("File `%s` already exist. Skip it." % (str(result['output_file']),))
    else:
        doJob(details, detect_phase=False)
    
    print("Done.")    

