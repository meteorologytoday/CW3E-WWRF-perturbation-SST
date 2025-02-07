from multiprocessing import Pool

from pathlib import Path
import numpy as np
import xarray as xr
import pandas as pd
import argparse
import traceback
import os

import shapely
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

import rasterio
from rasterio.transform import xy


def make_mask_with_shape(xx, yy, shp):
    
    N1, N2 = xx.shape
    mask = np.zeros((N1, N2), dtype=np.int32)

    for i1 in range(N1):
        for i2 in range(N2):
            p = Point(xx[i1, i2], yy[i1, i2])
            mask[i1, i2] = 1 if shp.contains(p) else 0

    return mask



def work(details):
    
    region = details["region"]
    polygon = details["polygon"]

    dataset = details["dataset"]
    llon = details["llon"]
    llat = details["llat"]
    ocn_idx = details["ocn_idx"]
    
    result = dict(details=details, status="UNKNOWN", output=None) 
    try:
        label = f"({dataset:s}, {region:s})"
        print(f"Doing (dataset, region) = {label:s}")
        _mask = make_mask_with_shape(llon, llat, polygon)
        _mask[ocn_idx] = 0
        result["output"] = _mask
        result["status"] = "OK"
       
        print(f"Done {label:s}") 
    except Exception as e:
        
        print("Loading error. Put None.")
        result["status"] = "ERROR"
        traceback.print_exc()


    return result
   




if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Process some integers.')
     
    parser.add_argument('--test-ERA5-file', type=str, required=True)
    parser.add_argument('--test-WRF-file', type=str, required=True)
    parser.add_argument('--test-PRISM-file', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--nproc', type=int, default=1)
    args = parser.parse_args()
    print(args)

    regions = ["CA", "sierra", "coastal", "city_LA"]

    # Region Polygon

    polygon_dict = dict( 

        CA = (CA := Polygon([
            (235.0, 42.0),
            (240.0, 42.0),
            (240.0, 39.0),
            (246.0, 35.0),
            (246.0, 32.7),
            (235.0, 32.7),
        ])),

        sierra = shapely.intersection(
            CA,
            Polygon([
                ((360-122.3), 40.5),
                ((360-100.3), 40.5),
                ((360-100.3), 35.0),
                ((360-119.0), 35.0),
            ]),
        ),

        coastal = shapely.intersection(
            CA,
            Polygon([
                ((360-130.0), 42.0),
                ((360-122.4), 42.0),
                ((360-122.3), 40.5),
                ((360-119.0), 35.0),
                ((360-116.5), 34.2),
                ((360-116.0), 32.2),
                ((360-130.0), 32.2),
            ]),
        ),


        city_LA = shapely.intersection(
            CA,
            Point((360-118.126), 34).buffer(0.5, quad_segs=360,),
        ),
    )



    # === PRISM ===

    ds_PRISM = xr.open_dataset(args.test_PRISM_file)

    test_da_PRISM = ds_PRISM["total_precipitation"].isel(time=0)
    ocn_idx_PRISM = np.isnan(test_da_PRISM.to_numpy())

    lon_PRISM = ds_PRISM.coords["lon"].to_numpy().astype(np.float64)
    lat_PRISM = ds_PRISM.coords["lat"].to_numpy().astype(np.float64)

    llat_PRISM, llon_PRISM = np.meshgrid(lat_PRISM, lon_PRISM, indexing='ij')

    mask_PRISM = xr.DataArray(
        data = np.zeros_like(llat_PRISM, dtype=np.int32),
        dims = ["lat_PRISM", "lon_PRISM"],
        coords = dict(
            lon_PRISM=(["lon_PRISM"], lon_PRISM),
            lat_PRISM=(["lat_PRISM"], lat_PRISM),
        ),
    )

    wgt_PRISM = mask_PRISM.copy().astype(np.float64).rename("wgt_PRISM")
    wgt_PRISM[:, :] = np.cos(llat_PRISM * np.pi/180.0)

    mask_PRISM = mask_PRISM.expand_dims(
        dim = dict(region=regions),
        axis=0,
    ).rename("mask_PRISM").copy()


    # === ERA5 ====
    ds_ERA5 = xr.open_dataset(args.test_ERA5_file)
    test_da_ERA5 = ds_ERA5["sst"].isel(time=0)

    full_mask_ERA5 = xr.ones_like(test_da_ERA5).astype(int)
    lnd_idx_ERA5 = np.isnan(test_da_ERA5.to_numpy())
    ocn_idx_ERA5 = np.isfinite(test_da_ERA5.to_numpy())

    # Copy is necessary so that the value can be assigned later
    mask_ERA5 = xr.zeros_like(full_mask_ERA5).expand_dims(
        dim = dict(region=regions),
        axis=0,
    ).rename("mask_ERA5").copy()

    lat_ERA5 = ds_ERA5.coords["latitude"]
    lon_ERA5 = ds_ERA5.coords["longitude"] % 360
    llat_ERA5, llon_ERA5 = np.meshgrid(lat_ERA5, lon_ERA5, indexing='ij')

    wgt_ERA5 = xr.apply_ufunc(np.cos, lat_ERA5*np.pi/180).rename("wgt_ERA5")

    # === WRF ====
    ds_WRF = xr.open_dataset(args.test_WRF_file).isel(Time=0)

    full_mask_WRF = xr.ones_like(ds_WRF["SST"]).astype(int)
    lnd_mask_WRF = ds_WRF["LANDMASK"].to_numpy()
    lnd_idx_WRF = lnd_mask_WRF == 1
    ocn_idx_WRF = lnd_mask_WRF == 0


    # Copy is necessary so that the value can be assigned later
    mask_WRF = xr.zeros_like(full_mask_WRF).expand_dims(
        dim = dict(region=regions),
        axis=0,
    ).rename("mask_WRF").copy()

     
    llat_WRF = ds_WRF.coords["XLAT"]
    llon_WRF = ds_WRF.coords["XLONG"] % 360
    wgt_WRF = ds_WRF["AREA2D"].rename("wgt_WRF")


    infos = dict(

        WRF = dict(
            llat = llat_WRF,
            llon = llon_WRF,
            wgt  = wgt_WRF,
            ocn_idx = ocn_idx_WRF,
            mask = mask_WRF,
        ),

        ERA5 = dict(
            llat = llat_ERA5,
            llon = llon_ERA5,
            wgt  = wgt_ERA5,
            ocn_idx = ocn_idx_ERA5,
            mask = mask_ERA5,
        ),

        PRISM = dict(
            llat = llat_PRISM,
            llon = llon_PRISM,
            wgt  = wgt_PRISM,
            ocn_idx = ocn_idx_PRISM,
            mask = mask_PRISM,
        ),
    )

    output_datasets = infos.keys()
    #output_datasets = ["ERA5", "PRISM"]
    mapping_region_to_idx = { region : i for i, region in enumerate(regions) }


    input_args = []
    for dataset in output_datasets:
        info = infos[dataset]
        for i, region in enumerate(regions):

            print(f"Making input arg for (dataset, region) = ({dataset:s}, {region:s})")

            details = dict(
                polygon = polygon_dict[region],
                region  = region,
                dataset = dataset,
                llon = info["llon"],
                llat = info["llat"],
                ocn_idx = info["ocn_idx"],
            )
        
            input_args.append((details,))


    failed_cases = []
    with Pool(processes=args.nproc) as pool:
        results = pool.starmap(work, input_args)
        for i, result in enumerate(results):
            
            print(f'Collecting output of (dataset, region) = ({dataset:s}, {region:s}).')
            if result['status'] == 'OK':
                
                details = result['details']
                output_mask = result['output']
                
                region_idx = mapping_region_to_idx[details["region"]]
                dataset = details["dataset"]
                infos[dataset]["mask"][region_idx, :, :] = output_mask
                
            else:
                
                print(f'!!! Failed to generate output of (dataset, region) = ({dataset:s}, {region:s}).')
                  
                failed_cases.append(dict(dataset=dataset, region=region))

    
    merge_data = [infos[dataset]['mask'] for dataset in output_datasets] + [infos[dataset]['wgt'] for dataset in output_datasets]
    
    new_ds = xr.merge(merge_data)
    
    print("Output file: ", args.output)
    new_ds.to_netcdf(args.output)

    print("Failed case: ")
    for i, (dataset, region) in enumerate(failed_cases):
        print("%d : (dataset, region) = (%s, %s)" % (i+1, dataset, region,))


    print("Done.")


           
