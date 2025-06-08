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

def computeBoxIndex(llat, llon, lat_rng, lon_rng, dlat, dlon):

    nbox_lat = int(np.floor( ( lat_rng[1] - lat_rng[0] ) / dlat))
    nbox_lon = int(np.floor( ( lon_rng[1] - lon_rng[0] ) / dlon))
 
    if nbox_lat == 0 or nbox_lon == 0:
        raise Exception("Error: The given lat lon range and spacing does not generate and box.")

    
    lat_idx = np.floor( (llat - lat_rng[0]) / dlat).astype(np.int32)
    lon_idx = np.floor( (llon - lon_rng[0]) / dlon).astype(np.int32)
    
    lat_idx[ (lat_idx >= nbox_lat) | (lat_idx < 0)] = -1    
    lon_idx[ (lon_idx >= nbox_lon) | (lon_idx < 0)] = -1  

    lat_regrid_bnds = np.linspace(lat_rng[0], lat_rng[1], nbox_lat + 1)
    lon_regrid_bnds = np.linspace(lon_rng[0], lon_rng[1], nbox_lon + 1)

    return lat_idx, lon_idx, lat_regrid_bnds, lon_regrid_bnds



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--input-file', type=str, help='Can be WRF file that provide XLAT and XLONG, PRISM file, or a regrid file.', required=True)
    parser.add_argument('--input-type', type=str, help='Can be `WRF`, `PRISM`, or `regrid`.', choices=["WRF", "PRISM", "regrid",], required=True)
    parser.add_argument('--output', type=str, help='WRF file that provide XLAT and XLONG.', required=True)
    parser.add_argument('--lat-rng', type=float, nargs=2, help="Latitudinal range.", default=[-90.0, 90.0])
    parser.add_argument('--lon-rng', type=float, nargs=2, help="Latitudinal range.", default=[0.0, 360.0])
    parser.add_argument('--dlat', type=float, help="dlat in latitudinal direction.", required=True)
    parser.add_argument('--dlon', type=float, help="dlon in longitudinal direction.", required=True)
    args = parser.parse_args()

    print(args)


    max_lat_idx = int(np.floor( ( args.lat_rng[1] - args.lat_rng[0] ) / args.dlat))
    max_lon_idx = int(np.floor( ( args.lon_rng[1] - args.lon_rng[0] ) / args.dlon))
    
    print("max_lat_idx: ", max_lat_idx)
    print("max_lon_idx: ", max_lon_idx)

    data_vars = dict()
    coords = dict()


    with xr.open_dataset(args.input_file, engine="netcdf4") as ds:

        if args.input_type == "WRF":

            llat = ds.coords["XLAT"].isel(Time=0).to_numpy()
            llon = ds.coords["XLONG"].isel(Time=0).to_numpy() % 360.0

            dims = ["south_north", "west_east"]
            coords.update(dict(
                XLAT = (dims, llat), 
                XLONG = (dims, llon), 
            ))
           
        elif args.input_type == "PRISM":
            
            lat = ds["lat"].to_numpy()
            lon = ds["lon"].to_numpy() % 360.0
            llat, llon = np.meshgrid(lat, lon, indexing='ij')
            
            dims = ["lat", "lon"]
            coords.update(dict(
                lat = (["lat"], lat), 
                lon = (["lon"], lon), 
            ))
        
        elif args.input_type == "regrid":

            lat = ds["lat_regrid"].to_numpy()
            lon = ds["lon_regrid"].to_numpy() % 360.0
            llat, llon = np.meshgrid(lat, lon, indexing='ij')
 
            dims = ["lat", "lon"]
            coords.update(dict(
                lat = (["lat"], lat), 
                lon = (["lon"], lon), 
            ))


        lat_idx, lon_idx, lat_regrid_bnds_, lon_regrid_bnds = computeBoxIndex(llat, llon, args.lat_rng, args.lon_rng, args.dlat, args.dlon)
        
 
        lat_regrid = ( lat_regrid_bnds[1:] + lat_regrid_bnds[:-1] ) / 2
        lon_regrid = ( lon_regrid_bnds[1:] + lon_regrid_bnds[:-1] ) / 2
     
        new_ds = xr.Dataset(
            data_vars = dict(
                lat_idx = (dims, lat_idx),
                lon_idx = (dims, lon_idx),
                lat_regrid_bnd = (["lat_regrid_bnd",], lat_regrid_bnds),
                lon_regrid_bnd = (["lon_regrid_bnd",], lon_regrid_bnds),
                lat_regrid = (["lat_regrid",], lat_regrid),
                lon_regrid = (["lon_regrid",], lon_regrid),
            ),
            coords = coords,
            attrs = dict(
                nlat_box = max_lat_idx+1,
                nlon_box = max_lon_idx+1,
            )
        )

        print("Output file: %s" % (args.output,))
        new_ds.to_netcdf(args.output)
        
