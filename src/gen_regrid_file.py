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
    parser.add_argument('--WRF-file', type=str, help='WRF file that provide XLAT and XLONG.', required=True)
    parser.add_argument('--PRISM-file', type=str, help='PRISM file that provide XLAT and XLONG.', required=True)
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
    
    WRF_ds = xr.open_dataset(args.WRF_file)
    
    WRF_llat = WRF_ds.coords["XLAT"].isel(Time=0).to_numpy()
    WRF_llon = WRF_ds.coords["XLONG"].isel(Time=0).to_numpy() % 360.0

    WRF_lat_idx, WRF_lon_idx, lat_regrid_bnds, lon_regrid_bnds = computeBoxIndex(WRF_llat, WRF_llon, args.lat_rng, args.lon_rng, args.dlat, args.dlon)

    PRISM_ds = xr.open_dataset(args.PRISM_file)
    PRISM_lat = PRISM_ds["lat"].to_numpy()
    PRISM_lon = PRISM_ds["lon"].to_numpy() % 360.0
    PRISM_llat, PRISM_llon = np.meshgrid(PRISM_lat, PRISM_lon, indexing='ij')
    PRISM_lat_idx, PRISM_lon_idx, _, _ = computeBoxIndex(PRISM_llat, PRISM_llon, args.lat_rng, args.lon_rng, args.dlat, args.dlon)
    
    new_ds = xr.Dataset(
        data_vars = dict(
            WRF_lat_idx = (["south_north", "west_east"], WRF_lat_idx),
            WRF_lon_idx = (["south_north", "west_east"], WRF_lon_idx),
            PRISM_lat_idx = (["lat", "lon"], PRISM_lat_idx),
            PRISM_lon_idx = (["lat", "lon"], PRISM_lon_idx),
            lat_regrid_bnd = (["lat_regrid_bnd",], lat_regrid_bnds),
            lon_regrid_bnd = (["lon_regrid_bnd",], lon_regrid_bnds),
        ),
        coords = dict(
            XLAT = (["south_north", "west_east"], WRF_llat),
            XLONG = (["south_north", "west_east"], WRF_llon),
            lat = (["lat"], PRISM_lat),
            lon = (["lon"], PRISM_lon),
        ),
        attrs = dict(
            nlat_box = max_lat_idx+1,
            nlon_box = max_lon_idx+1,
        )
    )

    print("Output file: %s" % (args.output,))
    new_ds.to_netcdf(args.output)
    
