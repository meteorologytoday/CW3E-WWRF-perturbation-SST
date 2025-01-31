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

parser = argparse.ArgumentParser(
                    prog = 'make_ECCC_AR_objects.py',
                    description = 'Postprocess ECCO data (Mixed-Layer integrated).',
)

parser.add_argument('--test-ERA5-file', type=str, required=True)
parser.add_argument('--test-WRF-file', type=str, required=True)
parser.add_argument('--test-PRISM-file', type=str, required=True)
parser.add_argument('--output', type=str, required=True)
args = parser.parse_args()
print(args)


def make_mask_with_shape(xx, yy, shp):
    
    N1, N2 = xx.shape
    mask = np.zeros((N1, N2), dtype=np.int32)

    for i1 in range(N1):
        for i2 in range(N2):
            p = Point(xx[i1, i2], yy[i1, i2])
            mask[i1, i2] = 1 if shp.contains(p) else 0

    return mask
    


regions = ["CA", "sierra", "coastal"]

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

)



# === PRISM ===

ds_PRISM = xr.open_dataset(args.test_PRISM_file)

test_da_PRISM = ds_PRISM["total_precipitation"].isel(time=0)
ocn_idx_PRISM = np.isnan(test_da_PRISM.to_numpy())

lon_PRISM = ds_PRISM.coords["lon"].to_numpy()
lat_PRISM = ds_PRISM.coords["lat"].to_numpy()

llat_PRISM, llon_PRISM = np.meshgrid(lat_PRISM, lon_PRISM, indexing='ij')

mask_PRISM = xr.DataArray(
    data = np.zeros_like(llat_PRISM, dtype=np.int32),
    dims = ["lat_PRISM", "lon_PRISM"],
    coords = dict(
        lon_PRISM=(["lon_PRISM"], lon_PRISM),
        lat_PRISM=(["lat_PRISM"], lat_PRISM),
    ),
)

wgt_PRISM = mask_PRISM.copy().rename("wgt_PRISM")
wgt_PRISM[:, :] = np.cos(llat_PRISM * np.pi/180.0)

mask_PRISM = mask_PRISM.expand_dims(
    dim = dict(region=regions),
    axis=0,
).rename("mask_PRISM").copy()


#zip+file://./PRISM/PRISM_ppt_stable_4kmD1_19810101_bil.zip!/PRISM_ppt_stable_4kmD1_19810101_bil.bil


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

for i, region in enumerate(regions):

    print(f"Making mask for region: {region:s}")

    polygon = polygon_dict[region] 
 
    print("Make PRISM mask with shape...")
    _mask_PRISM = make_mask_with_shape(llon_PRISM, llat_PRISM, polygon)
    _mask_PRISM[ocn_idx_PRISM] = 0
    mask_PRISM[i, :, :] = _mask_PRISM

    """
    print("Make ERA5 mask with shape...")
    _mask_ERA5 = make_mask_with_shape(llon_ERA5, llat_ERA5, polygon)
    _mask_ERA5[ocn_idx_ERA5] = 0
    mask_ERA5[i, :, :] = _mask_ERA5
    


    print("Make WRF mask with shape...")
    _mask_WRF = make_mask_with_shape(llon_WRF, llat_WRF, polygon)
    _mask_WRF[ocn_idx_WRF] = 0
    mask_WRF[i, :, :] = _mask_WRF
    print("Done.")
    """

#new_ds = xr.merge([mask_PRISM, wgt_PRISM, mask_WRF, wgt_WRF, mask_ERA5])
new_ds = xr.merge([mask_PRISM, wgt_PRISM])

print("Output file: ", args.output)
new_ds.to_netcdf(args.output)






