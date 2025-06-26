from multiprocessing import Pool

import scipy
import xarray as xr
import traceback
import pandas as pd
import numpy as np
import argparse
import tool_fig_config
import wrf_load_helper 
import datetime
import os
import wrf_preprocess
import cmocean
import CRPS_tools
import geopandas as gpd
import WRF_ens_tools
from pathlib import Path
import re

print("Loading Plotting Modules: Matplotlib and Cartopy.")
import matplotlib as mpl
mpl.use('Agg')
mpl.rc('font', size=15)
mpl.rc('axes', labelsize=15)


import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Rectangle
import matplotlib.transforms as transforms
from matplotlib.dates import DateFormatter
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cmocean as cmo

import tool_fig_config
print("Done.")     

def testIfIn(di, key, default):
    
    return di[key] if (key in di) else default


def doJob(details, detect_phase=False):

    # phase \in ['detect', 'work']
    result = dict(details = details, status="UNKNOWN", need_work=False, detect_phase=detect_phase)

    try:
        
        input_file  = details["input_file"]
        title       = details["title"]
        output_root = Path(details["output_root"])
        ext         = details["ext"]


        ds = xr.open_dataset(input_file, engine="netcdf4")

        regions = ds.coords["region"].to_numpy()

        output_files = [
            output_root / ("region-%s.%s" % (region, ext))
            for region in regions
        ]


        # Detecting
        result["output_files"] = output_files

        
        # First round is just to decide which files
        # to be processed to enhance parallel job 
        # distribution. I use variable `phase` to label
        # this stage.
        if detect_phase is True:

            result['need_work'] = not np.all([output_file.exists() for output_file in output_files ])

            if result['need_work']:
                result['status'] = 'OK'
            else:           
                result['status'] = 'OK'
                result['need_work'] = False

            return result

        #for _, output_file in output_files.items():

        #    output_file.parent.mkdir(parents=True, exist_ok=True)


        coords = ds.coords
        
        lat_bnds = coords["lat_regrid_bnd"].to_numpy()
        lon_bnds = coords["lon_regrid_bnd"].to_numpy() % 360.0

        for i, region in enumerate(regions):

            print("Plotting region: ", region)
            output_file = output_files[i]
            _ds = ds.sel(region=region)            

            ncol = 2
            nrow = 1
            
            lon_rng = np.array(args.lon_rng, dtype=float)
            lat_rng = np.array(args.lat_rng, dtype=float)
            
            plot_lon_l, plot_lon_r = lon_rng
            plot_lat_b, plot_lat_t = lat_rng
            
            lon_span = lon_rng[1] - lon_rng[0]
            lat_span = lat_rng[1] - lat_rng[0]
            
            h = 4.0
            w_map = h * lon_span / lat_span
            cent_lon=180.0

            proj = ccrs.PlateCarree(central_longitude=cent_lon)
            map_transform = ccrs.PlateCarree()

            figsize, gridspec_kw = tool_fig_config.calFigParams(
                w = w_map,
                h = h,
                wspace = 1.5,
                hspace = 1.0,
                w_left = 1.0,
                w_right = 1.5,
                h_bottom = 1.5,
                h_top = 1.0,
                ncol = ncol,
                nrow = nrow,
            )
            
            fig, ax = plt.subplots(
                nrow, ncol,
                figsize=figsize,
                subplot_kw=dict(
                    aspect="auto",
                    projection=proj,
                ),
                gridspec_kw=gridspec_kw,
                constrained_layout=False,
                squeeze=False,
                sharex=False,
            )
            
            ax_flatten = ax.flatten()

            print("lat_bnds:" , lat_bnds) 
            print("lon_bnds:" , lon_bnds) 
            _ax = ax_flatten[0]
                
            _shading = _ds["fyT"].to_numpy()
            value_std = np.nanstd(_shading)
            mappable = _ax.pcolormesh(
            #mappable = _ax.contourf(
                lon_bnds,
                lat_bnds,
                _shading,
                vmin = - 4*value_std, #np.nanmin(_shading),
                vmax = 4*value_std,  #np.nanmax(_shading),
                cmap = "bwr",
                transform = map_transform,
            )

            cax = tool_fig_config.addAxesNextToAxes(fig, _ax, "right", thickness=0.03, spacing=0.05)
            cb = plt.colorbar(mappable, cax=cax, orientation="vertical", pad=0.00)
            
            _ax.set_title("(a) Sensitivity $ f \\cdot y^T $ ") 

            _ax = ax_flatten[1]
            _shading = _ds["GreenFunc"].to_numpy()
            value_std = np.nanstd(_shading)
            mappable = _ax.pcolormesh(
            #mappable = _ax.contourf(
                lon_bnds,
                lat_bnds,
                _shading,
                vmin = - 4*value_std,
                vmax = 4*value_std,
                cmap = "bwr",
                transform = map_transform,
            )
            cax = tool_fig_config.addAxesNextToAxes(fig, _ax, "right", thickness=0.03, spacing=0.05)
            cb = plt.colorbar(mappable, cax=cax, orientation="vertical", pad=0.00)
            
            _ax.set_title("(b) Green's Function") 
 
            for __ax in ax.flatten(): 

                __ax.set_global()
                #__ax.gridlines()
                __ax.coastlines(color='gray')
                __ax.set_extent([plot_lon_l, plot_lon_r, plot_lat_b, plot_lat_t], crs=map_transform)

                gl = __ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                                  linewidth=1, color='gray', alpha=0.5, linestyle='--')

                gl.xlabels_top   = False
                gl.ylabels_right = False

                #gl.xlocator = mticker.FixedLocator([120, 150, 180, -150, -120])#np.arange(-180, 181, 30))
                #gl.ylocator = mticker.FixedLocator([10, 20, 30, 40, 50])
                
                gl.xformatter = LONGITUDE_FORMATTER
                gl.yformatter = LATITUDE_FORMATTER
                gl.xlabel_style = {'size': 12, 'color': 'black'}
                gl.ylabel_style = {'size': 12, 'color': 'black'}

                """
                for shp_name, shp in shps.items():
                    print("Putting geometry %s" % (shp_name,))
                    print(shp)
                    __ax.add_geometries(
                        shp["geometry"], crs=map_transform, facecolor="none", edgecolor="black"
                    )
                """

            fig.suptitle("%s : %s" % (title, region))
            
            print("Saving output: ", output_file) 
            fig.savefig(output_file, dpi=200)

            plt.close(fig)

        result['status'] = 'OK'

    except Exception as e:

        result['status'] = 'ERROR'
        #traceback.print_stack()
        traceback.print_exc()
        print(e)


    return result




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--title', type=str, help='Input sensitivity file', required=True)
    parser.add_argument('--input-file', type=str, help='Input sensitivity file', required=True)
    parser.add_argument('--output-root', type=str, help='Input CRPS diag files. The first one will be treated as reference', required=True)
    parser.add_argument('--extension', type=str, help='analysis beg time', default="svg")
    parser.add_argument('--lat-rng', type=float, nargs=2, help="Latitude range for plotting", default=[-90.0, 90.0])
    parser.add_argument('--lon-rng', type=float, nargs=2, help="Latitude range for plotting", default=[0.0, 360.0])
    parser.add_argument('--nproc', type=int, help="Number of processors", default=1)
    
    args = parser.parse_args()
    
    print(args)
    

    input_args = []
    failed_dates = []
    for input_file in [args.input_file,]:
        
        details = dict(
            title = args.title,
            input_file = input_file,
            output_root = args.output_root,
            ext = args.extension,
        )

        print("[Detect] Checking input_file = %s" % (input_file,))
        
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
                print('!!! Failed to generate output file %s.' % (str(result['output_files']),))
                failed_dates.append(result['details'])


    print("Tasks finished.")

    print("Failed output files: ")
    for i, failed_detail in enumerate(failed_dates):
        print("%d : " % (i+1), failed_detail)

    print("Done.")    
