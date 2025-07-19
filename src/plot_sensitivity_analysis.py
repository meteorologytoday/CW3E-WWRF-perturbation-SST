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
        
        input_file = Path(details["input_file"])
        output_root = Path(details["output_root"])
        output_label = details["output_label"]
        modes = details["modes"]
        regions = details["regions"]
        last_modes = details["last_modes"]        
        output_dir = output_root / output_label 

        output_types = [
            "variance_map2",
            "variance_map",
            "singular_values",
            "matrix_fTf",
        ]


        output_files = {
            output_type : output_dir / "{output_type:s}.{ext:s}".format(
                output_type = output_type,
                ext = args.extension,
            )
            for output_type in output_types
        }

        for mode in range(1, modes+1):
            output_files["mode-%02d" % mode] = output_dir / "modes" / "mode-{mode:02d}.{ext:s}".format(
               ext = args.extension,
                mode = mode,
            )
        
        for region in regions:
            for output_type in ["wgt", "greenfunc", "tradeoff"]:
                output_files["%s-%s" % (output_type, region)] = output_dir / "{output_type:s}-{region:s}.{ext:s}".format(
                    region = region,
                    output_type = output_type,
                    ext = args.extension,
                )
     
           
        # Detecting
        result["output_files"] = output_files
        # First round is just to decide which files
        # to be processed to enhance parallel job 
        # distribution. I use variable `phase` to label
        # this stage.
        if detect_phase is True:

            all_exists = np.all( [ output_file.exists() for _, output_file in output_files.items() ] )

            result['need_work'] = not all_exists

            if result['need_work']:
                result['status'] = 'OK'
            else:           
                result['status'] = 'OK'
                result['need_work'] = False

            return result

        for _, output_file in output_files.items():

            output_file.parent.mkdir(parents=True, exist_ok=True)


        ds = xr.open_dataset(input_file)
        rank = ds.attrs["rank"] 

        #ds = ds.sel(region=region)

        # Figure : trade-off curve

        
        for region in regions:
            print("[%s] Plotting Trade-off Curve..." % (output_label,))
            output_file = output_files["tradeoff-%s" % (region,)]
            nrow, ncol = 1, 1 
            figsize, gridspec_kw = tool_fig_config.calFigParams(
                w = 6,
                h = 4,
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
                ),
                gridspec_kw=gridspec_kw,
                constrained_layout=False,
                squeeze=True,
                sharex=False,
            )
            
            unnormalized_tradeoff = ds["tradeoff"].sel(region=region).to_numpy()[:rank]
            normalized_tradeoff = unnormalized_tradeoff / unnormalized_tradeoff[0]

            ax.plot(np.arange(rank) + 1, normalized_tradeoff, "k-", marker='o', markersize=5)
            fig.suptitle(output_label)
            ax.set_title("Trade-off Curve")

            ax.grid(True)
            ax.set_xlabel("Mode") 
            print("Saving output: ", output_file)
            fig.savefig(output_file, dpi=200)


        # wgt
        for region in regions:

            print("[%s] Plotting wgt - %s..." % (output_label, region))
            output_file = output_files["wgt-%s" % (region,)]
            nrow, ncol = 1, 1 
            figsize, gridspec_kw = tool_fig_config.calFigParams(
                w = 6,
                h = 4,
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
                ),
                gridspec_kw=gridspec_kw,
                constrained_layout=False,
                squeeze=True,
                sharex=False,
            )

            ax.plot(np.arange(rank) + 1, ds["G_wgt"].sel(region=region).to_numpy()[:rank], "k-", marker='o', markersize=5)
            fig.suptitle(output_label)
            ax.set_title("Weight $w = \\left( \\Sigma \\Sigma^T \\right)^+ \\Sigma U^T f y^T $")

            ax.grid(True)
            ax.set_xlabel("Mode") 
            print("Saving output: ", output_file)
            fig.savefig(output_file, dpi=200)






















        # Figure : singular values
        print("[%s] Plotting singular values..." % (output_label,))
        output_file = output_files["singular_values"]
        nrow, ncol = 1, 1 
        figsize, gridspec_kw = tool_fig_config.calFigParams(
            w = 6,
            h = 4,
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
            ),
            gridspec_kw=gridspec_kw,
            constrained_layout=False,
            squeeze=True,
            sharex=False,
        )

        ax.plot(np.arange(rank) + 1, ds["SIGMA"].to_numpy()[:rank], "k-", marker='o', markersize=5)
        fig.suptitle(output_label)
        ax.set_title("Signular values")

        ax.grid(True)
        ax.set_xlabel("Mode") 
        print("Saving output: ", output_file)
        fig.savefig(output_file, dpi=200)


        # wgt
        for region in regions:

            print("[%s] Plotting wgt - %s..." % (output_label, region))
            output_file = output_files["wgt-%s" % (region,)]
            nrow, ncol = 1, 1 
            figsize, gridspec_kw = tool_fig_config.calFigParams(
                w = 6,
                h = 4,
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
                ),
                gridspec_kw=gridspec_kw,
                constrained_layout=False,
                squeeze=True,
                sharex=False,
            )

            ax.plot(np.arange(rank) + 1, ds["G_wgt"].sel(region=region).to_numpy()[:rank], "k-", marker='o', markersize=5)
            fig.suptitle(output_label)
            ax.set_title("Weight $w = \\left( \\Sigma \\Sigma^T \\right)^+ \\Sigma U^T f y^T $")

            ax.grid(True)
            ax.set_xlabel("Mode") 
            print("Saving output: ", output_file)
            fig.savefig(output_file, dpi=200)

        # Figure : matrix_fTf
 
        print("[%s] Plotting matrix_fTf.." % (output_label,))
        output_file = output_files["matrix_fTf"]
        nrow, ncol = 1, 1 
        figsize, gridspec_kw = tool_fig_config.calFigParams(
            w = 6,
            h = 6,
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
            ),
            gridspec_kw=gridspec_kw,
            constrained_layout=False,
            squeeze=True,
            sharex=False,
        )

        ax.matshow(ds["fTf"].to_numpy())
        fig.suptitle(output_label)
        ax.set_title("Matrix $ f^T f $")

        ax.set_xlabel("Ens") 
        ax.set_ylabel("Ens")
         
        print("Saving output: ", output_file)
        fig.savefig(output_file, dpi=200)


        # Figure : green function

        for region in regions:

            print("[%s] Plotting greenfunc - %s..." % (output_label, region))
            output_file = output_files["greenfunc-%s" % (region,)]

            
            ncol = 1
            nrow = 1
            
            lon_rng = np.array(args.lon_rng, dtype=float)
            lat_rng = np.array(args.lat_rng, dtype=float)
            
            plot_lon_l, plot_lon_r = lon_rng
            plot_lat_b, plot_lat_t = lat_rng
            
            lon_span = lon_rng[1] - lon_rng[0]
            lat_span = lat_rng[1] - lat_rng[0]

            h = 4.0
            w_map = h * lon_span / lat_span

            proj = ccrs.PlateCarree(central_longitude=180)
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
                squeeze=True,
                sharex=False,
            )

            lat = ds.coords["lat"]
            lon = ds.coords["lon"] % 360
            
            d = ds["GT"].sel(region=region).to_numpy()
            
            # This is just for plotting to regulate the range
            d /= 2 * np.nanstd(d)


            #d[d==0] = np.nan
            levs = np.linspace(-1, 1, 21)
            
            mappable = ax.contourf(
                lon, lat,
                d,
                levs,
                cmap=cmo.cm.balance,
                transform=map_transform,
                extend="both",
            )

            cax = tool_fig_config.addAxesNextToAxes(fig, ax, "right", thickness=0.03, spacing=0.05)
            cb = plt.colorbar(mappable, cax=cax, orientation="vertical", pad=0.00)
                        
            for __ax in [ax,]:

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


            fig.suptitle("%s" % (output_label,))
            
            print("Saving output: ", output_file) 
            fig.savefig(output_file, dpi=200)

 
        # Figure : variance_map
        print("[%s] Plotting variance map..." % (output_label,))
        output_file = output_files["variance_map"]
        
        ncol = 1
        nrow = 1
        
        lon_rng = np.array(args.lon_rng, dtype=float)
        lat_rng = np.array(args.lat_rng, dtype=float)
        
        plot_lon_l, plot_lon_r = lon_rng
        plot_lat_b, plot_lat_t = lat_rng
        
        lon_span = lon_rng[1] - lon_rng[0]
        lat_span = lat_rng[1] - lat_rng[0]

        h = 4.0
        w_map = h * lon_span / lat_span

        proj = ccrs.PlateCarree(central_longitude=180)
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
            squeeze=True,
            sharex=False,
        )

        lat = ds.coords["lat"]
        lon = ds.coords["lon"] % 360
        
        d = ds["U"].sel(mode=slice(0, rank)).std(dim="mode").to_numpy()
        
        # This is just for plotting to regulate the range
        d /= np.quantile(d, q=1.0)
        d[d==0] = np.nan
   
        levs = np.linspace(0, 1, 11)
        
        mappable = ax.contourf(
            lon, lat,
            d,
            levs,
            cmap=cmo.cm.thermal_r,
            transform=map_transform,
            extend="max",
        )

        cax = tool_fig_config.addAxesNextToAxes(fig, ax, "right", thickness=0.03, spacing=0.05)
        cb = plt.colorbar(mappable, cax=cax, orientation="vertical", pad=0.00)
                    
        for __ax in [ax,]:

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


        fig.suptitle("%s" % (output_label,))
        
        print("Saving output: ", output_file) 
        fig.savefig(output_file, dpi=200)


      
        
        # Figure : variance_map : last N modes
        print("[%s] Plotting variance map..." % (output_label,))
        output_file = output_files["variance_map2"]
        
        ncol = 1
        nrow = 1
        
        lon_rng = np.array(args.lon_rng, dtype=float)
        lat_rng = np.array(args.lat_rng, dtype=float)
        
        plot_lon_l, plot_lon_r = lon_rng
        plot_lat_b, plot_lat_t = lat_rng
        
        lon_span = lon_rng[1] - lon_rng[0]
        lat_span = lat_rng[1] - lat_rng[0]

        h = 4.0
        w_map = h * lon_span / lat_span

        proj = ccrs.PlateCarree(central_longitude=180)
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
            squeeze=True,
            sharex=False,
        )

        lat = ds.coords["lat"]
        lon = ds.coords["lon"] % 360
        
        d = ds["U"].sel(mode=slice(rank - last_modes, rank)).std(dim="mode").to_numpy()
        
        # This is just for plotting to regulate the range
        d /= np.quantile(d, q=1.0)
        d[d==0] = np.nan
   
        levs = np.linspace(0, 1, 11)
        
        mappable = ax.contourf(
            lon, lat,
            d,
            levs,
            cmap=cmo.cm.thermal_r,
            transform=map_transform,
            extend="max",
        )

        cax = tool_fig_config.addAxesNextToAxes(fig, ax, "right", thickness=0.03, spacing=0.05)
        cb = plt.colorbar(mappable, cax=cax, orientation="vertical", pad=0.00)
                    
        for __ax in [ax,]:

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


        fig.suptitle("%s" % (output_label,))
        
        print("Saving output: ", output_file) 
        fig.savefig(output_file, dpi=200)


        # modes

        for mode in range(1, modes+1):

            print("[%s] Plotting mode-%02d map..." % (output_label, mode,))
            output_file = output_files["mode-%02d" % mode]
            
            ncol = 1
            nrow = 1
            
            lon_rng = np.array(args.lon_rng, dtype=float)
            lat_rng = np.array(args.lat_rng, dtype=float)
            
            plot_lon_l, plot_lon_r = lon_rng
            plot_lat_b, plot_lat_t = lat_rng
            
            lon_span = lon_rng[1] - lon_rng[0]
            lat_span = lat_rng[1] - lat_rng[0]

            h = 4.0
            w_map = h * lon_span / lat_span

            proj = ccrs.PlateCarree(central_longitude=180)
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
                squeeze=True,
                sharex=False,
            )

            lat = ds.coords["lat"]
            lon = ds.coords["lon"] % 360
            
            d = ds["U"].sel(mode=mode).to_numpy()
            
            # This is just for plotting to regulate the range
            d /= 2 * np.std(d)
            d[d==0] = np.nan
      
            levs = np.linspace(-1, 1, 21)
            
            mappable = ax.contourf(
                lon, lat,
                d,
                levs,
                cmap=cmo.cm.balance,
                transform=map_transform,
                extend="both",
            )

            cax = tool_fig_config.addAxesNextToAxes(fig, ax, "right", thickness=0.03, spacing=0.05)
            cb = plt.colorbar(mappable, cax=cax, orientation="vertical", pad=0.00)
                        
            for __ax in [ax,]:

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


            fig.suptitle("%s. mode = %d" % (output_label, mode))
            
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
    parser.add_argument('--input-file', type=str, help='Input CRPS diag files. The first one will be treated as reference', required=True)
    parser.add_argument('--output-root', type=str, help='Input CRPS diag files. The first one will be treated as reference', required=True)
    parser.add_argument('--output-label', type=str, help='Input CRPS diag files. The first one will be treated as reference', required=True)
    parser.add_argument('--regions', type=str, nargs="+", help='Input CRPS diag files. The first one will be treated as reference', required=True)
    parser.add_argument('--extension', type=str, help='analysis beg time', default="svg")
    parser.add_argument('--lat-rng', type=float, nargs=2, help="Latitude range for plotting", default=[-90.0, 90.0])
    parser.add_argument('--lon-rng', type=float, nargs=2, help="Latitude range for plotting", default=[0.0, 360.0])
    parser.add_argument('--nproc', type=int, help="Number of processors", default=1)
    
    parser.add_argument('--modes', type=int, help="Number of modes to plot", default=4)
    parser.add_argument('--last-modes', type=int, help="Number of modes to plot", default=10)
    
    
    args = parser.parse_args()
    
    print(args)
    

    
    failed_dates = []
    input_args = []
    
    details = dict(
        input_file = args.input_file,
        output_root = args.output_root,
        output_label = args.output_label,
        modes = args.modes,
        last_modes = args.last_modes,
        regions = args.regions,
    )

    print("[Detect] Checking ")
    
    result = doJob(details, detect_phase=True)
    
    if not result['need_work']:
        print("File `%s` already exist. Skip it." % (str(result['output_files']),))
    else: 
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
