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


plot_infos = dict(

    PSFC = dict(
        selector = None,
        full = dict(levs=np.arange(950, 1050, 5), cmap="rainbow"),
        anom = dict(levs=np.linspace(-1, 1, 21) * 2, cmap=cmocean.cm.balance),
        label = "$P_\\mathrm{sfc}$",
        unit = "hPa",
        factor = 1e2,
    ), 

    TTL_RAIN = dict(
        selector = None,
        full = dict(levs=np.arange(0, 100, 5), cmap=cmocean.cm.rain),
        anom = dict(levs=np.linspace(-1, 1, 11) * 15, cmap=cmocean.cm.balance),
        label = "$ \\mathrm{ACC}_{\\mathrm{ttl}}$",
        unit = "mm",
        cmap = cmocean.cm.balance,
    ), 

    SST = dict(
        selector = None,
        full = dict(levs=np.arange(0, 30, 5), cmap=cmocean.cm.rain),
        anom = dict(levs=np.linspace(-1, 1, 21) * 2, cmap=cmocean.cm.balance),
        label = "SST",
        unit = "K",
        offset = 273.15,
        levs = np.linspace(-1, 1, 11) * 2,
        cmap = cmocean.cm.balance,
    ),
 
    SSTSK = dict(
        selector = None,
        full = dict(levs=np.arange(0, 30, 5), cmap=cmocean.cm.rain),
        anom = dict(levs=np.linspace(-1, 1, 21) * 2, cmap=cmocean.cm.balance),
        label = "SSTSK",
        unit = "K",
        offset = 273.15,
        levs = np.linspace(-1, 1, 11) * 2,
        cmap = cmocean.cm.balance,
    ),

    T2 = dict(
        selector = None,
        full = dict(levs=np.arange(0, 850, 50), cmap=cmocean.cm.rain),
        anom = dict(levs=np.linspace(-1, 1, 21) * 30, cmap=cmocean.cm.balance),
        label = "$T_\\mathrm{2m}$",
        unit = "$\\mathrm{kg} / \\mathrm{m} / \\mathrm{s} $",
        levs = np.linspace(-1, 1, 11) * 50,
        cmap = cmocean.cm.balance,
    ), 



    IVT = dict(
        selector = None,
        full = dict(levs=np.arange(0, 850, 50), cmap=cmocean.cm.rain, markerlevs=[500.0]),
        anom = dict(levs=np.linspace(-1, 1, 11) * 50, cmap=cmocean.cm.balance),
        label = "IVT",
        unit = "$\\mathrm{kg} / \\mathrm{m} / \\mathrm{s} $",
        levs = np.linspace(-1, 1, 11) * 50,
        cmap = cmocean.cm.balance,
    ), 





    convective_precipitation = dict(
        selector = None,
        label = "Daily convective precip",
        unit = "$\\mathrm{mm}$",
        factor = 1e3,
        lim = [0, 30],
        #levs = np.linspace(-1, 1, 11) * 50,
        #cmap = cmocean.cm.balance,
    ), 


    large_scale_precipitation = dict(
        selector = None,
        label = "Daily large-scale precip",
        unit = "$\\mathrm{mm}$",
        factor = 1e3,
        lim = [0, 30],
        #levs = np.linspace(-1, 1, 11) * 50,
        #cmap = cmocean.cm.balance,
    ), 


    total_precipitation = dict(
        selector = None,
        label = "Daily total precip",
        unit = "$\\mathrm{mm}$",
        factor = 1e3,
        lim = [0, 30],
        #levs = np.linspace(-1, 1, 11) * 50,
        #cmap = cmocean.cm.balance,
    ), 



    SST_NOLND = dict(
        selector = None,
        label = "SST",
        unit = "K",
        levs = np.linspace(-1, 1, 11) * 2,
        cmap = cmocean.cm.balance,
    ), 


    PH850 = dict(
        selector = None,
        label = "$ \\Phi_{850}$",
        unit = "$\\mathrm{m}^2 / \\mathrm{s}^2$",
        levs = np.linspace(-1, 1, 11) * 50,
        cmap = cmocean.cm.balance,
    ), 


    PH500 = dict(
        selector = None,
        label = "$ \\Phi_{500}$",
        unit = "$\\mathrm{m}^2 / \\mathrm{s}^2$",
        levs = np.linspace(-1, 1, 11) * 200,
        cmap = cmocean.cm.balance,
    ), 



    TA = dict(
        selector = dict(bottom_top=0),
        wrf_varname = "T",
        label = "$\\Theta_{A}$",
        unit = "K",
    ), 

    TOA = dict(
        wrf_varname = "TOA",
        label = "$\\Theta_{OA}$",
        unit = "K",
    ), 

    QOA = dict(
        wrf_varname = "QOA",
        label = "$Q_{OA}$",
        unit = "g / kg",
    ), 

    CH = dict(
        wrf_varname = "CH",
        label = "$C_{H}$",
    ), 

    CQ = dict(
        wrf_varname = "CQ",
        label = "$C_{Q}$",
    ), 

    UA = dict(
        selector = dict(bottom_top=0),
        wrf_varname = "U",
        label = "$u_{A}$",
        unit = "$ \\mathrm{m} \\, / \\, \\mathrm{s}$",
    ), 

    VA = dict(
        selector = dict(bottom_top=0),
        wrf_varname = "V",
        label = "$v_{A}$",
        unit = "$ \\mathrm{m} \\, / \\, \\mathrm{s}$",
    ), 

)


def doJob(details, detect_phase=False):

    # phase \in ['detect', 'work']
    result = dict(details = details, status="UNKNOWN", need_work=False, detect_phase=detect_phase)

    try:
        
        varnames        = details["varnames"]
        input_root      = Path(details["input_root"])
        expnames        = details["expnames"]
        groups          = details["groups"]
        subgroups       = details["subgroups"]
        
        pval       = details["pval"]


        exp_beg_time    = pd.Timestamp(details["exp_beg_time"])
        plot_rel_time   = pd.Timedelta(hours=details["plot_rel_time"])
        
        output_root      = Path(details["output_root"]) 

        plot_time = exp_beg_time + plot_rel_time

        iter_obj = list(zip(expnames, groups, subgroups))

        full_names = []
        for expname, group, subgroup in iter_obj:
            full_names.append(f"{expname:s}-{group:s}-{subgroup}") 

        output_file = output_root / "-".join(full_names) / "{varname:s}_{plot_time:s}.{ext:s}".format(
            varname = ",".join(varnames),
            plot_time = plot_time.strftime("%Y-%m-%dT%H_%M_%S"),
            ext = args.extension,
        )

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

        output_file.parent.mkdir(parents=True, exist_ok=True)

        data = dict()

        for varname in varnames:
            
            tmp = []
            for expname, group, subgroup in iter_obj:

                load_file = input_root / expname / group / subgroup / "{varname:s}-{time:s}.nc".format(
                    varname = varname,
                    time = plot_time.strftime("%Y-%m-%dT%H:%M:%S"),
                )

                da = xr.open_dataset(load_file)[varname]
                da = da.isel(time=0)
                tmp.append(da)
                     

            data[varname] = tmp

        ncol = len(iter_obj)
        nrow = len(varnames)

        lon_rng = np.array(args.lon_rng, dtype=float)
        lat_rng = np.array(args.lat_rng, dtype=float)

        plot_lon_l, plot_lon_r = lon_rng
        plot_lat_b, plot_lat_t = lat_rng

        lon_span = lon_rng[1] - lon_rng[0]
        lat_span = lat_rng[1] - lat_rng[0]

        h = 4.0
        w_map = h * lon_span / lat_span
        cent_lon=0.0

        proj = ccrs.PlateCarree(central_longitude=cent_lon)
        map_transform = ccrs.PlateCarree()

        def plot_hatch(ax, x, y, d, threshold, hatch="."):

            _dot = np.zeros_like(d)
            _significant_idx =  (d < threshold ) 
            _dot[ _significant_idx                 ] = 0.75
            _dot[ np.logical_not(_significant_idx) ] = 0.25
            cs = ax.contourf(x, y, _dot, colors='none', levels=[0, 0.5, 1], hatches=[None, hatch], transform=map_transform)

            # Remove the contour lines for hatches 
            for _, collection in enumerate(cs.collections):
                collection.set_edgecolor((.2, .2, .2))
                collection.set_linewidth(0.)


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


        for i, varname in enumerate(varnames):

            plot_info = plot_infos[varname]
            axes = ax[i, :]
            _data = data[varname]
            
            _da_ref = _data[0]
           
            factor = testIfIn(plot_info, "factor", 1) 
            offset = testIfIn(plot_info, "offset", 0.0) 
 
            for j, _da in enumerate(_data):
                
                _ax = axes[j]
                
                is_ref = j==0
                
                lat = _da_ref.coords["lat"]
                lon = _da_ref.coords["lon"] % 360
                
                if is_ref:
            
                    markerlevs = testIfIn(plot_info["full"], "markerlevs", None) 
                    _data_plot = ( _da.sel(stat="mean").to_numpy() - offset) / factor
                    mappable = _ax.contourf(
                        lon, lat,
                        _data_plot,
                        plot_info["full"]["levs"],
                        cmap=plot_info["full"]["cmap"],
                        transform=proj,
                        extend="max",
                    )
                    
                    cax = tool_fig_config.addAxesNextToAxes(fig, _ax, "right", thickness=0.03, spacing=0.05)
                    cb = plt.colorbar(mappable, cax=cax, orientation="vertical", pad=0.00)
                    cb.ax.set_ylabel("[ %s ]" % (plot_info["unit"],))

                    if markerlevs is not None:
                        cs = _ax.contour(
                            lon, lat,
                            _data_plot,
                            markerlevs,
                            transform=proj,
                            colors="yellow",
                        )
     
                    #plot_hatch(_ax, lon, lat, rel_CRPS, 0.2, hatch="..")
                    #plot_hatch(_ax, lon, lat, -rel_CRPS, 0.2, hatch="//")
                    
                    #cs = _ax.contour(_ds.coords["lon"] % 360, _ds.coords["lat"], rel_CRPS, np.linspace(-1, 1, 5), transforms=proj, cmap="bwr")
                    
                    _ax.set_title("Ref %s" % (plot_info["label"],))

                else:
                    _da_diff = _da - _da_ref


                    mean1 = _da_ref.sel(stat="mean").to_numpy()
                    std1  = _da_ref.sel(stat="std").to_numpy()
                    nobs1 = _da_ref.sel(stat="count").to_numpy()

                    mean2 = _da.sel(stat="mean").to_numpy()
                    std2  = _da.sel(stat="std").to_numpy()
                    nobs2 = _da.sel(stat="count").to_numpy()

                    print(np.max(nobs1))
                    print(np.max(nobs2))

                    pval_test = scipy.stats.ttest_ind_from_stats(
                        mean1, std1, nobs1,
                        mean2, std2, nobs2,
                        equal_var=True,
                        alternative='two-sided'
                    )

                    mappable = _ax.contourf(lon, lat, _da_diff.sel(stat="mean").to_numpy() / factor, plot_info["anom"]["levs"], cmap="bwr", transform=proj, extend="both")
                    cax = tool_fig_config.addAxesNextToAxes(fig, _ax, "right", thickness=0.03, spacing=0.05)
                    cb = plt.colorbar(mappable, cax=cax, orientation="vertical", pad=0.00)
                    cb.ax.set_ylabel("[ %s ]" % (plot_info["unit"],))

                    _ax.set_title("Diff %s" % (plot_info["label"],))
        
                    plot_hatch(_ax, lon, lat, pval_test.pvalue, pval, hatch=".")
                    
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

        fig.suptitle("[pval=%.1f] Time: %s" % (
            args.pval,
            plot_time.strftime("%Y/%m/%d %H:%M:%S"),
        ))
        
        print("Saving output: ", output_file) 
        fig.savefig(output_file, dpi=200)


        result['status'] = 'OK'

    except Exception as e:

        result['status'] = 'ERROR'
        #traceback.print_stack()
        traceback.print_exc()
        print(e)


    return result




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--input-root', type=str, help='Input CRPS diag files. The first one will be treated as reference', required=True)
    parser.add_argument('--output-root', type=str, help='Input CRPS diag files. The first one will be treated as reference', required=True)
    parser.add_argument('--extension', type=str, help='analysis beg time', default="svg")
    parser.add_argument('--expnames',  type=str, nargs="+", help='Input directories.', required=True)
    parser.add_argument('--groups',    type=str, nargs="+", help='Input directories.', required=True)
    parser.add_argument('--subgroups', type=str, nargs="+", help='Input directories.', required=True)
    
    parser.add_argument('--varnames',  type=str, nargs="+", help='Input directories.', required=True)

    parser.add_argument('--exp-beg-time', type=str, help='analysis beg time', required=True)
    parser.add_argument('--time-beg',    type=int, help='Beginning of time. Hours` after `--exp-beg-time`', required=True)
    parser.add_argument('--time-end',    type=int, help='Ending of time. Hours after `--exp-beg-time`', required=True)
    parser.add_argument('--time-stride', type=int, help='Each plot differ by this time interval. Hours', required=True)
    
    parser.add_argument('--lat-rng', type=float, nargs=2, help="Latitude range for plotting", default=[-90.0, 90.0])
    parser.add_argument('--lon-rng', type=float, nargs=2, help="Latitude range for plotting", default=[0.0, 360.0])
    parser.add_argument('--nproc', type=int, help="Number of processors", default=1)
    parser.add_argument('--pval', type=float, help="p-value", default=0.1)
    
    
    args = parser.parse_args()
    
    print(args)
    
    # Check length
    if len(args.expnames) != len(args.groups) or len(args.expnames) != len(args.subgroups):
        raise Exception("Error: lengths of `--expnames`, `--groups`, `--subgroups` do not match.")
   
    """ 
    needed_shapes = dict(
        CA             = "data/shapefiles/ca_state.zip",
        Dam_Oroville   = "data/shapefiles/OrovilleDam.zip",
        Dam_Shasta     = "data/shapefiles/ShastaDam.zip",
        Dam_NewMelones = "data/shapefiles/NewMelonesDam.zip",
        Dam_SevenOaks  = "data/shapefiles/SevenOaksDam.zip",
    )


    print("Load CA shapefile")
    shps = {
        k : gpd.read_file(v).to_crs(epsg=4326) for k, v in needed_shapes.items()
    }
    
    """
    
    
    N = int((args.time_end - args.time_beg) / args.time_stride)
    
    failed_dates = []
    input_args = []
    
    for i in range(N):
        
        plot_rel_time = args.time_beg + i * args.time_stride 
        details = dict(
            varnames = args.varnames,
            input_root = args.input_root,
            expnames   = args.expnames,
            groups     = args.groups,
            subgroups  = args.subgroups,
            exp_beg_time = args.exp_beg_time,
            plot_rel_time = plot_rel_time,
            output_root = args.output_root,
            pval = args.pval,
        )

        print("[Detect] Checking hour=%d" % (plot_rel_time,))
        
        result = doJob(details, detect_phase=True)
        
        if not result['need_work']:
            print("File `%s` already exist. Skip it." % (result['output_file'],))
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
