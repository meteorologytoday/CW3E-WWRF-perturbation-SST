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

def genStat(da):

    
    name = da.name
    merge_data = []
    merge_data.append(da.std(dim="ens").expand_dims(dim={"stat": np.array(["std",])}, axis=1))
    merge_data.append(da.mean(dim="ens").expand_dims(dim={"stat": np.array(["mean",])}, axis=1))
    merge_data.append(da.count(dim="ens").expand_dims(dim={"stat": np.array(["count",])}, axis=1))
    
    new_da = xr.merge(merge_data)[name]

    return new_da







plot_infos = {

    "PSFC" : dict(
        selector = None,
        mean = dict(
            full = dict(levs=np.arange(950, 1050, 5), cmap="rainbow"),
            anom = dict(levs=np.linspace(-1, 1, 21) * 2, cmap=cmocean.cm.balance),
        ),
        std = dict(
            full = dict(levs=np.linspace(0, 1, 21) * 50,  cmap=cmocean.cm.rain),
            anom = dict(levs=np.linspace(-1, 1, 11) * 5, cmap=cmocean.cm.balance),
        ),

        label = "$P_\\mathrm{sfc}$",
        unit = "hPa",
        factor = 1e2,
    ), 

    "TTL_RAIN" : dict(
        selector = None,
        mean = dict(
            full = dict(levs=np.linspace(0, 1, 21) * 100, cmap=cmocean.cm.rain),
            anom = dict(levs=np.linspace(-1, 1, 11) * 10, cmap=cmocean.cm.balance),
        ),
        std = dict(
            full = dict(levs=np.linspace(0, 1, 21) * 50,  cmap=cmocean.cm.rain),
            anom = dict(levs=np.linspace(-1, 1, 11) * 5, cmap=cmocean.cm.balance),
        ),
        anom_quantile = dict(levs=np.linspace(-1, 1, 21) * 30, cmap=cmocean.cm.balance),
        label = "$ \\mathrm{ACC}_{\\mathrm{ttl}}$",
        unit = "mm",
        cmap = cmocean.cm.balance,
    ), 

    "SST" : dict(
        selector = None,
        mean = dict(
            full = dict(levs=np.arange(0, 30, 5), cmap=cmocean.cm.rain),
            anom = dict(levs=np.linspace(-1, 1, 21) * 2, cmap=cmocean.cm.balance),
        ),
        std = dict(
            full = dict(levs=np.linspace(0, 1, 5), cmap=cmocean.cm.rain),
            anom = dict(levs=np.linspace(-1, 1, 21), cmap=cmocean.cm.balance),
        ),
        label = "SST",
        unit = "K",
        offset = 273.15,
        cmap = cmocean.cm.balance,
    ),
 
    "T2" : dict(
        selector = None,
        full = dict(levs=np.arange(0, 850, 50), cmap=cmocean.cm.rain),
        anom = dict(levs=np.linspace(-1, 1, 21) * 30, cmap=cmocean.cm.balance),
        label = "$T_\\mathrm{2m}$",
        unit = "$\\mathrm{kg} / \\mathrm{m} / \\mathrm{s} $",
        levs = np.linspace(-1, 1, 11) * 50,
        cmap = cmocean.cm.balance,
    ), 


    "IWV" : dict(
        selector = None,
        mean = dict(
            full = dict(levs=np.arange(0, 30, 5), cmap=cmocean.cm.rain),
            anom = dict(levs=np.linspace(-1, 1, 11) * 2, cmap=cmocean.cm.balance),
        ),
        std = dict(
            full = dict(levs=np.arange(0, 30, 5), cmap=cmocean.cm.rain),
            anom = dict(levs=np.linspace(-1, 1, 11) * 2, cmap=cmocean.cm.balance),
        ),

        label = "IWV",
        unit = "$\\mathrm{kg} / \\mathrm{m}^3 $",
        levs = np.linspace(-1, 1, 11) * 50,
        cmap = cmocean.cm.balance,
    ), 



    "IVT" : dict(
        selector = None,
        mean = dict(
            full = dict(levs=np.arange(0, 850, 50), cmap=cmocean.cm.rain, markerlevs=[500.0]),
            anom = dict(levs=np.linspace(-1, 1, 11) * 50, cmap=cmocean.cm.balance),
        ),
        std = dict(
            full = dict(levs=np.arange(0, 850, 50), cmap=cmocean.cm.rain, markerlevs=[500.0]),
            anom = dict(levs=np.linspace(-1, 1, 11) * 50, cmap=cmocean.cm.balance),
        ),
        label = "IVT",
        unit = "$\\mathrm{kg} / \\mathrm{m} / \\mathrm{s} $",
        levs = np.linspace(-1, 1, 11) * 50,
        cmap = cmocean.cm.balance,
    ), 

    "WND::850" : dict(
        selector = None,
        full = dict(levs=np.arange(0, 55, 5), cmap=cmocean.cm.rain, markerlevs=[40.0]),
        anom = dict(levs=np.linspace(-1, 1, 11) * 2, cmap=cmocean.cm.balance),
        label = "$ \\mathrm{WND}_{850}$",
        unit = "$\\mathrm{m} / \\mathrm{s}$",
    ), 

    "WND::200" : dict(
        selector = None,
        full = dict(levs=np.arange(0, 55, 5), cmap=cmocean.cm.rain, markerlevs=[40.0]),
        anom = dict(levs=np.linspace(-1, 1, 11) * 5, cmap=cmocean.cm.balance),
        label = "$ \\mathrm{WND}_{200}$",
        unit = "$\\mathrm{m} / \\mathrm{s}$",
    ), 



    "PH::850" : dict(
        selector = None,
        full = dict(levs=np.arange(1000, 2000, 50), cmap=cmocean.cm.rain, markerlevs=[500.0]),
        anom = dict(levs=np.linspace(-1, 1, 11) * 10, cmap=cmocean.cm.balance),
        label = "$ \\Phi_{850}$",
        unit = "$\\mathrm{m}^2 / \\mathrm{s}^2$",
        factor = 9.8,
    ), 

    "PH::200" : dict(
        selector = None,
        full = dict(levs=np.arange(10000, 12000, 50), cmap="rainbow"),
        anom = dict(levs=np.linspace(-1, 1, 11) * 50, cmap=cmocean.cm.balance),
        label = "$ \\Phi_{200}$",
        unit = "$\\mathrm{m}^2 / \\mathrm{s}^2$",
        factor = 9.8,
    ), 


}

def doJob(details, detect_phase=False):

    # phase \in ['detect', 'work']
    result = dict(details = details, status="UNKNOWN", need_work=False, detect_phase=detect_phase)

    try:
        
        varnames        = details["varnames"]
        input_root      = Path(details["input_root"])
        expblobs       = details["expblobs"]
        ref_expblob    = details["ref_expblob"]
        plot_quantile = details["plot_quantile"]
        
        pval       = details["pval"]
        qs         = details["quantiles"]        
        
        exp_beg_time    = pd.Timestamp(details["exp_beg_time"])
        plot_rel_time   = pd.Timedelta(hours=details["plot_rel_time"])
        
        output_root      = Path(details["output_root"]) 

        plot_time = exp_beg_time + plot_rel_time

        expblobs = [ WRF_ens_tools.parseExpblob(expblob) for expblob in expblobs ]

        full_names = []
        for expblob in expblobs:
            for expname, group, _ in expblob:
                full_names.append(f"{expname:s}-{group:s}") 

        output_types = ["mean", "std"]
        if plot_quantile:
            output_types.append("quantile")

        output_files = {
            output_type : output_root / "-".join(full_names) / "{output_type:s}_{varname:s}_{plot_time:s}.{ext:s}".format(
                output_type = output_type,
                varname = ",".join(varnames),
                plot_time = plot_time.strftime("%Y-%m-%dT%H_%M_%S"),
                ext = args.extension,
            )
            for output_type in output_types
        }


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

        data = dict()

        for varname in varnames:
        
            tmp = []

            for expblob in expblobs:
                da = WRF_ens_tools.loadExpblob(expblob, varname, plot_time, root=input_root)[varname]
                da = da.isel(time=0)
                tmp.append(da)

            """
            for expname, group, ens_rng in iter_obj:
                
                ens_ids = WRF_ens_tools.parseRanges(ens_rng)
                print("Ensemble ids: %s => %s" % (ens_ids, ",".join(["%d" % i for i in ens_ids] ) ))


                print("Load %s - %s" % (expname, group,)) 
                da = WRF_ens_tools.loadGroup(expname, group, ens_ids, varname, plot_time, root=input_root)[varname]
                da = da.isel(time=0)
                tmp.append(da)
            """


            data[varname] = tmp
        

        ncol = len(expblobs)
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


        for output_type in output_types:

            if output_type in ["mean", "std"]:

                output_file = output_files[output_type]

                stat = output_type

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
                    
                    _da_ref = _data[ref_expblob]
                   
                    factor = testIfIn(plot_info, "factor", 1) 
                    offset = testIfIn(plot_info, "offset", 0.0) 


                    _da_ref_stat = genStat(_da_ref)
                    
                    for j in range(len(_data)):
                        
                        _ax = axes[j]
                        _da = _data[j]

                        plot_ref = j == ref_expblob
                        
                        lat = _da_ref.coords["lat"]
                        lon = _da_ref.coords["lon"] % 360
                      
                        # Skip the reference 
                        #if plot_ref and da_idx == ref_expblob:
                        #    da_idx += 1
                         
                        if plot_ref:
                               
                            #print("varname = ", varname)
                            #print(plot_info) 
                            #print(plot_info[stat]["full"]["levs"])
                   
                            markerlevs = testIfIn(plot_info[stat]["full"], "markerlevs", None) 
                            _data_plot = ( _da_ref_stat.sel(stat=stat).to_numpy() - offset) / factor
                            mappable = _ax.contourf(
                                lon, lat,
                                _data_plot,
                                plot_info[stat]["full"]["levs"],
                                cmap=plot_info[stat]["full"]["cmap"],
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

                            #da_idx += 1
                            _da_stat = genStat(_da)
                            _da_stat_diff = _da_stat - _da_ref_stat

                            mappable = _ax.contourf(lon, lat, _da_stat_diff.sel(stat=stat).to_numpy() / factor, plot_info[stat]["anom"]["levs"], cmap="bwr", transform=proj, extend="both")
                            cax = tool_fig_config.addAxesNextToAxes(fig, _ax, "right", thickness=0.03, spacing=0.05)
                            cb = plt.colorbar(mappable, cax=cax, orientation="vertical", pad=0.00)
                            cb.ax.set_ylabel("[ %s ]" % (plot_info["unit"],))

                            _ax.set_title("Diff %s" % (plot_info["label"],))
 


                            # Plotting significancy

                            if stat == "mean":
                                mean1 = _da_ref_stat.sel(stat="mean").to_numpy()
                                std1  = _da_ref_stat.sel(stat="std").to_numpy()
                                nobs1 = _da_ref_stat.sel(stat="count").to_numpy()

                                mean2 = _da_stat.sel(stat="mean").to_numpy()
                                std2  = _da_stat.sel(stat="std").to_numpy()
                                nobs2 = _da_stat.sel(stat="count").to_numpy()

                                pval_test = scipy.stats.ttest_ind_from_stats(
                                    mean1, std1, nobs1,
                                    mean2, std2, nobs2,
                                    equal_var=True,
                                    alternative='two-sided'
                                )
                   
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

            if output_type == "quantile":
                
                output_file = output_files["quantile"]

                ncol = 1 + len(qs) * (len(iter_obj)-1)

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


                    _da_ref_stat = genStat(_da_ref)
         
                    ax_idx = 0
                    for j, _da in enumerate(_data):
                        
                        plot_ref = j== 0
                        
                        lat = _da_ref.coords["lat"]
                        lon = _da_ref.coords["lon"] % 360
                        
                        if plot_ref:
                            _ax = axes[ax_idx] ; ax_idx+=1
                    
                            markerlevs = testIfIn(plot_info["full"], "markerlevs", None) 
                            _data_plot = ( _da_ref_stat.sel(stat="mean").to_numpy() - offset) / factor
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

                            for q in qs:
                                
                                _ax = axes[ax_idx] ; ax_idx+=1
                                print("Plotting q = ", q)
                                _da_qstat_diff = (_da - _da_ref).chunk(dict(ens=-1)).quantile(q, dim="ens")

                                plotted_data = _da_qstat_diff / factor
                                #plotted_data = _da_qstat_diff.sel(quantile=q, drop=True) / factor

                                print(plotted_data)
                                anom_contourf_info = testIfIn(plot_info, "anom_quantile", plot_info["anom"])
                                
                                mappable = _ax.contourf(
                                    plotted_data.coords["lon"],
                                    plotted_data.coords["lat"], 
                                    plotted_data.to_numpy(), 
                                    anom_contourf_info["levs"],
                                    cmap=anom_contourf_info["cmap"], transform=proj, extend="both",
                                )
                                
                                cax = tool_fig_config.addAxesNextToAxes(fig, _ax, "right", thickness=0.03, spacing=0.05)
                                cb = plt.colorbar(mappable, cax=cax, orientation="vertical", pad=0.00)
                                cb.ax.set_ylabel("[ %s ]" % (plot_info["unit"],))

                                _ax.set_title("[q=%.2f] Diff %s" % (q, plot_info["label"],))
                    
                            
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
    parser.add_argument('--expblobs',  type=str, nargs="+", help='Expblobs.', required=True)
    parser.add_argument('--ref-expblob',  type=int, help='Reference idx of expblob.', default=0)
    parser.add_argument('--varnames',  type=str, nargs="+", help='Input directories.', required=True)

    parser.add_argument('--exp-beg-time', type=str, help='analysis beg time', required=True)
    parser.add_argument('--time-beg',    type=int, help='Beginning of time. Hours` after `--exp-beg-time`', required=True)
    parser.add_argument('--time-end',    type=int, help='Ending of time. Hours after `--exp-beg-time`', required=True)
    parser.add_argument('--time-stride', type=int, help='Each plot differ by this time interval. Hours', required=True)
    
    parser.add_argument('--lat-rng', type=float, nargs=2, help="Latitude range for plotting", default=[-90.0, 90.0])
    parser.add_argument('--lon-rng', type=float, nargs=2, help="Latitude range for plotting", default=[0.0, 360.0])
    parser.add_argument('--nproc', type=int, help="Number of processors", default=1)
    parser.add_argument('--pval', type=float, help="p-value", default=0.1)
    parser.add_argument('--quantiles', type=float, nargs="+", help="Quantiles", default=[0.25, 0.75])
    
    parser.add_argument('--plot-quantile', action="store_true", help="p-value")
    
    
    args = parser.parse_args()
    
    print(args)
    


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
            expblobs = args.expblobs,
            ref_expblob = args.ref_expblob,
            exp_beg_time = args.exp_beg_time,
            plot_rel_time = plot_rel_time,
            output_root = args.output_root,
            pval = args.pval,
            plot_quantile = args.plot_quantile,
            quantiles = args.quantiles,
        )

        print("[Detect] Checking hour=%d" % (plot_rel_time,))
        
        result = doJob(details, detect_phase=True)
        
        if not result['need_work']:
            print("File `%s` already exist. Skip it." % (str(result['output_files']),))
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
