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

def parseVarname(varname):
    varname = varname.strip()
    result = None
    m = re.match(r"^(?P<varname>[a-zA-Z0-9]+)(-(?P<pressure>[0-9]+\.?[0-9]*))?$", varname)
    if m is None:
        raise Exception("Cannot parse %s" % (varname,))
    else:
        result = m.groupdict()
        if result["pressure"] is not None:
            result["pressure"] = float(result["pressure"])
    return result 




plot_infos = {

    "TOmA" : dict(
        selector = None,
        mean = dict(
            full = dict(levs=np.linspace(-1, 1, 11) * 5, cmap=cmocean.cm.balance),
            anom = dict(levs=np.linspace(-1, 1, 11) * 1, cmap=cmocean.cm.balance),
        ),
        std = dict(
            full = dict(levs=np.linspace(0, 1, 21) * 50,  cmap=cmocean.cm.balance),
            anom = dict(levs=np.linspace(-1, 1, 11) * 5, cmap=cmocean.cm.balance),
        ),

        label = "$\\mathrm{SST} - T_{2\\mathrm{m}}$",
        unit = "K",
        factor = 1,
    ), 


    "PBLH" : dict(
        mean = dict(
            full = dict(levs=np.arange(0, 2001, 100), cmap="rainbow"),
            anom = dict(levs=np.linspace(-1, 1, 21) * 200, cmap=cmocean.cm.balance),
        ),
        std = dict(
            full = dict(levs=np.linspace(0, 1, 21) * 50,  cmap=cmocean.cm.rain),
            anom = dict(levs=np.linspace(-1, 1, 11) * 5, cmap=cmocean.cm.balance),
        ),

        label = "PBLH",
        unit = "m",
        factor = 1.0,
    ), 



    "PSFC" : dict(
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

    "QVAPOR" : dict(
        
        selector = None,
        mean = dict(
            full = dict(levs=np.linspace(0, 1, 11)[1:] * 10, cmap=cmocean.cm.ice_r, extend="max"),
            anom = dict(levs=np.linspace(-1, 1, 11) * 1, cmap=cmocean.cm.balance),
        ),
        
        std = dict(
            full = dict(levs=np.arange(0, 850, 50), cmap=cmocean.cm.rain, markerlevs=[500.0]),
            anom = dict(levs=np.linspace(-1, 1, 11) * 50, cmap=cmocean.cm.balance),
        ),
        label = "$Q_\\mathrm{vap}$",
        factor = 1e-3,
        unit = "$\\mathrm{g} / \\mathrm{kg} $",
    ), 


    "W" : dict(
        
        selector = None,
        mean = dict(
            full = dict(levs=np.linspace(-1, 1, 11) * 10, cmap=cmocean.cm.balance, extend="both"),
            anom = dict(levs=np.linspace(-1, 1, 11) * 5, cmap=cmocean.cm.balance),
        ),
        
        std = dict(
            full = dict(levs=np.arange(0, 850, 50), cmap=cmocean.cm.rain, markerlevs=[500.0]),
            anom = dict(levs=np.linspace(-1, 1, 11) * 50, cmap=cmocean.cm.balance),
        ),
        label = "W",
        factor=1e-2,
        unit = "$\\mathrm{cm} / \\mathrm{s} $",
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
       
        cxs_info = details["cross_section_info"] 
        unparsed_vars   = details["unparsed_vars"]
        input_root      = Path(details["input_root"])
        expblobs       = details["expblobs"]
        ref_expblob    = details["ref_expblob"]
        
        pval       = details["pval"]
        
        exp_beg_time    = pd.Timestamp(details["exp_beg_time"])
        plot_rel_time   = pd.Timedelta(hours=details["plot_rel_time"])
        
        output_root      = Path(details["output_root"]) 
       
        if len(unparsed_vars) > 2:
            raise Exception("Only two variables can be provided. First is shading, second is contour")

 
        plot_time = exp_beg_time + plot_rel_time
        
        expblobs = [ WRF_ens_tools.parseExpblob(expblob) for expblob in expblobs ]
        
        full_names = []
        for expblob in expblobs:
            for expname, group, _ in expblob:
                full_names.append(f"{expname:s}-{group:s}")
        
        output_types = ["mean",]
        output_files = {
            output_type : output_root / "-".join(full_names) / "{output_type:s}_{varname:s}_{plot_time:s}.{ext:s}".format(
                output_type = output_type,
                varname = ",".join(unparsed_vars),
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
       
        parsed_vars = [parseVarname(s) for s in unparsed_vars]
        for parsed_var in parsed_vars:
       
            print("Loading var: ", parsed_var)  
            tmp = []

            sel_pressure = parsed_var["pressure"] is not None

            for expblob in expblobs:
                da = WRF_ens_tools.loadExpblob(expblob, parsed_var["varname"], plot_time, root=input_root)
                da = da.isel(time=0)
                if sel_pressure:
                    da = da.sel(pressure=parsed_var["pressure"])
                

                # Slice cross section
                
                fxd_coord = None
                rng_coord = None
                if cxs_info["dir"] == "meridional":
                    fxd_coord = "lon"
                    rng_coord = "lat"
                    
                elif cxs_info["dir"] == "latitudinal":
                    fxd_coord = "lat"
                    rng_coord = "lon"

                #fxd_rng = cxs_info["loc_rng"]
                #da = da.sel(**{ fxd_coord : fxd_idx })
                da = da.where(
                    (da.coords[rng_coord] >= cxs_info["rng"][0])
                    & (da.coords[rng_coord] <= cxs_info["rng"][1])
                    & (da.coords[fxd_coord] >= cxs_info["loc_rng"][0])
                    & (da.coords[fxd_coord] <= cxs_info["loc_rng"][1])
                , drop=True).mean(dim=fxd_coord)
               
                print("da = ", da) 

                
                tmp.append(da)


            data[parsed_var["varname"]] = tmp
        

        ncol = len(expblobs)
        nrow = 1

        h = 4.0
        w_map = h * 1.6

        def plot_hatch(ax, x, y, d, threshold, hatch="."):

            _dot = np.zeros_like(d)
            _significant_idx =  (d < threshold ) 
            _dot[ _significant_idx                 ] = 0.75
            _dot[ np.logical_not(_significant_idx) ] = 0.25
            cs = ax.contourf(x, y, _dot, colors='none', levels=[0, 0.5, 1], hatches=[None, hatch])

            # Remove the contour lines for hatches 
            for _, collection in enumerate(cs.collections):
                collection.set_edgecolor((.2, .2, .2))
                collection.set_linewidth(0.)


        for output_type in output_types:

            if output_type in ["mean",]:

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
                    ),
                    gridspec_kw=gridspec_kw,
                    constrained_layout=False,
                    squeeze=False,
                    sharex=False,
                )

                for i, parsed_var in enumerate(parsed_vars):
                    
                    varname = parsed_var["varname"]

                    plot_type = None               
                    if i == 0 :
                        plot_type = "contourf"
                    elif i == 1:
                        plot_type = "contour" 
                    else:
                        print("Cannot plot this variable `%s` because too many are plotted" % (varname,))
                        continue


                    plot_info = plot_infos[varname]
                    _data = data[varname]
                    
                    _da_ref = _data[ref_expblob]
                   
                    factor = testIfIn(plot_info, "factor", 1) 
                    offset = testIfIn(plot_info, "offset", 0.0) 

                    _da_ref_stat = genStat(_da_ref)
                    
                    for j in range(len(_data)):
                        
                        _ax = ax[0, j]
                        _da = _data[j]

                        plot_ref = j == ref_expblob
                        
                        x = _da_ref.coords[rng_coord]
                        z = _da_ref.coords["pressure"]
                      
                        if plot_ref:
                            markerlevs = testIfIn(plot_info[stat]["full"], "markerlevs", None)



                            _data_plot = ( _da_ref_stat.sel(stat=stat).to_numpy() - offset) / factor
                           
                            extend = testIfIn(plot_info[stat]["full"], "extend", "max")


                            if plot_type == "contourf":
   
                                mappable = _ax.contourf(
                                    x, z,
                                    _data_plot,
                                    plot_info[stat]["full"]["levs"],
                                    cmap=plot_info[stat]["full"]["cmap"],
                                    extend=extend,
                                )
                                    
                                cax = tool_fig_config.addAxesNextToAxes(fig, _ax, "right", thickness=0.03, spacing=0.05)
                                cb = plt.colorbar(mappable, cax=cax, orientation="vertical", pad=0.00)
                                cb.ax.set_ylabel("[ %s ]" % (plot_info["unit"],))

                            elif plot_type == "contour":
   
                                cs = _ax.contour(
                                    x, z,
                                    _data_plot,
                                    plot_info[stat]["full"]["levs"],
                                    colors="black",
                                )
                                
                                _ax.clabel(cs)
                                    
                            
                            _ax.set_title("Ref %s" % (plot_info["label"],))

                        else:

                            #da_idx += 1
                            _da_stat = genStat(_da)
                            _da_stat_diff = _da_stat - _da_ref_stat

                            #print(_da_stat_diff)

                            _data_plot = _da_stat_diff.sel(stat=stat).to_numpy() / factor
                            
                            if plot_type == "contourf":
                                
                                mappable = _ax.contourf(
                                    x,
                                    z,
                                    _data_plot,
                                    plot_info[stat]["anom"]["levs"],
                                    cmap="bwr",
                                    extend="both",
                                )
                                
                                cax = tool_fig_config.addAxesNextToAxes(fig, _ax, "right", thickness=0.03, spacing=0.05)
                                cb = plt.colorbar(mappable, cax=cax, orientation="vertical", pad=0.00)
                                cb.ax.set_ylabel("[ %s ]" % (plot_info["unit"],))

                            elif plot_type == "contour":
                                
                                cs = _ax.contour(
                                    x,
                                    z,
                                    _data_plot,
                                    plot_info[stat]["anom"]["levs"],
                                    colors="black",
                                )
                                
                                _ax.clabel(cs)
 
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
                   
                                plot_hatch(_ax, x, z, pval_test.pvalue, pval, hatch=".")
                for _ax in ax.flatten():

                    _ax.invert_yaxis()
                    _ax.grid() 
    
            
                fig.suptitle("[pval=%.1f] Time: %s. Direction: %s, fixed %s: [%.2f, %.2f]" % (
                    args.pval,
                    plot_time.strftime("%Y/%m/%d %H:%M:%S"),
                    cxs_info["dir"],
                    fxd_coord,
                    cxs_info["loc_rng"][0],
                    cxs_info["loc_rng"][1],
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
    
    parser.add_argument('--cxs-dir', type=str, help="Latitude range for plotting", required=True, choices=["meridional", "latitudinal"])
    parser.add_argument('--cxs-loc-rng', type=float, nargs=2, help="Range of axis along the `---cxs-dir`.", required=True)
    parser.add_argument('--cxs-rng', type=float, nargs=2, help="Range of latitude or longitude for cross section", required=True)
    parser.add_argument('--nproc', type=int, help="Number of processors", default=1)
    parser.add_argument('--pval', type=float, help="p-value", default=0.1)
    
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
            unparsed_vars = args.varnames,
            input_root = args.input_root,
            expblobs = args.expblobs,
            ref_expblob = args.ref_expblob,
            exp_beg_time = args.exp_beg_time,
            plot_rel_time = plot_rel_time,
            output_root = args.output_root,
            pval = args.pval,
            cross_section_info = dict(
                dir = args.cxs_dir,
                loc_rng = args.cxs_loc_rng,
                rng = args.cxs_rng,
            ),
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
