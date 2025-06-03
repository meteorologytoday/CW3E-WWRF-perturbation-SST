import traceback
import xarray as xr
import pandas as pd
import numpy as np
import argparse
import tool_fig_config
import wrf_load_helper 
import datetime
import os
import wrf_preprocess
import cmocean
from pathlib import Path
import WRF_ens_tools

import multiprocessing
from multiprocessing import Pool 

def testIfIn(di, key, default):
    return di[key] if (key in di) else default


plot_infos = {

    "TTL_RAIN" : dict(
        label = "Precipitation",
        unit = "mm",
        factor = 1.0,
    ), 

} 


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--input-files', type=str, nargs="+", help='Input directories.', required=True)
    parser.add_argument('--varname', type=str, help='Variable name.', required=True)
    parser.add_argument('--region', type=str, help='Region name.', required=True)
    parser.add_argument('--xvals', type=float, nargs="+", help='Input directories.', required=True)
    parser.add_argument('--xlabel', type=str, help='Input directories.', required=True)
    parser.add_argument('--xticklabels', type=str, nargs="+", help='Input directories.', default=None)
    parser.add_argument('--output', type=str, help='Output filename.', required=True)
    parser.add_argument('--no-display', action="store_true")

    args = parser.parse_args()

    print(args)

    data = []
    xvals = np.array(args.xvals)

    for i, (input_file, xval) in enumerate(zip(args.input_files, xvals)):
        da = xr.open_dataset(input_file)
        da = da[args.varname].isel(time=0).sel(region=args.region).expand_dims({"xval" : [xval, ]})
        
        data.append(da)

    da = xr.merge(data)[args.varname] 
    #da = xr.open_mfdataset(args.input_files)[args.varname].isel(time=0, region=args.region) 
    print(da)
   
    print("Loading Plotting Modules: Matplotlib and Cartopy.")
    import matplotlib as mpl
    mpl.use('Agg')
    mpl.rc('font', size=15)
    mpl.rc('axes', labelsize=15)


    import matplotlib.pyplot as plt
    if args.no_display is False:
        mpl.use('TkAgg')
    else:
        mpl.use('Agg')
        mpl.rc('font', size=15)
        mpl.rc('axes', labelsize=15)
 
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


    ncol = 1
    nrow = 1

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
        squeeze=False,
        sharex=False,
    )


    _ax = ax[0, 0]

    plot_info = plot_infos[args.varname]
        
    fig.suptitle("%s" % (
        plot_info["label"],
    ))
   

    _ax.errorbar(
        xvals,
        da.sel(stat="mean").to_numpy(),
        da.sel(stat="std").to_numpy() / np.sqrt(da.sel(stat="count").to_numpy()),
        color="black",
        linewidth=2,
        zorder=5,
        fmt='-o',
    )
 
    _ax.set_ylabel("[ %s ]" % (
        plot_info["unit"],
    ))
 
    if args.xlabel is not None:
        _ax.set_xlabel(args.xlabel)
    
    _ax.set_xticks(xvals)


    if args.xticklabels is not None:
        _ax.set_xticklabels(xvals, args.xticklabels)
    

    _ax.grid()
 
    print("Saving output: ", args.output) 
    fig.savefig(args.output, dpi=200)



