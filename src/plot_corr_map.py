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
import re
import pprint 
        
pp = pprint.PrettyPrinter(indent=4)

def testIfIn(di, key, default):
    
    return di[key] if (key in di) else default


from scipy.signal import convolve2d
def moving_average_2d(data, window_size):
    """Calculates the 2D moving average of a 2D array."""

    # Create a kernel for the moving average
    kernel = np.ones((window_size, window_size)) / (window_size ** 2)

    # Use convolve2d to calculate the moving average
    return convolve2d(data, kernel, mode='same')    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--input', type=str, help='Input directories.', required=True)
    parser.add_argument('--title', type=str, help='Input directories.', default="")
    parser.add_argument('--output', type=str, help='Output filename in png.', required=True)
    parser.add_argument('--no-display', action="store_true")
    parser.add_argument('--varnames', type=str, nargs="+", help="Varnames to do the analysis.", required=True)
    parser.add_argument('--lat-rng', type=float, nargs=2, help="Latitude range for plotting", default=[-90.0, 90.0])
    parser.add_argument('--lon-rng', type=float, nargs=2, help="Latitude range for plotting", default=[0.0, 360.0])

    args = parser.parse_args()

    pp.pprint(args)

    ds = xr.open_dataset(args.input).isel(time=0)

    # Plot data
    print("Loading Plotting Modules: Matplotlib and Cartopy.")
    import matplotlib as mpl
    if args.no_display is False:
        mpl.use('TkAgg')
    else:
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

    import tool_fig_config
    print("Done.")     

    lon_rng = np.array(args.lon_rng, dtype=float)
    lat_rng = np.array(args.lat_rng, dtype=float)

    plot_lon_l, plot_lon_r = lon_rng
    plot_lat_b, plot_lat_t = lat_rng

    lon_span = lon_rng[1] - lon_rng[0]
    lat_span = lat_rng[1] - lat_rng[0]
    
    ncol = 1
    nrow = len(args.varnames)

    h = 4.0
    w_map = h * lon_span / lat_span

    figsize, gridspec_kw = tool_fig_config.calFigParams(
        w = w_map,
        h = h,
        wspace = 2.0,
        hspace = 1.0,
        w_left = 1.0,
        w_right = 2.0,
        h_bottom = 1.5,
        h_top = 1.0,
        ncol = ncol,
        nrow = nrow,
    )

    cent_lon = 180.0

    proj = ccrs.PlateCarree(central_longitude=cent_lon)
    proj_norm = ccrs.PlateCarree()

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

    fig.suptitle(args.title)

    for j, varname in enumerate(args.varnames):
       
        print("[%d/%d] Plotting varname : %s" % (j+1, len(args.varnames), varname,))
        
        _ax = ax[j, 0]

        coords = ds.coords
        x = coords["XLONG"] % 360
        y = coords["XLAT"]
        d = ds[varname]
        
        levs = np.linspace(-1, 1, 21)
        
        cmap = "cmo.balance"
        mappable = _ax.contourf(x, y, d, levs, cmap=cmap, extend="both", transform=proj_norm)
        cax = tool_fig_config.addAxesNextToAxes(fig, _ax, "right", thickness=0.03, spacing=0.05)
        
        cb = plt.colorbar(mappable, cax=cax, orientation="vertical", pad=0.00)
        cb.ax.set_ylabel("Correlation")
        
        _ax.set_title("(%s) %s" % ("abcdefghijklmn"[j], varname, ))
   
    
 
    for _ax in ax_flatten:
        _ax.set_global()
        _ax.coastlines()
        _ax.set_extent([plot_lon_l, plot_lon_r, plot_lat_b, plot_lat_t], crs=proj_norm)

        gl = _ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                          linewidth=1, color='gray', alpha=0.5, linestyle='--')

        gl.xlabels_top   = False
        gl.ylabels_right = False

        #gl.xlocator = mticker.FixedLocator(np.arange(-180, 181, 30))
        #gl.xlocator = mticker.FixedLocator([120, 150, 180, -150, -120])#np.arange(-180, 181, 30))
        #gl.ylocator = mticker.FixedLocator([10, 20, 30, 40, 50])

        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        gl.xlabel_style = {'size': 10, 'color': 'black'}
        gl.ylabel_style = {'size': 10, 'color': 'black'}


    if args.output != "":
        print("Saving output: ", args.output) 
        fig.savefig(args.output, dpi=200)



