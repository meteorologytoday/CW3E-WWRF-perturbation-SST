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
g0 = 9.81

def testIfIn(di, key, default):
    
    return di[key] if (key in di) else default


from scipy.signal import convolve2d
def moving_average_2d(data, window_size):
    """Calculates the 2D moving average of a 2D array."""

    # Create a kernel for the moving average
    kernel = np.ones((window_size, window_size)) / (window_size ** 2)

    # Use convolve2d to calculate the moving average
    return convolve2d(data, kernel, mode='same')    


plot_infos = dict(
    SST = dict(
        selector = None,
        wrf_varname = "TSK",
        label = "SST",
        unit = "K",
        levs = np.linspace(-1, 1, 11) * 1,
        cmap = cmocean.cm.balance,
    ), 
)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--input-files', type=str, nargs=2, help='Input directories.', required=True)
    parser.add_argument('--labels', type=str, nargs="+", help='Input directories.', default=None)
    parser.add_argument('--extra-title', type=str, default="")
    parser.add_argument('--output', type=str, help='Output filename in png.', required=True)
    parser.add_argument('--no-display', action="store_true")
    parser.add_argument('--lat-rng', type=float, nargs=2, help="Latitude range for plotting", default=[-90.0, 90.0])
    parser.add_argument('--lon-rng', type=float, nargs=2, help="Latitude range for plotting", default=[0.0, 360.0])

    args = parser.parse_args()

    pp.pprint(args)

    labels = args.labels

    var_infos = []

    if labels is None:
        labels = [ "%d" % i for i in range(len(args.input_files)) ]
        
    elif len(labels) != len(args.input_files):
        raise Exception("Length of `--labels` (%d) does not equal to length of `--input-dirs` (%d). " % (
            len(labels),
            len(args.input_files),
        ))

    # Loading     
    data = [] 
    for i in range(len(args.input_files)):
        input_file      = args.input_files[i]
        print("Loading the %d-th wrf dir: %s" % (i, input_file,))
        da = xr.open_dataset(input_file)['sst'].isel(pentadstamp=0)
        data.append(da)

    da_diff = data[1] - data[0]
 
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
    nrow = 1

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
        sharex=False,
    )

    fig.suptitle("Difference : %s minus %s" % (
        args.labels[0],
        args.labels[1],
    ))

    # Ctl
    _ax = ax

    _ax.set_title(args.extra_title)

    plot_info = plot_infos["SST"] 

    coords = da_diff.coords
    x = coords["lon"] % 360
    y = coords["lat"]
    d = da_diff.to_numpy()
    
    levs = testIfIn(plot_info, "levs", 11)
    plot_type = testIfIn(plot_info, "plot_type_ctl", "contourf") 
    factor = testIfIn(plot_info, "factor", 1.0)
    offset = testIfIn(plot_info, "offset", 0.0)
    d_factor = factor * (d - offset)

    if plot_type not in ["contourf", "contour"]:
        print("Warning: Unknown plot_type %s. Change it to contourf." % (plot_type,))
        plot_type = "contourf"
     
    print("Plotting contourf...")
    cmap = testIfIn(plot_info, "cmap", "gnuplot")
    mappable = _ax.contourf(x, y, d_factor, levs, cmap=cmap, extend="both", transform=proj_norm)
    cax = tool_fig_config.addAxesNextToAxes(fig, _ax, "right", thickness=0.03, spacing=0.05)
    
    cb = plt.colorbar(mappable, cax=cax, orientation="vertical", pad=0.00)
    cb.ax.set_ylabel("%s [%s]" % (plot_info["label"], plot_info["unit"]))

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



