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

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--input-files', type=str, nargs="+", help='Input three stats used to compute their difference as the forcing effect.', required=True)
    parser.add_argument('--labels', type=str, nargs="+", help='Labels that match the `--input-forcing-files`', default=None)
    parser.add_argument('--no-display', action="store_true")
    parser.add_argument('--precip-threshold', type=float, help="Threshold of precipitation. If below this threshold then do not show. In unit of mm.", default=5.0) 
    parser.add_argument('--precip-max', type=float, help='Maximum of precipitation for shading.', default=200.0)
    parser.add_argument('--std-precip-max', type=float, help='Maximum of precipitation for shading.', default=50.0)
    parser.add_argument('--diff-precip-max', type=float, help='Maximum of precipitation for shading.', default=10)
    parser.add_argument('--diff-std-precip-max', type=float, help='Maximum of precipitation for shading.', default=10)
    parser.add_argument('--title', type=str, help='Title.', default="")
    parser.add_argument('--output', type=str, help='Output filename in png.', required=True)
    parser.add_argument('--lat-rng', type=float, nargs=2, help="Latitude range for plotting", default=[-90.0, 90.0])
    parser.add_argument('--lon-rng', type=float, nargs=2, help="Latitude range for plotting", default=[0.0, 360.0])
    
    
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
    
    
    data = [
        xr.open_dataset(fname, engine="netcdf4").isel(time=0)["TTL_RAIN"] for fname in args.input_files
    ]
    
    
    ref_da = data[0]
    lat = ref_da.coords["lat"]
    lon = ref_da.coords["lon"] % 360
 
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
    import cmocean as cmo

    import tool_fig_config
    print("Done.")     

    precip_levs = np.linspace(0, args.precip_max, 11)[1:]
    std_precip_levs = np.linspace(0, args.std_precip_max, 11)[1:]
    delta_precip_levs = np.linspace(-1, 1, 11) * args.diff_precip_max
    delta_std_precip_levs = np.linspace(-1, 1, 11) * args.diff_std_precip_max

    # 1. west-wrf spread
    # 2. forcing difference
    # 3. forcing difference / west-wrf
    ncol = len(args.input_files)
    nrow = 2

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
    proj_norm = ccrs.PlateCarree()

    def plot_hatch(ax, x, y, d, threshold, hatch="."):

        _dot = np.zeros_like(d)
        _significant_idx =  (d > threshold ) 
        _dot[ _significant_idx                 ] = 0.75
        _dot[ np.logical_not(_significant_idx) ] = 0.25
        cs = ax.contourf(x, y, _dot, colors='none', levels=[0, 0.5, 1], hatches=[None, hatch], transform=proj_norm)

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

    for i, da in enumerate(data):

        is_ref = i == 0
        ax_flatten = ax[:, i].flatten()

        diff_da = da - ref_da
        
        # First : mean value 
        _ax = ax_flatten[0]

        if is_ref: #

            mappable = _ax.contourf(lon, lat, da.sel(stat="mean"), precip_levs, cmap="cmo.rain", transform=proj, extend="max")
            cax = tool_fig_config.addAxesNextToAxes(fig, _ax, "right", thickness=0.03, spacing=0.17)
            cb = plt.colorbar(mappable, cax=cax, orientation="vertical", pad=0.00)
            cb.ax.set_ylabel("Precip [ $\\mathrm{mm}$ ]")
            
        else: 

            mappable = _ax.contourf(lon, lat, diff_da.sel(stat="mean"), delta_precip_levs, cmap="cmo.balance", transform=proj, extend="both")
            cax = tool_fig_config.addAxesNextToAxes(fig, _ax, "right", thickness=0.03, spacing=0.17)
            cb = plt.colorbar(mappable, cax=cax, orientation="vertical", pad=0.00)
            cb.ax.set_ylabel("Precip Diff [ $\\mathrm{mm}$ ]")
 
        # First : mean value 
        _ax = ax_flatten[1]

        if is_ref: #

            mappable = _ax.contourf(lon, lat, da.sel(stat="std"), delta_precip_levs, cmap="cmo.rain", transform=proj, extend="max")
            cax = tool_fig_config.addAxesNextToAxes(fig, _ax, "right", thickness=0.03, spacing=0.17)
            cb = plt.colorbar(mappable, cax=cax, orientation="vertical", pad=0.00)
            cb.ax.set_ylabel("Precip std [ $\\mathrm{mm}$ ]")
            
        else: 

            mappable = _ax.contourf(lon, lat, diff_da.sel(stat="std"), delta_std_precip_levs, cmap="cmo.balance", transform=proj, extend="both")
            cax = tool_fig_config.addAxesNextToAxes(fig, _ax, "right", thickness=0.03, spacing=0.17)
            cb = plt.colorbar(mappable, cax=cax, orientation="vertical", pad=0.00)
            cb.ax.set_ylabel("Precip std Diff [ $\\mathrm{mm}$ ]")
        
       
        for __ax in ax_flatten: 

            __ax.set_global()
            #__ax.gridlines()
            __ax.coastlines(color='gray')
            __ax.set_extent([plot_lon_l, plot_lon_r, plot_lat_b, plot_lat_t], crs=proj_norm)

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
                    shp["geometry"], crs=proj_norm, facecolor="none", edgecolor="black"
                )
            """


    """
    ax_flatten[0].set_title("Time: %s~%s" % (
        selected_time[0].strftime("%Y/%m/%d"),
        selected_time[-1].strftime("%Y/%m/%d"),
    ))
    """


    if args.output != "":
        print("Saving output: ", args.output) 
        fig.savefig(args.output, dpi=200)



