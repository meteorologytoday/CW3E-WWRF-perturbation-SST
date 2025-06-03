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


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--input-obs-file', type=str, help='Input observational data such as PRISM.', required=True)
    parser.add_argument('--input-fcst-files', type=str, nargs="+", help='The stat (West-wrf) that has standard deviation', required=True)
    parser.add_argument('--labels', type=str, nargs="+", help='The label that matches the `--input-fcst`.', default=None)
    parser.add_argument('--no-display', action="store_true")
    parser.add_argument('--precip-threshold', type=float, help="Threshold of precipitation. If below this threshold then do not show. In unit of mm.", default=5.0) 
    parser.add_argument('--precip-max', type=float, help='Maximum of precipitation for shading.', default=50)
    parser.add_argument('--precip-interval', type=float, help='Maximum of precipitation for shading.', default=2.0)
    parser.add_argument('--title', type=str, help='Title.', required=True)
    parser.add_argument('--output', type=str, help='Output filename in png.', required=True)
    parser.add_argument('--lat-rng', type=float, nargs=2, help="Latitude range for doing statistics", default=[-90.0, 90.0])
    parser.add_argument('--lon-rng', type=float, nargs=2, help="Latitude range for doing statistics", default=[0.0, 360.0])
    
    
    args = parser.parse_args()

    print(args)

    def selLatLon(ds):

        ds = ds.where(
              (ds.coords["lat"] > args.lat_rng[0]) 
            & (ds.coords["lat"] < args.lat_rng[1])
            & (ds.coords["lon"] > args.lon_rng[0])
            & (ds.coords["lon"] < args.lon_rng[1])
        )

        return ds

    data = dict(
        fcst = [ selLatLon(xr.open_dataset(fname, engine="netcdf4").isel(time=0)["TTL_RAIN"]) for fname in args.input_fcst_files ],
        obs  = selLatLon(xr.open_dataset(args.input_obs_file, engine="netcdf4").isel(time=0)["obs"]),
    )

    

    


    skills  = [ np.abs( da_fcst.sel(stat="mean").to_numpy() - data["obs"]).to_numpy()  for  da_fcst in data["fcst"] ]
    spreads = [ da_fcst.sel(stat="std").to_numpy() for da_fcst in data["fcst"] ]

    obs = data["obs"].to_numpy() 
    valid_idx = np.isfinite(obs) & (obs > args.precip_threshold)

    #ratio[data["target"].sel(stat="mean").to_numpy() < args.precip_threshold ] = np.nan
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

    ncol = len(skills)
    nrow = 1

    figsize, gridspec_kw = tool_fig_config.calFigParams(
        w = 5,
        h = 5,
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

    ax_flatten = ax.flatten()
   
    hist_levs = None
    hist_upperbound = None 
    for i, (skill, spread, fname) in enumerate(zip(skills, spreads, args.input_fcst_files)):
        _ax = ax_flatten[i]

        print("skill.shape = ", skill.shape)
        print("spread.shape = ", spread.shape)
        nlevs = int(np.ceil(args.precip_max / args.precip_interval + 1))
        xedges = np.linspace(0, args.precip_max, nlevs)
        yedges = np.linspace(0, args.precip_max, nlevs)
        print(skill.shape)
        print(spread.shape)

        x = skill[valid_idx]
        y = spread[valid_idx]

        hist, xedges, yedges = np.histogram2d(x, y, bins=(xedges, yedges))
        midx = (xedges[:-1] + xedges[1:]) / 2
        midy = (yedges[:-1] + yedges[1:]) / 2

        if hist_upperbound is None:
            
            hist_upperbound = np.max(hist.flatten()) * 0.95
            print("hist_upperbound = ", hist_upperbound)
            hist_levs = np.linspace(0, hist_upperbound, 21) 

        #_ax.scatter(skill, spread, label="%d" % i)
        mappable = _ax.contourf(midx, midy, hist, hist_levs, cmap="gnuplot")

        _ax.legend()
        _ax.set_xlabel("Precip Std [ $\\mathrm{mm}$ ]")
        _ax.set_ylabel("| Fcst - Obs | [ $\\mathrm{mm}$ ]")
     
        lim = np.array([0, args.precip_max])
        _ax.set_xlim(lim) 
        _ax.set_ylim(lim) 
        _ax.set_title(fname.split("/")[-1])

    if args.output != "":
        print("Saving output: ", args.output) 
        fig.savefig(args.output, dpi=200)



