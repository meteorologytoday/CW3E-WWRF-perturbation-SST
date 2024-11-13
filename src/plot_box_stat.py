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

param_infos = dict(
    epsilon = dict(
        label = "$\\epsilon$",
        unit  = "None",
    ),
)


plot_infos = dict(

    IVT = dict(
        selector = None,
        wrf_varname = "IVT",
        label = "IVT",
        unit = "$\\mathrm{kg} / \\mathrm{m} / \\mathrm{s} $",
        levs = np.linspace(-1, 1, 11) * 50,
        cmap = cmocean.cm.balance,
    ), 


    SST = dict(
        selector = None,
        wrf_varname = "TSK",
        label = "SST",
        unit = "K",
        levs = np.linspace(-1, 1, 11) * 2,
        cmap = cmocean.cm.balance,
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

    TTL_RAIN = dict(
        selector = None,
        label = "$ \\mathrm{ACC}_{\\mathrm{ttl}}$",
        unit = "mm",
        levs = np.linspace(-1, 1, 11) * 50,
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

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--input-dirs', type=str, nargs="+", help='Input directories.', required=True)
    parser.add_argument('--param', type=str, help='Input directories.', required=True)
    parser.add_argument('--param-values', type=float, nargs="+", help='Input directories.', required=True)
    parser.add_argument('--no-display', action="store_true")

    parser.add_argument('--time-rng', type=int, nargs=2, help="Time range in hours after --exp-beg-time", required=True)
    parser.add_argument('--exp-beg-time', type=str, help='analysis beg time', required=True)
    parser.add_argument('--wrfout-data-interval', type=int, help='Time interval between each adjacent record in wrfout files in seconds.', required=True)
    parser.add_argument('--frames-per-wrfout-file', type=int, help='Number of frames in each wrfout file.', required=True)
    parser.add_argument('--wrfout-suffix', type=str, default="")

    parser.add_argument('--varnames', type=str, nargs="+", help="Varnames to do the analysis.", required=True)

    parser.add_argument('--lat-rng', type=float, nargs=2, help="Latitude range for plotting", default=[-90.0, 90.0])
    parser.add_argument('--lon-rng', type=float, nargs=2, help="Latitude range for plotting", default=[0.0, 360.0])
    
    parser.add_argument('--output', type=str, help='Output filename in png.', required=True)

    args = parser.parse_args()

    print(args)

    exp_beg_time = pd.Timestamp(args.exp_beg_time)
    wrfout_data_interval = pd.Timedelta(seconds=args.wrfout_data_interval)
    time_beg = exp_beg_time + pd.Timedelta(hours=args.time_rng[0])
    time_end = exp_beg_time + pd.Timedelta(hours=args.time_rng[1])

    wsm = wrf_load_helper.WRFSimMetadata(
        start_datetime  = exp_beg_time,
        data_interval   = wrfout_data_interval,
        frames_per_file = args.frames_per_wrfout_file,
    )
    
    # Loading     
    data = [] 
    
    for i in range(len(args.input_dirs)):
        
        input_dir      = args.input_dirs[i] 
        print("Loading the %d-th wrf dir: %s" % (i, input_dir,))

        try:
            ds = wrf_load_helper.loadWRFDataFromDir(
                wsm, 
                input_dir,
                beg_time = time_beg,
                end_time = time_end,
                suffix=args.wrfout_suffix,
                avg=None,
                verbose=False,
                inclusive="left",
            )
                
            ds = xr.merge([
                ds,
                wrf_preprocess.genAnalysis(ds, wsm.data_interval, varnames=args.varnames),
            ])
           
            lon = ds.coords["XLONG"] % 360
            lat = ds.coords["XLAT"]

            box_rng = (
                (lat >= args.lat_rng[0]) &
                (lat <  args.lat_rng[1]) &
                (lon >= args.lon_rng[0]) &
                (lon <  args.lon_rng[1])
            )


            extracted_data = []
            for varname in args.varnames:
                plot_info = plot_infos[varname]
                selector = plot_info["selector"] if "selector" in plot_info else None
                wrf_varname = plot_info["wrf_varname"] if "wrf_varname" in plot_info else varname

                da = ds[wrf_varname]

                if selector is not None:
                    da      = da.isel(**selector)

                avg_dims = []
                for dimname in ["south_north", "south_north_stag", "west_east_stag", "west_east"]:
                    if dimname in da.dims:
                        avg_dims.append(dimname) 
                 
                da = da.where(box_rng).mean(
                    dim=avg_dims, skipna=True
                ).mean(dim="time").compute()

                extracted_data.append(da)


            data.append(xr.merge(extracted_data))
            
            
        except Exception as e:
            print("Loading error. Put None.")
            data.append(None)


    stat = dict()
    param_values = np.array(args.param_values)
    for varname in args.varnames:

        tmp = np.zeros_like(param_values)
        for i, _d in enumerate(data):
            
            if _d is None:
                tmp[i] = np.nan
            else:
                tmp[i] = _d[varname].to_numpy()
        
        stat[varname] = tmp
        
    

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
    import tool_fig_config
    print("Done.")     

    ncol = 1
    nrow = len(args.varnames)

    h = 4.0
    w = 6.0

    figsize, gridspec_kw = tool_fig_config.calFigParams(
        w = w,
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

    ax_flatten = ax.flatten()

    fig.suptitle("Time: %s ~ %s" % (
        time_beg.strftime("%Y-%m-%d %H:%M:%S"),        
        time_end.strftime("%Y-%m-%d %H:%M:%S"),        
    ))

    for i, varname in enumerate(args.varnames):
        
        print("Varname : ", varname)
        _ax = ax[i, 0]
        
        plot_info = plot_infos[varname] 

        x = param_values
        y = stat[varname]
        
        _ax.plot(x, y, marker="o", markersize=5)

        _ax.set_title("%s [%s]" % (plot_info["label"], plot_info["unit"]))

    param_info = param_infos[args.param]
    
    for _ax in ax_flatten:
        _ax.set_xlabel("%s [%s]" % (param_info["label"], param_info["unit"]))
        
    for _ax in ax_flatten:
        _ax.grid()

    if args.output != "":
        print("Saving output: ", args.output) 
        fig.savefig(args.output, dpi=200)



