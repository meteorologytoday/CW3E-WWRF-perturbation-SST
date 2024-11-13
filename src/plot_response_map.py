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

    PRECIP = dict(
        selector = None,
        label = "Precip",
        factor = 24.0,
        unit = "$\\mathrm{mm} / \\mathrm{day} $",
        levs = np.linspace(-1, 1, 11) * 10,
        cmap = cmocean.cm.balance,
    
        levs_ctl = np.arange(0, 20, 0.5),
        cmap_ctl = cmocean.cm.deep,
        plot_type_ctl = "contourf", 
    ), 


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
        offset = 273.15,
        levs_ctl = np.arange(0, 30, 1),
        cmap_ctl = cmocean.cm.thermal,
        plot_type_ctl = "contourf", 
    ), 


    PH850 = dict(
        selector = None,
        label = "$ \\Phi_{850}$",
        unit = "$\\mathrm{m}$",
        levs = np.linspace(-1, 1, 11) * 5,
        cmap = cmocean.cm.balance,
        factor = g0**(-1),

        levs_ctl = np.arange(1000, 2500, 25),
        plot_type_ctl = "contour", 
        clabel_fmt_ctl = "%d",
        contour_nolnd_ctl = True, 
    ), 


    PH500 = dict(
        selector = None,
        label = "$ \\Phi_{500}$",
        unit = "$\\mathrm{m}$",
        levs = np.linspace(-1, 1, 11) * 20,
        cmap = cmocean.cm.balance,
        factor = g0**(-1),

        levs_ctl = np.arange(5000, 7000, 50),
        plot_type_ctl = "contour", 
        clabel_fmt_ctl = "%d", 
        contour_nolnd_ctl = True, 
    ), 

    TTL_RAIN = dict(
        selector = None,
        label = "$ \\mathrm{ACC}_{\\mathrm{ttl}}$",
        unit = "mm",
        levs = np.linspace(-1, 1, 11) * 50,
        cmap = cmocean.cm.balance,
        low_pass = 7,

        levs_ctl = np.arange(0, 200, 5),
        cmap_ctl = cmocean.cm.deep,
        plot_type_ctl = "contourf", 
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
    parser.add_argument('--input-dirs-base', type=str, nargs="+", help='Input directories.', required=True)
    parser.add_argument('--labels', type=str, nargs="+", help='Input directories.', default=None)
    parser.add_argument('--output', type=str, help='Output filename in png.', required=True)
    parser.add_argument('--no-display', action="store_true")

    parser.add_argument('--time-rng', type=int, nargs=2, help="Time range in hours after --exp-beg-time", required=True)
    parser.add_argument('--exp-beg-time', type=str, help='analysis beg time', required=True)
    parser.add_argument('--wrfout-data-interval', type=int, help='Time interval between each adjacent record in wrfout files in seconds.', required=True)
    parser.add_argument('--frames-per-wrfout-file', type=int, help='Number of frames in each wrfout file.', required=True)
    parser.add_argument('--wrfout-suffix', type=str, default="")

    parser.add_argument('--varnames', type=str, nargs="+", help="Varnames to do the analysis.", required=True)

    parser.add_argument('--lat-rng', type=float, nargs=2, help="Latitude range for plotting", default=[-90.0, 90.0])
    parser.add_argument('--lon-rng', type=float, nargs=2, help="Latitude range for plotting", default=[0.0, 360.0])

    args = parser.parse_args()

    pp.pprint(args)

    labels = args.labels


    var_infos = []

    pattern = r"(?P<varname>[_a-zA-Z0-9]+)(\.FILTER-(?P<FILTER>[a-zA-Z]+))?"
    for i, varname_full in enumerate(args.varnames):
        detail = dict()
        match = re.search(pattern, varname_full)
        if match:
            varname = match.group("varname")
            detail["FILTER"] = match.group("FILTER")

        var_info = dict(
            varname = varname,
            detail  = detail,
        )

        print("[%d] Parsed info: " % (i+1, ))
        pp.pprint(var_info)

        var_infos.append(var_info)

    if labels is None:
        
        labels = [ "%d" % i for i in range(len(args.input_dirs)) ]
        
    elif len(labels) != len(args.input_dirs):
        raise Exception("Length of `--labels` (%d) does not equal to length of `--input-dirs` (%d). " % (
            len(labels),
            len(args.input_dirs),
        ))

    same_base = False
    if len(args.input_dirs_base) == 1:
        args.input_dirs_base = [ args.input_dirs_base[0] ] * len(args.input_dirs) 

    if np.all( [ input_dir_base == args.input_dirs_base[0] for input_dir_base in args.input_dirs_base  ] ):
        same_base = True
        print("# same_base = ", same_base)

    if len(args.input_dirs_base) != len(args.input_dirs):
        
        raise Exception("Length of `--input-dirs-base` (%d) does not equal to length of `--input-dirs` (%d). " % (
            len(args.input_dirs_base),
            len(args.input_dirs),
        ))

 
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
    data_ctl = None
    landmask = None
    for i in range(len(args.input_dirs)):
        
        input_dir_base = args.input_dirs_base[i] 
        input_dir      = args.input_dirs[i]
         
        print("Loading base wrf dir: %s" % (input_dir_base,))
        if i == 0 or not same_base:
            ds_base = wrf_load_helper.loadWRFDataFromDir(
                wsm, 
                input_dir_base,
                beg_time = time_beg,
                end_time = time_end,
                suffix=args.wrfout_suffix,
                avg="ALL",
                verbose=False,
                inclusive="left",
            )

            ds_base = xr.merge([
                ds_base,
                wrf_preprocess.genAnalysis(ds_base, wsm.data_interval),
            ])

            landmask = ds_base["LANDMASK"].isel(time=0)

        #print(list(ds_base.keys()))

        print("Loading the %d-th wrf dir: %s" % (i, input_dir,))
        ds = wrf_load_helper.loadWRFDataFromDir(
            wsm, 
            input_dir,
            beg_time = time_beg,
            end_time = time_end,
            suffix=args.wrfout_suffix,
            avg="ALL",
            verbose=False,
            inclusive="left",
        )
            
        ds = xr.merge([
            ds,
            wrf_preprocess.genAnalysis(ds, wsm.data_interval),
        ])
            
        #print(ds)
     
        extracted_data = []
        extracted_data_ctl = []
        for var_info in var_infos:
            
            varname = var_info["varname"]
            
            plot_info = plot_infos[varname]

            selector = plot_info["selector"] if "selector" in plot_info else None
            wrf_varname = plot_info["wrf_varname"] if "wrf_varname" in plot_info else varname

            da_base = ds_base[wrf_varname]
            da = ds[wrf_varname]


            if selector is not None:
                da_base = da_base.isel(**selector)
                da      = da.isel(**selector)
           
             
            dvar = da - da_base
            dvar = dvar.rename(varname)
            extracted_data.append(dvar)
            extracted_data_ctl.append(da_base)
            
        data.append(xr.merge(extracted_data))
        data_ctl = xr.merge(extracted_data_ctl)

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
    
    ncol = len(data) + 1
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

    fig.suptitle("Time: %s ~ %s" % (
        time_beg.strftime("%Y-%m-%d %H:%M:%S"),        
        time_end.strftime("%Y-%m-%d %H:%M:%S"),        
    ))

    # Ctl
    print("Plotting control case")
    _ds = data_ctl.isel(time=0)

    for j, var_info in enumerate(var_infos):
       
        varname = var_info["varname"]
         
        print("Varname : ", varname)
        _ax = ax[j, 0]

        plot_info = plot_infos[varname] 

        coords = _ds.coords
        x = coords["XLONG"] % 360
        y = coords["XLAT"]
        d = _ds[varname]
        
       
        levs = testIfIn(plot_info, "levs_ctl", 11)
        plot_type = testIfIn(plot_info, "plot_type_ctl", "contourf") 
        factor = testIfIn(plot_info, "factor", 1.0)
        offset = testIfIn(plot_info, "offset", 0.0)
        print("offset = ", offset)
        d_factor = factor * (d - offset)

        if "FILTER" in (detail := var_info["detail"]):
            fltr = detail["FILTER"]
            if fltr is None:
                pass # Do nothing
            elif fltr == "LP":
                window_size = 7
                d_factor[:, :] = moving_average_2d(d_factor.to_numpy(), window_size)
            else:
                raise Exception("Unknown filter: %s" % (str(fltr),))
 
        if plot_type not in ["contourf", "contour"]:
            print("Warning: Unknown plot_type %s. Change it to contourf." % (plot_type,))
            plot_type = "contourf"
         
        if plot_type == "contourf":
            
            print("Plotting contourf...")
            cmap = testIfIn(plot_info, "cmap_ctl", "cmo.gnuplot")
            mappable = _ax.contourf(x, y, d_factor, levs, cmap=cmap, extend="both", transform=proj_norm)
            cax = tool_fig_config.addAxesNextToAxes(fig, _ax, "right", thickness=0.03, spacing=0.05)
            
            cb = plt.colorbar(mappable, cax=cax, orientation="vertical", pad=0.00)
            cb.ax.set_ylabel("%s [%s]" % (plot_info["label"], plot_info["unit"]))
        
        elif plot_type == "contour":

            print("Plotting contour...")
            fmt = testIfIn(plot_info, "clabel_fmt_ctl", "%.f")
            contour_nolnd = testIfIn(plot_info, "contour_nolnd_ctl", False)
    
            if contour_nolnd:
                print("Remove land!")
                print("Before: ", np.sum(np.isfinite(d_factor.to_numpy())))
                d_factor = d_factor.where(landmask != 1.00)
                print("After: ", np.sum(np.isfinite(d_factor.to_numpy())))


            cs = _ax.contour(x, y, d_factor, levs, colors="black", transform=proj_norm)
            clables = plt.clabel(cs, fmt=fmt)
           
        
        _ax.set_title("CTL: %s" % (plot_info["label"],))
        
    # Diff
    for i, ds in enumerate(data):
        _ds = ds.isel(time=0)
        print("Case : ", i)
        for j, var_info in enumerate(var_infos):
            varname = var_info["varname"] 
            print("Varname : ", varname)

            _ax = ax[j, i+1]

            plot_info = plot_infos[varname] 

            coords = _ds.coords
            x = coords["XLONG"] % 360
            y = coords["XLAT"]
            d = _ds[varname]
            levs = plot_info["levs"]
            cmap = plot_info["cmap"] if "cmap" in plot_info else "cmo.balance"
            
            factor = plot_info["factor"] if "factor" in plot_info else 1
            d_factor = d * factor

            if "FILTER" in (detail := var_info["detail"]):
                fltr = detail["FILTER"]
                if fltr is None:
                    pass # Do nothing
                elif fltr == "LP":
                    window_size = 5
                    d_factor[:, :] = moving_average_2d(d_factor.to_numpy(), window_size)
                else:
                    raise Exception("Unknown filter: %s" % (str(fltr),))
     

            mappable = _ax.contourf(x, y, d_factor, levs, cmap=cmap, extend="both", transform=proj_norm)
            cax = tool_fig_config.addAxesNextToAxes(fig, _ax, "right", thickness=0.03, spacing=0.05)
            
            cb = plt.colorbar(mappable, cax=cax, orientation="vertical", pad=0.00)
            cb.ax.set_ylabel("%s [%s]" % (plot_info["label"], plot_info["unit"]))
        
            _ax.set_title("ANOM: %s" % (plot_info["label"],))



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



