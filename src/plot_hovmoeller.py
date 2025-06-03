from multiprocessing import Pool
import multiprocessing



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
from pathlib import Path

def testIfIn(di, key, default):
    return di[key] if (key in di) else default


plot_infos = dict(

    IVT = dict(
        selector = None,
        wrf_varname = "IVT",
        label = "IVT",
        unit = "$\\mathrm{kg} / \\mathrm{m} / \\mathrm{s} $",
        full = dict(levs=np.arange(0, 1000, 50), cmap=cmocean.cm.rain),
        anom = dict(levs=np.linspace(-1, 1, 21) * 50, cmap=cmocean.cm.balance),

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
        full = dict(levs=np.arange(0, 100, 5), cmap=cmocean.cm.rain),
        anom = dict(levs=np.linspace(-1, 1, 11) * 10, cmap=cmocean.cm.balance),
        label = "$ \\mathrm{ACC}_{\\mathrm{ttl}}$",
        unit = "mm",
        cmap = cmocean.cm.balance,
    ), 

    PSFC = dict(
        selector = None,
        full = dict(levs=np.arange(950, 1050, 5), cmap="rainbow"),
        anom = dict(levs=np.linspace(-1, 1, 21) * 2, cmap=cmocean.cm.balance),
        label = "$P_\\mathrm{sfc}$",
        unit = "hPa",
        factor = 1e2,
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

        spatio_direction = details['spatio_direction']
        latlon_rng = details['latlon_rng']

        varname     = details["varname"]
        input_root  = Path(details["input_root"])
        casenames   = details["casenames"]
        
        time_beg     = details["time_beg"]
        time_end     = details["time_end"]
        time_stride  = details["time_stride"]
        
        output_dir     = Path(details["output_dir"]) 
        output_prefix   = details["output_prefix"] 
        
        ref_index = details["ref_index"] 

        plot_latlon_rng = details["plot_latlon_rng"]
 
        output_file = output_dir / ("Hovmoeller_%s_%s_%s-to-%s.svg" % (
            output_prefix,
            varname,
            time_beg.strftime("%Y-%m-%dT%H"),
            time_end.strftime("%Y-%m-%dT%H"),
        ))
        
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
       
        avg_dim = None
        resolved_dim = None
        if spatio_direction == "south_north":
            resolved_dim = "lat"
            avg_dim = "lon"
        elif spatio_direction == "west_east":
            resolved_dim = "lon"
            avg_dim = "lat"
        else:
            raise Exception("Unknown spatio_direction: %s" % (spatio_direction,))

 
        output_file.parent.mkdir(parents=True, exist_ok=True)

        data = []

        for casename in casenames:
       
            full_ds = []
            # Loop through all time
            filenames = []
            for dt in pd.date_range(time_beg, time_end, freq=time_stride, inclusive="both"):
    
                print("Load %s - %s" % (casename, dt.strftime("%Y-%m-%d %H:%M:%S"),))
                filenames.append(input_root / "{casename:s}_{varname:s}_{timestr:s}.nc".format(
                    casename = casename,
                    varname  = varname,
                    timestr = dt.strftime("%Y-%m-%d_%H"),
                ))

            
            _ds = xr.open_mfdataset(filenames)
            coords = _ds.coords
            _ds = _ds.where(
                (coords[avg_dim] >= latlon_rng[0]) &
                (coords[avg_dim] <  latlon_rng[1]) 
            ).mean(dim=avg_dim)
            
            da = _ds.transpose("time", "stat", resolved_dim)[varname]
            data.append(da)
            
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
        import tool_fig_config
        print("Done.")     

        ncol = 1
        nrow = len(data)
        
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
        plot_info = plot_infos[varname] 

        ref_da = data[ref_index]
        for i, _da in enumerate(data):
            
            print("Case : ", i)
            
            _ax = ax_flatten[i]
            x = _da.coords["time"]
            y = _da.coords[resolved_dim]
           
            offset = testIfIn(plot_info, "offset", 0.0)
            factor = testIfIn(plot_info, "factor", 1.0)
            
            if i == ref_index: # Reference case
                
                _data_plot = ( _da.sel(stat="mean").to_numpy() - offset) / factor
                mappable = _ax.contourf(
                    x, y,
                    _data_plot.transpose(),
                    plot_info["full"]["levs"],
                    cmap=plot_info["full"]["cmap"],
                    extend="max",
                )
                
                cax = tool_fig_config.addAxesNextToAxes(fig, _ax, "right", thickness=0.03, spacing=0.05)
                cb = plt.colorbar(mappable, cax=cax, orientation="vertical", pad=0.00)
                cb.ax.set_ylabel("[ %s ]" % (plot_info["unit"],))

            else:
            
                _da -= ref_da
 
                _data_plot = _da.sel(stat="mean").to_numpy() / factor
                mappable = _ax.contourf(
                    x, y,
                    _data_plot.transpose(),
                    plot_info["anom"]["levs"],
                    cmap=plot_info["anom"]["cmap"],
                    extend="both",
                )
                
                cax = tool_fig_config.addAxesNextToAxes(fig, _ax, "right", thickness=0.03, spacing=0.05)
                cb = plt.colorbar(mappable, cax=cax, orientation="vertical", pad=0.00)
                cb.ax.set_ylabel("[ %s ]" % (plot_info["unit"],))
               
        date_format = DateFormatter('%m/%d') # Example format, customize as needed
        for _ax in ax_flatten:
            
            _ax.grid()
            _ax.xaxis.set_major_formatter(date_format)
            _ax.tick_params(axis='x', labelrotation=45)

            if plot_latlon_rng is not None:
                _ax.set_ylim(plot_latlon_rng)

        print("Saving output: ", output_file) 
        fig.savefig(output_file, dpi=200)



        result['status'] = 'OK'

    except Exception as e:

        traceback.print_exc()
        result['status'] = 'ERROR'
        #traceback.print_stack()

        print(e)


    return result




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--input-root', type=str, help='Input directories.', required=True)
    parser.add_argument('--casenames', type=str, nargs="+", help='Case names.', required=True)
    parser.add_argument('--varnames', type=str, nargs="+", help='Variables processed. If only a particular layer is selected, use `-` to separate the specified layer. Example: PH-850', required=True)
    parser.add_argument('--exp-beg-time', type=str, help='analysis beg time', required=True)
    parser.add_argument('--time-beg',    type=int, help='Beginning of time. Hours` after `--exp-beg-time`', required=True)
    parser.add_argument('--time-end',    type=int, help='Ending of time. Hours after `--exp-beg-time`', required=True)
    parser.add_argument('--time-stride', type=int, help='Each plot differ by this time interval. Hours', required=True)
    parser.add_argument('--spatio-direction', type=str, help='The spatial direction of hovmoeller. Can be `south_north`, or `west_east`.', required=True, choices=["south_north", "west_east"])
    parser.add_argument('--latlon-rng', type=float, nargs=2, help='The range of lat or lon to be averaged. If `--spatio-direction` is `south_north`, then this range is for longitude. If `--spatio-direction` is `west_east`, then this range is for latitude.', required=True)
    parser.add_argument('--plot-latlon-rng', type=float, help='The plotted range of the spatio direction of hovmoeller.', nargs=2, default=None)
    parser.add_argument('--output-dir', type=str, help='Output filename in nc file.', required=True)
    parser.add_argument('--output-prefix', type=str, help='analysis beg time', required=True)
    parser.add_argument('--nproc', type=int, help="Number of processors", default=1)
    parser.add_argument('--ref-index',    type=int, help='Ending of time. Hours after `--exp-beg-time`', default=0)
    args = parser.parse_args()

    print(args)
 
    failed_dates = []
    input_args = []
            

    time_beg = pd.Timestamp(args.exp_beg_time) + pd.Timedelta(hours=args.time_beg)
    time_end = pd.Timestamp(args.exp_beg_time) + pd.Timedelta(hours=args.time_end)
    time_stride = pd.Timedelta(hours=args.time_stride)
    
    for varname in args.varnames:
        
        details = dict(
            spatio_direction = args.spatio_direction,
            latlon_rng = args.latlon_rng,
            varname     = varname,
            input_root  = args.input_root,
            casenames = args.casenames,
            time_beg = time_beg,
            time_end = time_end,
            time_stride = time_stride,
            output_dir  = args.output_dir,
            output_prefix = args.output_prefix,
            ref_index = args.ref_index,
            plot_latlon_rng = args.plot_latlon_rng,
        )

        result = doJob(details, detect_phase=True)
        
        if not result['need_work']:
            print("File `%s` already exist. Skip it." % (str(result['output_file']),))
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

