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


def pullVariableOut(da):

    merge = dict()
    for varname in da.coords["variable"]:
       
        varname = str(varname.to_numpy())
        new_da = da.sel(variable=varname).drop_vars("variable").rename(varname)
        merge[varname] = new_da


    return xr.Dataset(merge)


def genACC(ds, varnames=None, include_old=False):

    if varnames is None:
        varnames = list(ds.keys())

    merge = dict()
   
     
    for varname in varnames:
       
        new_varname = "ACC_%s" % (varname,)
        new_da = ds[varname].cumsum(dim="time")
        merge[new_varname] = new_da

    new_ds = xr.Dataset(merge)

    if include_old:
        new_ds = xr.merge([new_ds, ds])

    return new_ds


def computeCRPS(mu, sig, obs):
    crps = np.zeros((len(y_mean),))
    for k in range(len(crps)):
        crps[k] = CRPS_tools.CRPS_gaussian(mu[k], sig[k], obs[k])


    return crps



plot_infos = dict(

    convective_precipitation = dict(
        selector = None,
        label = "Daily convective precip",
        unit = "$\\mathrm{mm}$",
        factor = 1e3,
        lim = [0, 30],
        #levs = np.linspace(-1, 1, 11) * 50,
        #cmap = cmocean.cm.balance,
    ), 


    large_scale_precipitation = dict(
        selector = None,
        label = "Daily large-scale precip",
        unit = "$\\mathrm{mm}$",
        factor = 1e3,
        lim = [0, 30],
        #levs = np.linspace(-1, 1, 11) * 50,
        #cmap = cmocean.cm.balance,
    ), 


    total_precipitation = dict(
        selector = None,
        label = "Daily total precip",
        unit = "$\\mathrm{mm}$",
        factor = 1e3,
        lim = [0, 30],
        #levs = np.linspace(-1, 1, 11) * 50,
        #cmap = cmocean.cm.balance,
    ), 

    ACC_total_precipitation = dict(
        selector = None,
        label = "Accumulative total precip",
        unit = "$\\mathrm{mm}$",
        factor = 1e3,
        lim = [0, 200],
        #levs = np.linspace(-1, 1, 11) * 50,
        #cmap = cmocean.cm.balance,
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
    parser.add_argument('--input-WRF', type=str, nargs="+", help='Input directories.', required=True)
#    parser.add_argument('--input-ERA5', type=str, help='Input directories.', required=True)
    parser.add_argument('--input-PRISM', type=str, help='Input directories.', required=True)
    
    parser.add_argument('--region', type=str, help='Input directories.', required=True)
    
    parser.add_argument('--no-display', action="store_true")
    parser.add_argument('--no-legend', action="store_true")

    parser.add_argument('--varnames', type=str, nargs="+", help="Varnames to do the analysis.", required=True)

    parser.add_argument('--output', type=str, help='Output filename in png.', required=True)

    args = parser.parse_args()

    print(args)

    def preprocess(ds):
        return genACC(pullVariableOut(ds), include_old=True)
    
    data_WRF = [ preprocess(xr.open_dataset(fname).sel(region=args.region)["data"]) for fname in args.input_WRF ]
#    ds_ERA5  = preprocess(xr.open_dataset(args.input_ERA5).sel(region=args.region)["obs_data"])
    ds_PRISM_clim = preprocess(xr.open_dataset(args.input_PRISM).sel(region=args.region)["clim_data"])
    ds_PRISM_obs  = preprocess(xr.open_dataset(args.input_PRISM).sel(region=args.region)["obs_data"])
   

    print(list(data_WRF[0].keys()))


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


    fig.suptitle("Region = %s" % ( args.region, ))
    ax_flatten = ax.flatten()

    for j, varname in enumerate(args.varnames):

        print("Varname : ", varname)

        _ax0 = ax[j, 0]
        #_ax1 = ax[j, 1]
        ##_ax2 = ax[j, 2]

        plot_info = plot_infos[varname] 
        factor = plot_info["factor"]

        # Plot PRISM observation data
        x = ds_PRISM_obs.coords["time"]
        y_obs = ds_PRISM_obs[varname]
        print(y_obs)
        _ax0.plot(x, y_obs, "k-", linewidth=2, label="PRISM_obs", zorder=5)
        
        
        #print("keys = ", list(ds_PRISM_obs.keys()))
        if varname in ["total_precipitation", ]:

            print("Variable in PRISM...")
            
            # I am not using PRISM time because it is climatology, 
            # I put dummy year 2001
            da_clim = ds_PRISM_clim[varname]
            clim_mean = da_clim.mean(dim="year").to_numpy()
            clim_std  = da_clim.std(dim="year").to_numpy()

            print()

            _ax0.errorbar(x.to_numpy(), PRISM_mean, PRISM_std, color="blue", linewidth=2, zorder=5, label="PRISM_clim")



        for i, ds in enumerate(data_WRF):
            
            # Plot WRF
            x = ds.coords["time"]
            
            y_ens = ds[varname] * factor
            y_mean = y_ens.mean(dim="ens_id")
            y_std  = y_ens.std(dim="ens_id")

            _ax0.plot(x, y_ens, linewidth=1, label="WRF", zorder=4, linestyle="dashed")
                        
            
            #print("y_std = ", y_std)
            #print("y_mean = ", y_mean)
            _ax0.errorbar(x.to_numpy(), y_mean.to_numpy(), y_std.to_numpy(), color="red", linewidth=2, zorder=5)

            # compute CRPS

            crps_WRF = computeCRPS(y_mean, y_std, y_obs)

            #_ax1.plot(x, crps_WRF, color="black", linewidth=2, label="WRF", zorder=4,)
            

            if varname == "total_precipitation":
                
                # compute CRPSS
                crps_PRISM = computeCRPS(PRISM_mean, PRISM_std, y_obs)

                
                CRPSS = 1.0 - crps_WRF / crps_PRISM

                #_ax2.plot(x, CRPSS, color="black", linewidth=2, label="WRF", zorder=4,)
            
                
                



        _ax0.set_title("%s [%s]" % (plot_info["label"], plot_info["unit"]))

        if "lim" in plot_info:
            _ax0.set_ylim(plot_info["lim"])

        #_ax1.set_ylim([0, 8])
        #_ax2.set_ylim([-1, 1])

        
    date_format = DateFormatter('%m/%d') # Example format, customize as needed
    for _ax in ax_flatten:
        
        if not args.no_legend:
            _ax.legend()
        
        _ax.grid()
        _ax.xaxis.set_major_formatter(date_format)
        _ax.tick_params(axis='x', labelrotation=45)

    if args.output != "":
        print("Saving output: ", args.output) 
        fig.savefig(args.output, dpi=200)



