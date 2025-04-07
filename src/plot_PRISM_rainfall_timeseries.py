import numpy as np
import xarray as xr
import scipy
import traceback
from pathlib import Path
import pandas as pd
import argparse
import geopandas as gpd

import PRISM_tools

def parse_ranges(input_str):
    numbers = []
    for part in input_str.split(','):
        part = part.strip()
        if '-' in part:
            start, end = map(int, part.split('-'))
            numbers.extend(range(start, end + 1))
        else:
            numbers.append(int(part))
    return numbers


parser = argparse.ArgumentParser(
                    prog = 'plot_skill',
                    description = 'Plot prediction skill of GFS on AR.',
)

parser.add_argument('--wateryears',     type=str, help='Water year', required=True)
parser.add_argument('--input-dir',      type=str, help='Water year', required=True)
parser.add_argument('--output',         type=str, help='Output file', default="")
parser.add_argument('--region',         type=str, help='Output file', required=True)
parser.add_argument('--plot-pentad-rng',   type=int, nargs=2, help='Water year', default=None)
parser.add_argument('--plot-precip-rng',   type=float, nargs=2, help='Water year', default=[0, 50])
parser.add_argument('--no-display', action="store_true")

args = parser.parse_args()
print(args)
        
wateryears = parse_ranges(args.wateryears)
print("Parsed wateryears: ", wateryears)    
input_dir = Path(args.input_dir)

files = [ input_dir / ("PRISM_stat_year%04d.nc" % year) for year in range(np.min(wateryears)-1, np.max(wateryears)+1) ]

ds = xr.open_mfdataset(files, engine="netcdf4")
da = ds.sel(region=args.region, variable="total_precipitation")["obs_data"]



# Plot data
print("Loading Matplotlib...")
import matplotlib as mpl
if args.no_display is False:
    mpl.use('TkAgg')
else:
    mpl.use('Agg')
    mpl.rc('font', size=15)
    mpl.rc('axes', labelsize=15)

print("Done.")     
 
 
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Rectangle
import matplotlib.transforms as transforms
from matplotlib.dates import DateFormatter
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import tool_fig_config
import cmocean as cmo

from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

ncol = 1
nrow = 2

h = 4.0
w = 8.0

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
    subplot_kw=dict(aspect="auto"),
    gridspec_kw=gridspec_kw,
    constrained_layout=False,
    squeeze=False,
    sharex=True,
)

ax_flatten = ax.flatten()


_ax0 = ax_flatten[0]
_ax1 = ax_flatten[1]
_trans1 = transforms.blended_transform_factory(_ax1.transAxes, _ax1.transData)

for wateryear in wateryears:
    
    ref_beg_of_wateryear = pd.Timestamp(year=wateryear-1, month=10, day=1)
    selected_dts = pd.date_range(ref_beg_of_wateryear, ref_beg_of_wateryear + pd.offsets.DateOffset(years=1), inclusive="left")
   
    #print(da.coords["time"].to_numpy()) 
    _da = da.sel(time=selected_dts)

    x0 = (selected_dts[0] - pd.Timestamp(year=selected_dts[0].year, month=1, day=1)) / pd.Timedelta(days=5)
    x = x0 + np.arange(len(selected_dts)) / 5
    y = _da.to_numpy()

    print("There are %d NaN points. " % np.sum(np.isnan(y)))
    y[np.isnan(y)] = 0.0
    y_acc = np.cumsum(y)
    print("y.shape = ", y.shape)
    print("y_acc.shape = ", y_acc.shape)

    _ax0.plot(x, y, label="%04d" % wateryear)
     
    #print(x)
    #print(y_acc)
    _ax1.plot(x, y_acc)
    _ax1.text(0.95, y_acc[-1], "%04d" % wateryear, va="bottom", ha="right", transform=_trans1)
    

fig.suptitle("Region: %s" % (args.region,))
_ax0.legend()
_ax0.set_ylabel("Daily Precipitation [ $\\mathrm{mm} / \\mathrm{day}^{-1}$ ]")
_ax1.set_ylabel("Accumulative Rainfall [ $\\mathrm{mm} $ ]")

_ax1.set_xlabel("Time [ Pentad ]")

_ax0.grid()
_ax1.grid()

if args.plot_pentad_rng is not None:
    _ax1.set_xlim(args.plot_pentad_rng)


xlim = np.ceil(np.array(_ax1.get_xlim()))
xticks = np.arange(xlim[0], xlim[1], 6)#np.floor((xlim[1]-xlim[0])/6))
xticklabels = ["%d" % (_xtick%73) for _xtick in xticks ]
_ax1.set_xticks(xticks, labels=xticklabels) 




if args.plot_precip_rng is not None:
    _ax0.set_ylim(args.plot_precip_rng)


_ax1.set_ylim([0, 900])

if args.output != "":
    print("Saving output: ", args.output) 
    fig.savefig(args.output, dpi=200)



if not args.no_display:
    plt.show()

