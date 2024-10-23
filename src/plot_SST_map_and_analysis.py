import numpy as np
import xarray as xr
import scipy
import traceback
from pathlib import Path
import pandas as pd
import argparse
import colorblind

# Find the first value that is True
def findfirst(xs):

    for i, x in enumerate(xs):
        if x:
            return i

    return -1



parser = argparse.ArgumentParser(
                    prog = 'plot_skill',
                    description = 'Plot prediction skill of GFS on AR.',
)

parser.add_argument('--input-file', type=str, help='Input file', required=True)
parser.add_argument('--output', type=str, help='Output file', default="")
parser.add_argument('--lat-rng', type=float, nargs=2, help='The x axis range to be plot in km.', default=[None, None])
parser.add_argument('--lat-avg-interval', type=float, help='The x axis range to be plot in km.', default=0)
parser.add_argument('--lon-rng', type=float, nargs=2, help='The x axis range to be plot in km.', default=[None, None])
parser.add_argument('--ylim', type=float, nargs=2, help='The x axis range to be plot in km.', default=[None, None])
parser.add_argument('--cutoff-wvlen', type=float, help='The cutoff wavelength.', default=1.1)
parser.add_argument('--no-display', action="store_true")

args = parser.parse_args()
print(args)

linecolors = ['orange', 'blue', 'reddishpurple', 'bluishgreen', 'vermillion']


lat_span = args.lat_rng[1] - args.lat_rng[0]
lon_span = args.lon_rng[1] - args.lon_rng[0]

ds = xr.open_dataset(args.input_file, decode_times=False).isel(depth=0, time=0)

lon = ds.coords["lon"].to_numpy()

print("Making longitude 0 the first element...")
first_positive_idx = findfirst( lon >= 0 )
if first_positive_idx != -1:
    roll_by = - first_positive_idx
    ds = ds.roll(lon=roll_by).assign_coords(
        coords = {
            "lon" : np.roll(
                lon % 360,
                roll_by,
            )
        }
    )

print("Selecting data...")
SST = ds["water_temp"].sel(lat=slice(*args.lat_rng), lon=slice(*args.lon_rng))

lat = SST.coords["lat"].to_numpy()
number_of_lat_rngs = ( args.lat_rng[1] - args.lat_rng[0] ) / args.lat_avg_interval
if number_of_lat_rngs % 1 != 0:
    print("Warning: number_of_lat_rngs = %f, it is not an integer. " % (args.lat_avg_interval,))

number_of_lat_rngs = int(np.ceil(number_of_lat_rngs))

lat_band_idxes = []
lat_band_rngs = []
for i in range(number_of_lat_rngs):
    lat_subrng = [
        args.lat_rng[0] +  i    * args.lat_avg_interval ,  
        args.lat_rng[0] + (i+1) * args.lat_avg_interval ,  
    ]

    lat_band_rngs.append(lat_subrng)
    lat_band_idxes.append((lat >= lat_subrng[0]) & (lat < lat_subrng[1]))


print(SST.to_numpy().shape)
coords = { varname : SST.coords[varname].to_numpy() for varname in ds.coords}

dlon = coords["lon"][1] - coords["lon"][0]
L_lon = dlon * len(coords["lon"])

SST = SST.to_numpy()
SST_nonan = SST.copy()
SST_nonan[np.isnan(SST_nonan)] = 0.0

SST_lp = scipy.ndimage.uniform_filter(SST_nonan, size=(51, 25), mode='reflect')
#SST_lp = scipy.ndimage.uniform_filter(SST_nonan, size=(25, 13), mode='reflect')
SST_hp = SST - SST_lp


# Spectral analysis
if np.any(np.isnan(SST)):
    print("Warning: SST contains NaN.")


#print("dlon : ", dlon)
#print("fftfreq : ", np.fft.fftfreq(SST.shape[1]))
#SST_zm = np.nanmean(SST, axis=1, keepdims=True)
#SST_za = SST - SST_zm

SST_za = scipy.signal.detrend(SST, axis=1)

dft_coe = np.fft.fft(SST_za, axis=1)
fftfreq = np.fft.fftfreq(SST_za.shape[1]) / dlon  # cycle / total_longitude_width
#wvlens = dlon / np.fft.fftfreq(SST_za.shape[1])
wvlens = fftfreq ** (-1)


plot_wvlens  = wvlens[0:(len(wvlens) // 2)]
plot_specden = dft_coe[:, 0:(len(wvlens) // 2)]
plot_specden = np.abs(plot_specden)**2

selected_idx = plot_wvlens < args.cutoff_wvlen
plot_wvlens = plot_wvlens[selected_idx]
plot_specden = plot_specden[:, selected_idx]

plot_freqs = 1.0 / plot_wvlens

print("plot_wvlens: ", plot_wvlens)


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

ncol = 3
nrow = 1

h = 4.0
w_map = h * lon_span / lat_span
w_spec = h * 1.0



figsize, gridspec_kw = tool_fig_config.calFigParams(
    w = [w_map,]*2 + [w_spec,],
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
    sharex=False,
)

ax_flatten = ax.flatten()

#fig.suptitle("File: %s" % (args.input_file,))

SST_levs = np.linspace(5, 25, 41)
SST_lp_levs = np.linspace(5, 25, 11)
SST_hp_levs = np.linspace(-1, 1, 21) * 3
SST_hp_ticks = np.linspace(-1, 1, 9) * 3
SST_hp_cntr_levs = np.array([-1, 1]) * 0.5

_ax = ax_flatten[0]
mappable = _ax.contourf(coords["lon"], coords["lat"], SST, SST_levs, cmap="gnuplot", extend="both")
cax = tool_fig_config.addAxesNextToAxes(fig, _ax, "bottom", thickness=0.03, spacing=0.17)
cb = plt.colorbar(mappable, cax=cax, orientation="horizontal", pad=0.00)
cb.ax.set_xlabel("SST [ ${}^\\circ\\mathrm{C}$ ]")


_ax = ax_flatten[1]
#mappable = _ax.contourf(coords["lon"], coords["lat"], SST_hp, SST_hp_levs, cmap="bwr", extend="both")
mappable = _ax.contourf(coords["lon"], coords["lat"], SST_za, SST_hp_levs, cmap="bwr", extend="both")
cax = tool_fig_config.addAxesNextToAxes(fig, _ax, "bottom", thickness=0.03, spacing=0.17)
cb = plt.colorbar(mappable, cax=cax, orientation="horizontal", pad=0.00)
cb.ax.set_xlabel("SST Anomaly [ K ]")

#cs = _ax.contour(coords["lon"], coords["lat"], SST_hp, SST_hp_cntr_levs, colors="black", linewidths=0.5)

for _ax in ax_flatten[:2]:
    _ax.grid()
    _ax.set_xlabel("Longitude [ deg ]")
    _ax.set_ylabel("Latitude [ deg ]")
    _ax.set_yticks(np.linspace(args.lat_rng[0], args.lat_rng[1], number_of_lat_rngs + 1))

ax_flatten[0].set_title("(a) Full SST")
ax_flatten[1].set_title("(b) Zonally detrended SST anomaly")



_ax = ax_flatten[2]
_ax2 = _ax.twiny()

for i in range(number_of_lat_rngs):
    _ax.plot(np.log10(plot_freqs), np.log10(np.mean(plot_specden[lat_band_idxes[i], :], axis=0)), color=colorblind.BW8color[linecolors[i]], alpha=0.9, linestyle="dashed", linewidth=2, label="%d~%d" % (lat_band_rngs[i][0], lat_band_rngs[i][1]))

_ax.plot(np.log10(plot_freqs), np.log10(np.mean(plot_specden, axis=0)), "k-", linewidth=2, label="All")
_ax.legend()

xticks = np.array([-1, -.5, 0, .5, 1])
yticks = np.array([0, 1, 2, 3])


_ax.set_xticks(ticks=xticks, labels=["$10^{%.1f}$" % (d, ) for d in xticks])
_ax.set_yticks(ticks=yticks, labels=["$10^{%d}$" % (d, ) for d in yticks])

_ax2.set_xticks(ticks=xticks, labels=["$10^{%.1f}$" % (-d, ) for d in xticks])
_ax2.set_xlabel("Wavelength [deg / cycle]")

for __ax in [_ax, _ax2]:
    _ax.set_xlim([xticks[0], xticks[-1]])

_ax.set_ylim(args.ylim)

_ax.grid(True)
_ax.set_xlabel("Wavenumber [cycle / deg]")

_ax.set_ylabel("Spectral Intensity [$ \\mathrm{K}^2 $]")
_ax.set_title("(c) Spectrum density", pad=15)

if args.output != "":
    print("Saving output: ", args.output) 
    fig.savefig(args.output, dpi=200)



if not args.no_display:
    plt.show()

