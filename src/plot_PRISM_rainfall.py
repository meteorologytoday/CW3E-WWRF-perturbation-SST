import numpy as np
import xarray as xr
import scipy
import traceback
from pathlib import Path
import pandas as pd
import argparse
import geopandas as gpd

import PRISM_tools

parser = argparse.ArgumentParser(
                    prog = 'plot_skill',
                    description = 'Plot prediction skill of GFS on AR.',
)

parser.add_argument('--date-rng', type=str, nargs=2, help='Input file', required=True)
parser.add_argument('--output', type=str, help='Output file', default="")
parser.add_argument('--lat-rng', type=float, nargs=2, help='The x axis range to be plot in km.', default=[None, None])
parser.add_argument('--lon-rng', type=float, nargs=2, help='The x axis range to be plot in km.', default=[None, None])
parser.add_argument('--precip-levs', type=float, nargs="*", help='Bnds of precipitation.', default=None)
parser.add_argument('--no-display', action="store_true")

args = parser.parse_args()
print(args)

beg_dt = pd.Timestamp(args.date_rng[0])
end_dt = pd.Timestamp(args.date_rng[1])

if args.precip_levs is None:
    precip_levs = np.linspace(0, 500, 21)
else:
    precip_levs = args.precip_levs


needed_shapes = dict(
    CA="data/shapefiles/ca_state.zip",
    Dam_Oroville="data/shapefiles/OrovilleDam.zip",
    Dam_Shasta="data/shapefiles/ShastaDam.zip",
    Dam_NewMelones="data/shapefiles/NewMelonesDam.zip",
    Dam_SevenOaks="data/shapefiles/SevenOaksDam.zip",
)


print("Load CA shapefile")
shps = {
    k : gpd.read_file(v).to_crs(epsg=4326) for k, v in needed_shapes.items()
}

ds = PRISM_tools.loadDatasetWithTime(beg_dt, end_dt, inclusive="both")
ds = ds.sum(dim="time")


print(ds)


lat = ds.coords["lat"]
lon = ds.coords["lon"]


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



plot_lon_l = args.lon_rng[0]
plot_lon_r = args.lon_rng[1]
plot_lat_b = args.lat_rng[0]
plot_lat_t = args.lat_rng[1]

lat_span = args.lat_rng[1] - args.lat_rng[0]
lon_span = args.lon_rng[1] - args.lon_rng[0]
cent_lon=180.0

proj = ccrs.PlateCarree(central_longitude=cent_lon)
proj_norm = ccrs.PlateCarree()


ncol = 1
nrow = 1

h = 4.0
w_map = h * lon_span / lat_span

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
    subplot_kw=dict(projection=proj, aspect="auto"),
    gridspec_kw=gridspec_kw,
    constrained_layout=False,
    squeeze=False,
    sharex=False,
)

ax_flatten = ax.flatten()

_ax = ax_flatten[0]
mappable = _ax.contourf(lon, lat, ds["total_precipitation"], precip_levs, cmap="cmo.rain", extend="both", transform=proj_norm)
cax = tool_fig_config.addAxesNextToAxes(fig, _ax, "bottom", thickness=0.03, spacing=0.17)
cb = plt.colorbar(mappable, cax=cax, orientation="horizontal", pad=0.00)
cb.ax.set_xlabel("Total Precipitation [ $\\mathrm{mm}$ ]")


for shp_name, shp in shps.items():
    print("Putting geometry %s" % (shp_name,))
    print(shp)
    _ax.add_geometries(
        shp["geometry"], crs=proj_norm, facecolor="none", edgecolor="black"
    )

"""
for geo_shp in [CA_sf,]:
    
    df = geo_shp.geometry.get_coordinates()

    x = df['x'].to_numpy() % 360
    y = df['y'].to_numpy()

    print(np.array(x))
    print(np.array(y))

    _ax.plot(x, y, "k-", transform=proj_norm)

"""

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


ax_flatten[0].set_title("PRISM %s~%s" % (
    beg_dt.strftime("%Y/%m/%d"),
    end_dt.strftime("%Y/%m/%d"),
))


if args.output != "":
    print("Saving output: ", args.output) 
    fig.savefig(args.output, dpi=200)



if not args.no_display:
    plt.show()

