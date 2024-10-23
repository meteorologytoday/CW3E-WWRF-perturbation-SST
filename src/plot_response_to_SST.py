import xarray as xr
import pandas as pd
import numpy as np
import argparse
import tool_fig_config
import wrf_load_helper 
import datetime

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--input-files', type=str, nargs="+", help='Input file.', required=True)
parser.add_argument('--output', type=str, help='Output filename in png.', default="")
parser.add_argument('--title', type=str, help='Title', default="")

parser.add_argument('--no-display', action="store_true")
parser.add_argument('--varnames', type=str, nargs="+", help='Variable names.', required=True)
parser.add_argument('--varying-param', type=str, help='Parameters. The first and second parameters will be the varying parameters while the rest stays fixed.', required=True, choices=["dSST", "Ug", "Lx"])
parser.add_argument('--fixed-params', type=str, nargs='*', help='Parameters that stay fixed.', required=True, choices=["dSST", "Ug", "Lx", "wnm"])
parser.add_argument('--thumbnail-numbering', type=str, help='Thumbnail numbering.', default="abcdefghijklmn")
parser.add_argument('--fixed-param-values', type=float, nargs="*", help='The values of the fixed parameters', default=[])
parser.add_argument('--ncols', type=int, help='The number of thumbnail columns', default=1)
parser.add_argument('--ref-exp-order', type=int, help='The reference case (start from 0) to perform decomposition', default=0)
parser.add_argument('--colors', type=str, nargs="+", help='The reference case (start from 0) to perform decomposition', required=True)
parser.add_argument('--LH-rng', type=float, nargs=2, help='The values of the LH range', default=None)
parser.add_argument('--LH-corr-rng', type=float, nargs=2, help='The values of the LH range', default=None)
parser.add_argument('--HFX-rng', type=float, nargs=2, help='The values of the HFX range', default=None)
parser.add_argument('--HFX-corr-rng', type=float, nargs=2, help='The values of the HFX range', default=None)
parser.add_argument('--spacing', type=float, help='The small separation between different variables', default=0.02)
parser.add_argument('--labels', type=str, nargs="+", help='', required=True)



args = parser.parse_args()

print(args)

Nvars = len(args.varnames)
ncols = args.ncols
nrows = int(np.ceil( Nvars / ncols))


if len(args.colors) < len(args.input_files):
    raise Exception("Not enough colors in `--colors`.")

if len(args.labels) < len(args.input_files):
    raise Exception("Not enough labels in `--labels`.")


sel_dict = {}
for i, param in enumerate(args.fixed_params):
    sel_dict[param] = args.fixed_param_values[i]

    if param == "dSST":
        sel_dict[param] /= 1e2

print("sel_dict = ", str(sel_dict))
 

print("Start loading data.")
data = []

for input_file in args.input_files:
    ds = xr.open_dataset(input_file)#, engine="scipy")
    ds = ds.sel(**sel_dict)
    data.append(ds)


#PRECIP = ds["RAINNC"] + ds["RAINC"] + ds["RAINSH"]
#PRECIP = PRECIP.rename("PRECIP")

#merge_data = [ds, PRECIP]

#ds = xr.merge(merge_data)

print(ds)




coord_x = data[0].coords[args.varying_param]


HFX_rng = [ -15, 45]
LH_rng  = [ -10, 200 ]

LH_corr_rng = [-0.5, 15]
HFX_corr_rng = [-1.0, 4.5]


HFX_rng = [ -5, 15]
LH_rng  = [ -35, 10]

LH_corr_rng = LH_rng
HFX_corr_rng = HFX_rng


if args.LH_rng is not None:
    LH_rng = args.LH_rng

if args.LH_corr_rng is not None:
    LH_corr_rng = args.LH_corr_rng

if args.HFX_rng is not None:
    HFX_rng = args.HFX_rng

if args.HFX_corr_rng is not None:
    HFX_corr_rng = args.HFX_corr_rng



plot_infos = dict(


    C_Q_WND_QOA_cx = dict(
        factor = 2.5e6,
        label = "$L_q \\, \\overline{ C'_Q \\, U' \\, Q'_{OA} }$",
        unit = "$ \\mathrm{W} / \\mathrm{m}^2 $",
        ylim = LH_corr_rng,
    ),

    WND_QOA_cx_mul_C_Q = dict(
        factor = 2.5e6,
        label = "$L_q \\, \\overline{C}_Q \\, \\overline{ U' Q'_{OA} }$",
        unit = "$ \\mathrm{W} / \\mathrm{m}^2 $",
        ylim = LH_corr_rng,
    ),



    C_Q_QOA_cx_mul_WND = dict(
        factor = 2.5e6,
        label = "$L_q \\, \\overline{U} \\, \\overline{ C_Q' Q'_{OA} }$",
        unit = "$ \\mathrm{W} / \\mathrm{m}^2 $",
        ylim = LH_corr_rng,
    ),


    C_Q_WND_cx_mul_QOA = dict(
        factor = 2.5e6,
        label = "$L_q \\, \\overline{Q}_{OA} \\, \\overline{ C_Q' U' }$",
        unit = "$ \\mathrm{W} / \\mathrm{m}^2 $",
        ylim = LH_corr_rng,
    ),

    C_Q_WND_QOA = dict(
        factor = 2.5e6,
        label = "$L_q \\overline{C}_Q \\, \\overline{U} \\, \\overline{Q}_{OA}$",
        unit = "$ \\mathrm{W} / \\mathrm{m}^2 $",
        ylim = LH_rng,
    ),

    C_H_WND_TOA = dict(
        factor = 1.0,
        label = "$\\overline{C}_H \\, \\overline{U} \\, \\overline{T}_{OA}$",
        unit = "$ \\mathrm{W} / \\mathrm{m}^2 $",
        ylim = HFX_rng,
    ),


    WND_TOA_cx_mul_C_H = dict(
        label = "$\\overline{C}_T \\, \\overline{ U' T'_{OA} }$",
        unit = "$ \\mathrm{W} / \\mathrm{m}^2 $",
        ylim = HFX_corr_rng,
    ),


    C_H_WND_TOA_cx = dict(
        label = "$\\overline{ C'_H \\, U' \\, T'_{OA} }$",
        unit = "$ \\mathrm{W} / \\mathrm{m}^2 $",
        #ylim = [-0.2 , 3.6,],
        ylim = HFX_corr_rng,
    ),

    C_H_TOA_cx_mul_WND = dict(
        label = "$\\overline{U} \\, \\overline{ C_H' T'_{OA} }$",
        unit = "$ \\mathrm{W} / \\mathrm{m}^2 $",
        #ylim = [-0.2 , 3.6,],
        ylim = HFX_corr_rng,
    ),


    C_H_WND_cx_mul_TOA = dict(
        label = "$\\overline{T}_{OA} \\, \\overline{ C_H' U' }$",
        unit = "$ \\mathrm{W} / \\mathrm{m}^2 $",
        #ylim = [-0.2 , 3.6,],
        ylim = HFX_corr_rng,
    ),


    PRECIP = dict(
        factor = 86400.0,
        label = "Precip",
        unit = "$ \\mathrm{mm} / \\mathrm{day} $",
        #ylim = [-1, 10],
    ),


    TO = dict(
        factor = 1,
        label = "$T_O$",
        unit = "$ \\mathrm{K} $",
        ylim = [0, 8],
    ),

    TA = dict(
        factor = 1,
        offset = 273.15,
        label = "$\\overline{T_A}$",
        unit = "$ \\mathrm{K} $",
        ylim = [12, 17],
    ),

    TOA_m = dict(
        factor = 1,
        label = "$T_{OA}$",
        unit = "$ \\times 10^{-4} \\, \\mathrm{m} / \\mathrm{s}^2 $",
        ylim = [0, 8],
    ),

    QO = dict(
        factor = 1e3,
        label = "$Q_O$",
        unit = "$ \\mathrm{g} / \\mathrm{kg} $",
        ylim = [0, 10],
    ),

    QA = dict(
        factor = 1e3,
        label = "$Q_A$",
        unit = "$ \\mathrm{g} / \\mathrm{kg} $",
        ylim = [-1, 12],
    ),

    QOA_m = dict(
        factor = 1,
        label = "$Q_{OA}$",
        unit = "$ \\times 10^{-4} \\, \\mathrm{m} / \\mathrm{s}^2 $",
        ylim = [0, 8],
    ),

    PBLH = dict(
        factor = 1,
        label = "$\\overline{H_\\mathrm{PBL}}$",
        unit = "$ \\mathrm{m} $",
        ylim = [0, 2500],
    ),

    HFX = dict(
        factor = 1,
        label = "$\\overline{F_\\mathrm{sen}}$",
        unit = "$ \\mathrm{W} \\, / \\, \\mathrm{m}^2 $",
        ylim = HFX_rng,
    ),

    LH = dict(
        factor = 1,
        label = "$\\overline{F_\\mathrm{lat}}$",
        unit = "$ \\mathrm{W} \\, / \\, \\mathrm{m}^2 $",
        ylim = LH_rng,
    ),

 
    HFX_approx = dict(
        factor = 1,
        label = "$Approx \\overline{F_\\mathrm{sen}}$",
        unit = "$ \\mathrm{W} \\, / \\, \\mathrm{m}^2 $",
    ),

    LH_approx = dict(
        factor = 1,
        label = "Approx $\\overline{F_\\mathrm{lat}}$",
        unit = "$ \\mathrm{W} \\, / \\, \\mathrm{m}^2 $",
        ylim = [0, 100],
        #ylim = [0, None],
    ),

    WND_m = dict(
        label = "$\\overline{U} $",
        unit = "$ \\mathrm{m} / \\mathrm{s} $",
        ylim = [10, 20],
    ),

    C_H_m = dict(
        label = "$\\overline{C_H} $",
        unit = "$ \\mathrm{m} / \\mathrm{s} $",
    ),

    IVT = dict(
        label = "IVT",
        unit = "$ \\mathrm{kg} \\, / \\, \\mathrm{m} \\, / \\, \\mathrm{s} $",
        #ylim = [010, 20],
    ),

    IWV = dict(
        label = "IWV",
        unit = "$ \\mathrm{kg} \\, / \\, \\mathrm{m}^2 $",
        #ylim = [010, 20],
    ),

)



# =================================================================
# Figure: HFX decomposition
# =================================================================


# =================================================================
print("Loading matplotlib...")
import matplotlib
if args.no_display:
    matplotlib.use('Agg')
else:
    matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
print("Done")


print("Plotting decomposition...")


figsize, gridspec_kw = tool_fig_config.calFigParams(
    w = 5,
    h = 3,
    wspace = 1.0,
    hspace = 0.7,
    w_left = 1.0,
    w_right = 1.0,
    h_bottom = 1.0,
    h_top = 1.0,
    ncol = ncols,
    nrow = nrows,
)


fig, ax = plt.subplots(
    nrows, ncols,
    figsize=figsize,
    subplot_kw=dict(aspect="auto"),
    gridspec_kw=gridspec_kw,
    constrained_layout=False,
    squeeze=False,
    sharex=False,
)


ax_flattened = ax.flatten()

for i, varname in enumerate(args.varnames):

    if varname == "BLANK":
        print("BLANK detected. Remove axis.")
        fig.delaxes(_ax)
        continue





    _ax = ax_flattened[i]
    _plot_info = plot_infos[varname]


    factor = _plot_info["factor"] if "factor" in _plot_info else 1.0
    offset = _plot_info["offset"] if "offset" in _plot_info else 0.0
    ylim   = _plot_info["ylim"] if "ylim" in _plot_info else None


    for j, _ in enumerate(args.input_files):

        ds = data[j]
        color = args.colors[j]
        



        _plot_data = (ds[varname] + offset) * factor

        _ref_m = _plot_data.sel(stat="mean")[0].to_numpy()
        print("_ref = ", _ref_m)
        
        label = "%s (%.2f %s)" % (args.labels[j], _ref_m, _plot_info["unit"])
        
        d_m = _plot_data.sel(stat="mean") - _ref_m
        d_s = _plot_data.sel(stat="std")

        _coord_x = coord_x + args.spacing * j

        _ax.errorbar(_coord_x, d_m, yerr=d_s, fmt='o-', markersize=6, capsize=5, color=color, linewidth=1.5, elinewidth=1.5, linestyle="solid", label=label)

        #for j in range(len(d_s)):
        #    _ax.plot([coord_x[j], coord_x[j]], [d_m[j] - d_s[j], d_m[j] + d_s[j]], color="gray", linestyle="solid")
            
        #_ax.scatter(coord_x, d_m, s=20)
        #_ax.scatter(coord_x, d_m, s=20)
        #_ax.plot(coord_x, d_m)

 
    if args.varying_param == "dSST":
        title_param = "$\\Delta \\mathrm{SST}$"
    elif args.varying_param == "Ug":
        title_param = "$U_g$"
    elif args.varying_param == "Lx":
        title_param = "$L$"
 
    _ax.set_title("(%s) %s as a function of %s" % (args.thumbnail_numbering[i], _plot_info["label"], title_param, ) )
    _ax.set_ylabel("[ %s ] " % (_plot_info["unit"]))
    _ax.grid(visible=True)

    if ylim is not None:
        _ax.set_ylim(ylim)

    if args.varying_param == "dSST":
        _ax.set_xlabel("$\\Delta \\mathrm{SST}$ [ $\\mathrm{K}$ ]")
    elif args.varying_param == "Ug":
        _ax.set_xlabel("$U_\\mathrm{g}$ [ $\\mathrm{m} \\, / \\, \\mathrm{s}$ ]")
    elif args.varying_param == "Lx":
        _ax.set_xlabel("$ L $ [ $\\mathrm{km} $ ]")

    _ax.legend()


if args.output != "":
    print("Saving output: ", args.output)
    fig.savefig(args.output, dpi=300)

if not args.no_display:
    print("Showing figure...")
    plt.show()



