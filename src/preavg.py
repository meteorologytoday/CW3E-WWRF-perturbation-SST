import traceback
from multiprocessing import Pool
import multiprocessing
from pathlib import Path
import os

import xarray as xr
import pandas as pd
import numpy as np
import argparse
import tool_fig_config
import wrf_load_helper 
import datetime
import os
import shared_constants


cp_a  = shared_constants.cp_a
Lq = shared_constants.Lq
g0 = shared_constants.g0

def integrateVertically(X, ds, avg=False):

    MUB = ds.MUB
    DNW = ds.DNW
    MU_FULL = ds.MU + ds.MUB
    MU_STAR = MU_FULL / MUB
    integration_factor = - MUB * DNW / g0  # notice that DNW is negative, so we need to multiply by -1

    X_STAR = X * MU_STAR
    X_INT = (integration_factor * X_STAR).sum(dim="bottom_top")

    if avg:
        sum_integration_factor = integration_factor.sum(dim="bottom_top")
        X_INT = X_INT / sum_integration_factor

    return X_INT


def avgData(
    input_dir,
    output_filename,
    exp_beg_time,
    wrfout_data_interval,
    frames_per_wrfout_file,
    reltime_rngs,
):
    
    result = dict(output_filename=output_filename, status='UNKNOWN')

    try:
    
        exp_beg_time = pd.Timestamp(exp_beg_time)
    
        merge_data = []


        for i, reltime_rng in enumerate(reltime_rngs):
            
            selected_time_beg = exp_beg_time + reltime_rng[0]
            selected_time_end = exp_beg_time + reltime_rng[1]

            print("[%d] Analyzing time inverval: %s ~ %s" % (
                i,
                selected_time_beg.strftime("%Y-%m-%d_%H:%M:%S"), 
                selected_time_end.strftime("%Y-%m-%d_%H:%M:%S"),           
            ))
             
            ds = avgData_subset(
                input_dir,
                exp_beg_time,
                wrfout_data_interval,
                frames_per_wrfout_file,
                reltime_rng,
            )
    
            merge_data.append(ds)



        print("Merging data...")
        new_ds = xr.merge(merge_data)

            
        full_range_time_beg = exp_beg_time + reltime_rngs[0][0]
        full_range_time_end = exp_beg_time + reltime_rngs[-1][1]
        new_ds.attrs["time_beg"] = full_range_time_beg.strftime("%Y-%m-%d_%H:%M:%S"),
        new_ds.attrs["time_end"] = full_range_time_end.strftime("%Y-%m-%d_%H:%M:%S"),
 
        print("Writing file: %s" % (output_filename,))
        new_ds.to_netcdf(
            output_filename,
            unlimited_dims="time",
            encoding={'time':{'units':'hours since 2001-01-01'}}
        )

        
        result['status'] = 'OK'

    
    except Exception as e:
        result['status'] = 'ERROR'
        traceback.print_exc()

    return result

def avgData_subset(
    input_dir,
    exp_beg_time,
    wrfout_data_interval,
    frames_per_wrfout_file,
    reltime_rng,
):



    time_beg = exp_beg_time + reltime_rng[0]
    time_end = exp_beg_time + reltime_rng[1]
    
    time_beg_str = time_beg.strftime("%Y-%m-%dT%H:%M:%S")
    time_end_str = time_end.strftime("%Y-%m-%dT%H:%M:%S")

    time_bnd = xr.DataArray(
        name="time_bnd",
        data=[time_beg, time_end],
        dims=["time_bnd",],
        coords=dict(
            
        ),
        attrs=dict(
            reference_time=pd.Timestamp("1970-01-01"),
        ),
    )

    def relTimeInHrs(t):
        return (t - exp_beg_time.to_datetime64()) / np.timedelta64(1, 'h')


    data_interval = pd.Timedelta(seconds=wrfout_data_interval)

    # Loading data
        
    print("Loading directory: %s" % (input_dir,))

    wsm = wrf_load_helper.WRFSimMetadata(
        start_datetime = exp_beg_time,
        data_interval = data_interval,
        frames_per_file = frames_per_wrfout_file,
    )

    ds = wrf_load_helper.loadWRFDataFromDir(
        wsm, 
        input_dir,
        beg_time = time_beg,
        end_time = time_end,
        prefix="wrfout_d01_",
        avg="ALL",
        verbose=False,
        inclusive="both",
    )

    merge_data = []
    # Integrate water vapor, TKE

    for species in ["VAPOR", "CLOUD", "RAIN", "ICE", "SNOW"]:
        QX_varname = "Q%s" % (species,)
        QX_TTL_varname = "Q%s_TTL" % (species,)

        if not ( QX_varname in ds ):
            print("Variable %s not in file. Skip it." % (QX_varname,))
            continue

        QX_TTL = integrateVertically(ds[QX_varname], ds, avg=False).mean(dim="west_east").rename(QX_TTL_varname)
        merge_data.append(QX_TTL)

    THETA_MEAN = integrateVertically(300.0 + ds.T, ds, avg=True).mean(dim="west_east").rename("THETA_MEAN")
    merge_data.append(THETA_MEAN)

    # Convert V_T
    V_T = xr.zeros_like(ds["T"]).rename("V_T")
    _tmp = ds["V"].to_numpy()
    V_T[:, :, :, :] = _tmp[:, :, 0:1, :]

    # Convert U_T
    _tmp = ds["U"].to_numpy()
    U_T = xr.zeros_like(ds["T"]).rename("U_T")

    U_T[:, :, :, :] = (_tmp[:, :, :, 1:] + _tmp[:, :, :, :-1]) / 2
    U_T = U_T.rename("U_T")
 
    # IWV
    IWV = integrateVertically(ds["QVAPOR"], ds, avg=False).rename("IWV")
 
    # IVT
    IVT_x = integrateVertically(ds["QVAPOR"] * U_T, ds, avg=False)
    IVT_y = integrateVertically(ds["QVAPOR"] * V_T, ds, avg=False)
    
    # code order matters here
    IVT = ((IVT_x**2 + IVT_y**2)**0.5).mean(dim="west_east").rename("IVT")
    IVT_x = IVT_x.mean(dim="west_east").rename("IVT_x")
    IVT_y = IVT_y.mean(dim="west_east").rename("IVT_y")

    merge_data.extend([IWV, IVT, IVT_x, IVT_y])

    if "QKE" in ds:
        TKE_TTL = integrateVertically(ds.QKE/2, ds, avg=False).mean(dim="west_east").rename("TKE_TTL")
        merge_data.append(TKE_TTL)


    # Merging data
    extra_ds = xr.merge(merge_data)
    extra_ds = extra_ds.mean(dim='west_east', skipna=True, keep_attrs=True)
  
    # Unset XLAT and XLONG as coordinate
    # For some reason they disappeared after taking the time mean


    ds = xr.merge([ds, extra_ds])


    ds = ds.reset_coords(names=['XLAT', 'XLONG'])
    ds = ds.assign_coords(
        XLAT=( ds.XLAT.dims,   ds.XLAT.data), 
        XLONG=( ds.XLONG.dims, ds.XLONG.data),
    )
    #print("Check the coordinates...: ", ds.coords)


    
    ds = ds.compute()
    
    return ds


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--input-dir', type=str, help='Input directories.', required=True)
    parser.add_argument('--output-dir', type=str, help='Output directory.', required=True)
    parser.add_argument('--exp-beg-time', type=str, help='Experiment begin time.', required=True)
    parser.add_argument('--wrfout-data-interval', type=int, help='Time interval between each adjacent record in wrfout files in seconds.', required=True)
    parser.add_argument('--frames-per-wrfout-file', type=int, help='Number of frames in each wrfout file.', required=True)

    parser.add_argument('--time-rng', type=int, nargs=2, help="Time range in hours after --exp-beg-time", required=True)
    parser.add_argument('--time-avg-interval', type=int, help="The interval of time to do the average. Unit is in minutes. If it is not specified or 0 then all data is averaged.", default=0)
    parser.add_argument('--output-count', type=int, help="The numbers of output in a file.", default=1)
    parser.add_argument('--nproc', type=int, help="Number of parallel CPU.", default=1)
    
    args = parser.parse_args()

    print(args)
   
    # Decide the time decomposition
    print("Processing ...")

    exp_beg_time = pd.Timestamp(args.exp_beg_time)
    reltime_beg = pd.Timedelta(hours=args.time_rng[0])
    reltime_end = pd.Timedelta(hours=args.time_rng[1])
    time_avg_interval = pd.Timedelta(minutes=args.time_avg_interval)
    

    if time_avg_interval / pd.Timedelta(seconds=1) == 0:  # if not specified or 0
        print("The parameter `--time-avg-interval` is zero, assume the average is the entire interval.")
        time_avg_interval = time_end - time_beg
   
    print("Create dir: %s" % (args.output_dir,))
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

 

    input_args = []
    
    number_of_intervals = (reltime_end - reltime_beg) / time_avg_interval
    if number_of_intervals % 1 != 0:
        raise Exception("Error: the interval of time_rng is not a multiple of time_avg_interval. Ratio: %.f" % (number_of_intervals,)) 

    number_of_intervals = int( np.ceil(number_of_intervals) ) 
    
    number_of_output_files = number_of_intervals / args.output_count
    if number_of_intervals % 1 != 0:
        raise Exception("Error: the number_of_intervals = %d is not a multiple of number_of_output_files = %d " % (
            number_of_intervals,
            number_of_output_files,
        ))
    number_of_output_files = int(number_of_output_files)
    
    print("There will be %d output files. Each contains %d data points" % (number_of_output_files, args.output_count,))
    
    time_interval_per_file = time_avg_interval * args.output_count
    for i in range(number_of_output_files):

        reltime_rngs = []
            
        reltime_beg_of_file = reltime_beg + i * time_interval_per_file
        time_beg_of_file = exp_beg_time + reltime_beg_of_file

        for j in range(args.output_count):
            
            reltime_rngs.append(
                [ 
                    reltime_beg_of_file + j     * time_avg_interval,
                    reltime_beg_of_file + (j+1) * time_avg_interval,
                ]
            )

        filename = os.path.join(
            args.output_dir,
            "wrfout_d01_{timestr:s}".format(
                timestr = time_beg_of_file.strftime("%Y-%m-%d_%H:%M:%S"),
            )
        )

        if os.path.exists(filename):
            print("File %s already exists. Skip it." % (filename,))
            continue

       
        input_args.append(
            (
                args.input_dir,
                filename,
                args.exp_beg_time,
                args.wrfout_data_interval,
                args.frames_per_wrfout_file,
                reltime_rngs,
            )
        )
    
    
    
    failed_files = []
    with Pool(processes=args.nproc) as pool:

        results = pool.starmap(avgData, input_args)

        for i, result in enumerate(results):
            if result['status'] != 'OK':
                print('!!! Failed to generate output : %s.' % (result['output_filename'],))
                failed_files.append(result['output_filename'])


    print("Tasks finished.")

    print("Failed files: ")
    for i, failed_file in enumerate(failed_files):
        print("%d : %s" % (i+1, failed_file,))


    print("Done")
    
    
    
    
