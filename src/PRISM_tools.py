import pandas as pd
import xarray as xr

def getPRISMFilename(dt):

    PRISM_filename = "/expanse/lustre/scratch/t2hsu/temp_project/data/PRISM_stable_4kmD2/PRISM_ppt_stable_4kmD2-{ymd:s}.nc".format(
       ymd = dt.strftime("%Y-%m-%d"),
    )

    return PRISM_filename


def loadDatasetWithTime(beg_dt, end_dt=None, inclusive="both"):

    if end_dt is None:
        end_dt = beg_dt

    dts = pd.date_range(beg_dt, end_dt, freq="D", inclusive=inclusive)

    filenames = [ getPRISMFilename(dt) for dt in dts ]

    ds = xr.open_mfdataset(filenames)

    return ds

