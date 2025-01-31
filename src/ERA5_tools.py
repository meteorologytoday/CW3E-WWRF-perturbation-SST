import pandas as pd
import xarray as xr
import os
import numpy as np


archive_root = os.path.join("data", "ERA5")

mapping_longname_shortname = {
    'geopotential'                  : 'z',
    '10m_u_component_of_wind'       : 'u10',
    '10m_v_component_of_wind'       : 'v10',
    'mean_sea_level_pressure'       : 'msl',
    '2m_temperature'                : 't2m',
    'sea_surface_temperature'       : 'sst',
    'specific_humidity'             : 'q',
    'u_component_of_wind'           : 'u',
    'v_component_of_wind'           : 'v',
    'mean_surface_sensible_heat_flux'    : 'msshf',
    'mean_surface_latent_heat_flux'      : 'mslhf',
    'mean_surface_net_long_wave_radiation_flux'  : 'msnlwrf',
    'mean_surface_net_short_wave_radiation_flux' : 'msnswrf',
    "convective_precipitation"      : 'cp',
    "convective_rain_rate"          : 'crr',
    "large_scale_precipitation"     : 'lsp',
    "large_scale_rain_rate"         : 'lsrr',
    "total_precipitation"           : 'tp',
}

file_prefix = "ERA5"

def generate_filename(var_longname, dt, freq):
   
    dt_str = pd.Timestamp(dt).strftime("%Y-%m-%d_%H")

    save_dir = os.path.join(
        archive_root,
        freq,
        var_longname,
    )
  
    filename = os.path.join(
        save_dir,
        "{file_prefix:s}-{var_longname:s}-{time:s}.nc".format(
            file_prefix = file_prefix,
            var_longname = var_longname,
            time = dt_str,
        )
    )

    return filename

def open_dataset(varname_longname, dt, freq):
  
    loading_filename = generate_filename(varname_longname, dt, freq) 
    ds = xr.open_dataset(loading_filename)
    
    return ds    

def open_dataset_range(varname_longname, dts, freq):
   
    fnames = [ generate_filename(varname_longname, dt, freq) for dt in dts ] 
    ds = xr.open_mfdataset(loading_filename)
    
    return ds    


if __name__ == "__main__":   
    
    date = "2000-01-01"
    
    ds = open_dataset("total_precipitation", date, "24hr")

    print(ds)
 
