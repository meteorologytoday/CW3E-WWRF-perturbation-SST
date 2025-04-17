import xarray as xr
import numpy as np
from pathlib import Path
from scipy import sparse

def constructAvgMtx(lat_idx, lon_idx, nbox_lat, nbox_lon):
     
    Ny, Nx = lat_idx.shape
    original_total_grids = Ny * Nx
    total_boxes = nbox_lat * nbox_lon
    
    #print("Ny, Nx = ", lat_idx.shape)
   
    # This is the numbering of each regridded boxes 
    regrid_row_idx = np.arange(total_boxes).reshape((nbox_lat, nbox_lon)) 

    # This is the numbering of the original grids
    original_grid_idx = np.arange(original_total_grids).reshape((Ny, Nx))
   
    #print("shape of regrid_row_idx: ", regrid_row_idx.shape) 



    row_idxes = []
    col_idxes = []
    for i in range(Nx):
        for j in range(Ny):
 
            _lon_idx = lon_idx[j, i]
            _lat_idx = lat_idx[j, i]
          
            if _lon_idx >= 0 and _lat_idx >= 0:
 
                _row_idx = regrid_row_idx[ _lat_idx, _lon_idx]
                _col_idx = original_grid_idx[j, i]

                row_idxes.append(_row_idx)
                col_idxes.append(_col_idx)
                 
    vals = np.ones((len(row_idxes), ))
    

    avg_mtx = sparse.coo_array((vals, (row_idxes, col_idxes)), shape=(total_boxes, original_total_grids), dtype=np.float32)
    
    wgt = avg_mtx.sum(axis=1)

    mask = np.zeros((nbox_lat, nbox_lon), dtype=np.int32)
    mask[np.reshape(wgt, (nbox_lat, nbox_lon)) != 0] = 1
    
    wgt_mtx = sparse.dia_array( ([wgt**(-1),], [0,]), shape=(total_boxes, total_boxes))
    avg_mtx = wgt_mtx @ avg_mtx 

    regrid_info = dict(
        avg_mtx = avg_mtx,
        shape_original = (Ny, Nx),
        shape_regrid = (nbox_lat, nbox_lon),
        mask = mask,
    )

    return regrid_info

def regrid(regrid_info, arr):
   
    if len(arr.shape) == 3:
        
        print("The input array is three dimension. Treating the first dimenion as the time")
        result = [
            regrid(regrid_info, arr[i, :, :])
            for i in range(arr.shape[0])
        ]

        result = np.stack(result, axis=0)
        
        return result
        
        
    flattened_arr = np.array(arr).flatten()

    if len(flattened_arr) != regrid_info["shape_original"][0] * regrid_info["shape_original"][1]:
        print(regrid_info["shape_original"], "; ", arr.shape)
        raise Exception("Dimension of input array does not match avg_info.")
    
    result = regrid_info["avg_mtx"] @ np.array(arr).flatten()
    result = np.reshape(result, regrid_info["shape_regrid"])
    result[regrid_info['mask'] == 0] = np.nan

    return result 

