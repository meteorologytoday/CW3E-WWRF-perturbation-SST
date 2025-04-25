from pathlib import Path
import re
import xarray as xr
def parseVarname(varname_long):
    
    m = re.match(r'(?P<varname>\w+)(::(?P<level>[0-9]+))?', varname_long)

    varname = None
    level = None
    if m:
        
        d = m.groupdict()
        print("GROUPDICT: ", d)
        varname = d["varname"]
        level = d["level"]
        if level is not None:
            level = int(level)
        
    else:
        raise Exception("Varname %s cannot be parsed. ") 

    return varname, level


def parseRanges(input_str):
    numbers = []
    for part in input_str.split(','):
        part = part.strip()
        if '-' in part:
            start, end = map(int, part.split('-'))
            numbers.extend(range(start, end + 1))
        else:
            numbers.append(int(part))
    return numbers

def parseExpblob(expblob_str):
    
    expsets = expblob.split("|")

    results = []
    for expset in expset:
        print(expset)
        expname, group, ens_rng = expset.split("/")
        
        expname = expname.strip()
        group   = group.strip()
        ens_rng = ens_rng.strip()
        results.append((expname, group, parseRanges(ens_rng)))
    
    return results


def genWRFEnsRelPathDir(expname, group, ens_id, root=".", filename_style="default"):
   
    filename_style = filename_style.rstrip()
 
    if filename_style == "default":
        result_dir = Path(root) / expname / "runs" / group / f"{ens_id:02d}" / "output" / "wrfout"
    elif filename_style == "CW3E-WestWRF":
        result_dir = Path(root) / expname / f"{group:s}{ens_id:03d}"
    else:
        raise Exception("Unknown style: %s" % (filename_style,)) 

    return result_dir

"""
def genEnsStatFilename(expname, group, varname, dt, root=".", level=None):

    root = Path(root)
    
    # Detecting
    output_file = root / expname / group / ("{varname:s}{level:s}-{time:s}.nc".format(
        varname = varname,
        level = "" if level is None else "::%d" % level,
        time = dt.strftime("%Y-%m-%dT%H:%M:%S"),
    ))

    return output_file     
"""

def genEnsFilename(expname, group, ens_id, varname, dt, root="."):

    root = Path(root)
    
    # Detecting
    output_file = root / expname / group / f"{ens_id:02d}" / ("{varname:s}-{time:s}.nc".format(
        varname = varname,
        time = dt.strftime("%Y-%m-%dT%H:%M:%S"),
    ))

    return output_file     


def loadGroup(expname, group, ens_ids, varname, dt, root="."):

    files = [
        genEnsFilename(expname, group, ens_id, varname, dt, root=root)
        for ens_id in ens_ids 
    ]

    ds = xr.open_mfdataset(files)

    return ds

