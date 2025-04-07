from pathlib import Path



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


def genWRFEnsRelPathDir(expname, group, subgroup, ens_id, style="v1", root="."):
    
    root = Path(root)

    if subgroup == "BLANK":
        extra_casename = f"{group:s}_"
        subgroup_dir = ""

    else:
        extra_casename = f"{group:s}_{subgroup:s}_"
        subgroup_dir = subgroup

    if style == "v1":
        result_dir = root / expname / group / "runs" / subgroup_dir / f"{extra_casename:s}ens{ens_id:02d}" / "output" / "wrfout"
        
    elif style == "v2":
        result_dir = root / expname / group / "runs" / subgroup_dir / f"ens{ens_id:02d}" / "output" / "wrfout"
    
    else:
        raise Exception("Unknown style `%s`" % (style,))
 

    return result_dir

def genEnsStatFilename(expname, group, subgroup, varname, dt, root="."):

    root = Path(root)
    full_group_name = f"{group:s}_{subgroup:s}" 
    
    # Detecting
    output_file = root / expname / group / subgroup / ("{varname:s}-{time:s}.nc".format(
        varname = varname,
        time = dt.strftime("%Y-%m-%dT%H:%M:%S"),
    ))

    return output_file      
