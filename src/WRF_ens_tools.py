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


def genWRFEnsRelPathDir(expname, group, ens_id, root="."):
    
    result_dir = Path(root) / expname / "runs" / group / f"{ens_id:02d}" / "output" / "wrfout"

    return result_dir

def genEnsStatFilename(expname, group, varname, dt, root=".", level=None):

    root = Path(root)
    
    # Detecting
    output_file = root / expname / group / ("{varname:s}{level:s}-{time:s}.nc".format(
        varname = varname,
        level = "" if level is None else "::%d" % level,
        time = dt.strftime("%Y-%m-%dT%H:%M:%S"),
    ))

    return output_file     

