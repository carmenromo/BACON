import re
import glob
import uproot

from datetime import datetime

import numpy as np

def extract_date_obj_and_number(filename):
    match = re.search(r"(\d{2}_\d{2}_\d{4})-file_(\d+)", filename)
    if match is None:
        match = re.search(r"(\d{2}_\d{2}_\d{4})", filename)
        file_number = 0
    else:
        file_number = int(match.group(2))
    date_str = match.group(1)    
    date_obj = datetime.strptime(date_str, "%m_%d_%Y")
    return date_obj, file_number


datafiles    = glob.glob(f'/pscratch/sd/r/romo/bacon_data/run-*.root')
#datafiles    = glob.glob(f'/Users/romoluque_c/LEGEND/BACON/new_setup/datatest/run-*.root')
sorted_files = sorted(datafiles, key=extract_date_obj_and_number)

all_dates = []
for filename in sorted_files:
    date, fnum = extract_date_obj_and_number(filename)
    if date not in all_dates:
        all_dates.append(date)

for date0 in all_dates:
    t_diffs = []
    for filename in sorted_files:
        try:
            date, fnum = extract_date_obj_and_number(filename)
            if date == date0:
                RawTree    = uproot.open(filename)['RawTree']
                timestamp1 = RawTree['eventData/evtime'].array()[ 0]
                timestamp2 = RawTree['eventData/evtime'].array()[-1]

                # Calculate the difference in seconds
                time_difference = timestamp2 - timestamp1
                t_diffs.append(time_difference)
        except:
            continue

    print(f"Filename: {date0}. Time difference in mins:", np.sum(t_diffs)/60)


