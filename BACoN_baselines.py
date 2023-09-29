
import sys
import uproot

import numpy as np

import peak_functions as pf

arguments = pf.parse_args(sys.argv)
in_path   = arguments.in_path
file_name = arguments.file_name
out_path  = arguments.out_path

if file_name.endswith(".root"):
    file_name = file_name[:-5]
filename = f"{in_path}/{file_name}.root"
infile   = uproot.open(filename)
RawTree  = infile['RawTree']

outfile = f"{out_path}/BACoN_baselines_{file_name}"

total_SiPMs    = 9
dead_SiPMs     = [3]
mean_baselines = []

for channel in range(9):
    print(channel)
    if channel in dead_SiPMs:
        continue
    try:
        all_raw_wfs = np.array(RawTree[f'chan{channel}/rdigi'].array())
        bsl_raw_wfs = list(map(compute_baseline, all_raw_wfs))
        mean_baselines.append(np.mean(bsl_raw_wfs))
    except ValueError:
        continue

pmt_channel = 12
all_raw_wfs = np.array(RawTree[f'chan{pmt_channel}/rdigi'].array())
bsl_raw_wfs = list(map(compute_baseline, all_raw_wfs))
mean_baselines.append(np.mean(bsl_raw_wfs))
print(pmt_channel)

mean_baselines = np.array(mean_baselines)
all_channels   = np.array([0, 1, 2, 4, 5, 6, 7, 8, 12])

np.savez(outfile, 
         mean_baselines=mean_baselines,
         all_channels  =all_channels)