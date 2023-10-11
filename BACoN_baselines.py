
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

total_SiPMs   = 9
dead_SiPMs    = [3]
mean_bsl_mode = []
mean_bsl_mean = []
all_channels  = []
range_wf      = (0, 700)

for channel in range(9):
    print(channel)
    if channel in dead_SiPMs:
        continue
    try:
        all_raw_wfs = np.array(RawTree[f'chan{channel}/rdigi'].array())
        bsl_raw_wfs_mode = [pf.compute_baseline(wf[range_wf[0]:range_wf[1]], mode=True)  for wf in all_raw_wfs] #list(map(pf.compute_baseline, all_raw_wfs))
        bsl_raw_wfs_mean = [pf.compute_baseline(wf[range_wf[0]:range_wf[1]], mode=False) for wf in all_raw_wfs]
        mean_bsl_mode.append(np.mean(bsl_raw_wfs_mode))
        mean_bsl_mean.append(np.mean(bsl_raw_wfs_mean))
        all_channels .append(channel)
    except ValueError:
        continue

pmt_channel = 12
all_raw_wfs = np.array(RawTree[f'chan{pmt_channel}/rdigi'].array())
bsl_raw_wfs_mode = [pf.compute_baseline(wf[range_wf[0]:range_wf[1]], mode=True)  for wf in all_raw_wfs] #list(map(pf.compute_baseline, all_raw_wfs))
bsl_raw_wfs_mean = [pf.compute_baseline(wf[range_wf[0]:range_wf[1]], mode=False) for wf in all_raw_wfs]
mean_bsl_mode.append(np.mean(bsl_raw_wfs_mode))
mean_bsl_mean.append(np.mean(bsl_raw_wfs_mean))
all_channels .append(pmt_channel)
print(pmt_channel)

mean_bsl_mode = np.array(mean_bsl_mode)
mean_bsl_mean = np.array(mean_bsl_mean)
all_channels  = np.array(all_channels)

np.savez(outfile, 
         mean_bsl_mode=mean_bsl_mode,
         mean_bsl_mean=mean_bsl_mean,
         all_channels =all_channels)