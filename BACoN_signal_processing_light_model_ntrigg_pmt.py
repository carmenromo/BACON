
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

outfile = f"{out_path}/BACoN_sig_processing_light_model_ntrigg_{file_name}"

total_SiPMs = 9
dead_SiPMs  = [3]
peak_range  = (0, None)
num_wfs     = None # All wfs to be analyzed
sipm_thr    = 50 #ADCs
peak_sep    = 10

int_subt_wfs_filt = [np.array([]) for _ in range(total_SiPMs)]
num_triggers_ch   = [np.array([]) for _ in range(total_SiPMs)]

for channel in range(total_SiPMs):
    print(channel)
    if channel in dead_SiPMs:
        continue
    try:
        _, subt_wfs_filt, all_peaks = pf.get_peaks_using_peakutils(RawTree, channel, num_wfs=num_wfs, sipm_thr=sipm_thr, peak_range=peak_range)
        int_subt_wf_filt = np.sum(subt_wfs_filt, axis=0)
        n_triggers_ch    = len(subt_wfs_filt)
    except ValueError:
        continue

    int_subt_wfs_filt[channel] = np.append(int_subt_wfs_filt[channel], int_subt_wf_filt)
    num_triggers_ch  [channel] = np.append(num_triggers_ch  [channel], n_triggers_ch)

int_subt_wfs_filt = np.array(int_subt_wfs_filt, dtype=object)
num_triggers_ch   = np.array(num_triggers_ch,   dtype=object)

np.savez(outfile, 
         int_subt_wfs_filt=int_subt_wfs_filt,
         num_triggers_ch  =num_triggers_ch)