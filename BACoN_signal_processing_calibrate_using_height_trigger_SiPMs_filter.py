
import uproot
import numpy as np

import peak_functions as pf

arguments = pf.parse_args()
in_path   = arguments.in_path
file_name = arguments.file_name
out_path  = arguments.out_path

trigger_SiPMs = [9, 10, 11] 

peak_height_all_channels  = [[] for i in range(len(trigger_SiPMs))]

if file_name.endswith(".root"):
    file_name = file_name[:-5]
filename = f"{in_path}/{file_name}.root"
infile   = uproot.open(filename)
RawTree  = infile['RawTree']

peak_range   = (650,850)
max_smpl_bsl = 650
sg_filter    = True
sg_window    = 30
sg_polyorder = 3
sipm_thr     = 50 #ADCs
peak_sep     = 10

outfile = f"{out_path}/BACoN_cal_height_trigger_SiPMs_sg_filter_{sg_filter}_w{sg_window}_dist{peak_sep}_{file_name}"

for i, channel in enumerate(trigger_SiPMs):
    print(channel)
    heights = np.array([])
    try:
        _, subt_wfs_filt, all_peaks = pf.get_peaks_using_peakutils_no_PMT(RawTree, channel, sipm_thr=sipm_thr, peak_range=peak_range, wf_range_bsl=(0, max_smpl_bsl),
                                                                          sg_filter=sg_filter, window_length=sg_window, polyorder=sg_polyorder)
        heights                     = pf.height_of_peaks(subt_wfs_filt, all_peaks)
    except ValueError:
        continue
    peak_height_all_channels[i].append(heights)

peak_height_all_channels  = np.array(peak_height_all_channels,  dtype=object)

np.savez(outfile, 
         peak_height_all_channels=peak_height_all_channels)
