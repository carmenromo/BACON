
import sys
import uproot

import numpy as np

import peak_functions as pf

arguments = pf.parse_args(sys.argv)
in_path   = arguments.in_path
file_name = arguments.file_name
out_path  = arguments.out_path

total_SiPMs = 9
dead_SiPMs  = []

peak_height_all_channels  = [[] for i in range(total_SiPMs)]

if file_name.endswith(".root"):
    file_name = file_name[:-5]
filename = f"{in_path}/{file_name}.root"
infile   = uproot.open(filename)
RawTree  = infile['RawTree']

peak_range = (650,850)

sipm_thr = 50 #ADCs
peak_sep = 10

outfile = f"{out_path}/BACoN_cal_height_dist{peak_sep}_{file_name}"

for channel in range(total_SiPMs):
    print(channel)
    heights = np.array([])
    if channel in dead_SiPMs:
        continue
    try:
        _, subt_wfs_filt, all_peaks = pf.get_peaks_using_peakutils_no_PMT(RawTree, channel, sipm_thr=sipm_thr, peak_range=peak_range)
        heights                     = pf.height_of_peaks(subt_wfs_filt, all_peaks)
    except ValueError:
        continue
    peak_height_all_channels [channel].append(heights)

peak_height_all_channels = np.array(peak_height_all_channels, dtype=object)

np.savez(outfile, 
         peak_height_all_channels=peak_height_all_channels)
