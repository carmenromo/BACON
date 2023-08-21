
import sys
import uproot

import numpy as np

import peak_functions as pf

arguments = pf.parse_args(sys.argv)
in_path   = arguments.in_path
file_name = arguments.file_name
out_path  = arguments.out_path

total_SiPMs = 9
dead_SiPMs  = [3]

peak_height_all_channels  = [[] for i in range(total_SiPMs - len(dead_SiPMs))]
peak_area_all_channels    = [[] for i in range(total_SiPMs - len(dead_SiPMs))]
peak_area_zs_all_channels = [[] for i in range(total_SiPMs - len(dead_SiPMs))]

if file_name.endswith(".root"):
    file_name = file_name[:-5]
filename = f"{in_path}/{file_name}.root"
infile   = uproot.open(filename)
RawTree  = infile['RawTree']

outfile = f"{out_path}/BACoN_sig_processing_peak_height_and_area_{file_name}"

sipm_thr = 50 #ADCs
peak_sep = 10

for i,channel in enumerate(range(total_SiPMs)):
    print(channel)
    if channel in dead_SiPMs: continue
    try:
        zs_wfs, subt_wfs_filt, all_peaks = pf.get_peaks_using_peakutils(RawTree, channel, sipm_thr=sipm_thr)
        heights                          = pf.height_of_peaks(subt_wfs_filt, all_peaks)
        areas                            = pf.area_of_peaks(  subt_wfs_filt, all_peaks)
        areas_zs                         = pf.area_zs(zs_wfs, subt_wfs_filt, peak_sep=peak_sep)
        peak_height_all_channels [i].append(heights)
        peak_area_all_channels   [i].append(areas)
        peak_area_zs_all_channels[i].append(areas_zs)
    except ValueError:
        continue

peak_height_all_channels  = np.array(peak_height_all_channels,  dtype=object)
peak_area_all_channels    = np.array(peak_area_all_channels,    dtype=object)
peak_area_zs_all_channels = np.array(peak_area_zs_all_channels, dtype=object)

np.savez(outfile, 
         peak_height_all_channels=peak_height_all_channels,
         peak_area_all_channels=peak_area_all_channels, 
         peak_area_zs_all_channels=peak_area_zs_all_channels)
