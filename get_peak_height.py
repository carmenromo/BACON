
import sys
import argparse
import uproot

import numpy as np

import peak_functions as pf

arguments = pf.parse_args(sys.argv)
in_path   = arguments.in_path
file_name = arguments.file_name
out_path  = arguments.out_path

peak_heigh_all_channels = []

if file_name.endswith(".root"):
    file_name = file_name[:-5]
filename = f"{in_path}/{file_name}.root"
infile   = uproot.open(filename)
RawTree  = infile['RawTree']

outfile = f"{out_path}/bacon_peak_height_{file_name}"

sipm_thr = 50 #ADCs

total_SiPMs = 9
dead_SiPMs  = [3]

for channel in range(total_SiPMs):
    print(channel)
    if channel in dead_SiPMs: continue
    heights_peakutils = pf.peak_height_using_peakutils(RawTree, channel, sipm_thr=sipm_thr)
    peak_heigh_all_channels.append(heights_peakutils)

peak_heigh_all_channels = np.array(peak_heigh_all_channels, dtype=object)

np.savez(outfile, peak_heigh_all_channels=peak_heigh_all_channels)