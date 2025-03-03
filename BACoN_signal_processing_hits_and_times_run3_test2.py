
import sys
import time
import uproot

import numpy as np

import peak_functions as pf
import blr_functions  as blr

from functools import partial

from scipy.signal import savgol_filter

start_time = time.time()


arguments = pf.parse_args(sys.argv)
in_path   = arguments.in_path
file_name = arguments.file_name
out_path  = arguments.out_path

if file_name.endswith(".root"):
    file_name = file_name[:-5]
filename = f"{in_path}/{file_name}.root"
infile   = uproot.open(filename)
RawTree  = infile['RawTree']

"""to run this script:

python3 BACoN_signal_processing_hits_and_times.py /wherever/your/files/are my_file.root /output/data/"""

## Parameters
max_smpl_bsl        = 650
#std_bsl_thr        = 15
sg_filter_window    =  30
sg_filter_polyorder =   3
thr_ADC             =  80 #ths for the noise suppression and peak finder after SG filter
thr_ADC_trigg       = 200 #ths for the noise suppression and peak finder after SG filter for the trigger SiPMs
min_dist            =  25 #min distance in t samples between peaks for peakutils

## Thr values valid from 9/10/2024 since the bias voltage was changed
std_thr_dict = {0: 13,
                1: 13,
                2: 13,
                3: 13,
                4: 14,
                5: 13,
                6: 12,
                7: 13,
                8: 13,
                9: 30,
                10: 40,
                11: 40}

outfile = f"{out_path}/BACoN_run3_hits_and_times_thr{thr_ADC}_mean_w{sg_filter_window}_dist{min_dist}_abs_vals_ch9_{file_name}"

trigger_chs  = [9]

trigg_cwfs_dict = {ch: np.array([blr.pmt_deconvolver(wf, wf_range_bsl=(0, max_smpl_bsl), std_lim=3*std_thr_dict[ch])
                                 for wf in pf.wfs_from_rawtree(RawTree, ch)])
                   for ch in trigger_chs}

sg_filt_trigg_dict = {ch: savgol_filter(trigg_cwfs_dict[ch],
                                        window_length=sg_filter_window,
                                        polyorder=sg_filter_polyorder)
                      for ch in trigger_chs}

zs_sg_filt_trigg_dict = {ch: pf.noise_suppression(sg_filt_trigg_dict[ch],
                                                  threshold=thr_ADC_trigg)
                         for ch in trigger_chs}

## Get peaks above thr_ADC
partial_get_peaks_peakutils = partial(pf.get_peaks_peakutils, thres=thr_ADC, min_dist=min_dist, thres_abs=True)

idx_peaks_max_trigg = {ch: np.array(list(map(partial_get_peaks_peakutils, zs_sg_filt_trigg_dict[ch])), dtype=object)
                           for ch in trigger_chs}

idx_peaks_thr_trigg = {ch: np.array([pf.get_values_thr_from_zswf(wf, idx_peaks_max_trigg[ch][i])
                                            for i,wf in enumerate(zs_sg_filt_trigg_dict[ch])], dtype=object)
                               for ch in trigger_chs}

height_peaks_sg_trigg = {ch: np.array([pf.peak_height(wf, idx_peaks_max_trigg[ch][i])
                                               for i,wf in enumerate(zs_sg_filt_trigg_dict[ch])], dtype=object)
                                 for ch in trigger_chs}


np.savez(outfile,
         idx_peaks_max_trigg=idx_peaks_max_trigg,
         idx_peaks_thr_trigg=idx_peaks_thr_trigg,
         height_peaks_sg_trigg=height_peaks_sg_trigg)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} s, {elapsed_time/60} mins")