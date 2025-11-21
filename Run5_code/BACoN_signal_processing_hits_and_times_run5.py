import os
import sys
import time
import uproot
import numpy as np

from functools    import partial
from scipy.signal import savgol_filter

# Add the parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import peak_functions as pf
import blr_functions  as blr

start_time = time.time()


arguments = pf.parse_args()
in_path   = arguments.in_path
file_name = arguments.file_name
out_path  = arguments.out_path

if file_name.endswith(".root"):
    file_name = file_name[:-5]
filename = f"{in_path}/{file_name}.root"
infile   = uproot.open(filename)
RawTree  = infile['RawTree']

"""to run this script:

python3 BACoN_signal_processing_hits_and_times_run5.py /wherever/your/files/are my_file.root /output/data/"""

## Parameters
max_smpl_bsl        = 650
#std_bsl_thr        =  15
sg_filter_window    =  30
sg_filter_polyorder =   3
thr_ADC             =  50 #ths for the noise suppression and peak finder after SG filter
thr_ADC_trigg       = 250 #ths for the noise suppression and peak finder after SG filter for the trigger SiPMs
min_dist            =  25 #min distance in t samples between peaks for peakutils

std_thr_dict = {0: 12.75,
                1: 12.25,
                2: 11.85,
                3: 11.75,
                4: 12.75,
                5: 12.5,
                6: 11.5,
                7: 11.75,
                8: 12.75,
                9: 25,
                10: 50,
                11: 45,
                12: 4}

outfile = f"{out_path}/BACoN_run5_hits_and_times_thr{thr_ADC}_mean_w{sg_filter_window}_dist{min_dist}_{file_name}"

normal_chs  = range(9)
trigger_chs = [9, 10, 11]

#### CHANNELS FROM 0 TO 8:
## In the normal SiPMs, a filter is performed to remove the baseline evts
filt_wfs_dict = {ch: np.array([(evt, wf)
                               for evt, wf in enumerate(pf.wfs_from_rawtree(RawTree, ch)) if np.std(wf) > std_thr_dict[ch]], dtype=object)
                 for ch in normal_chs}

#filt_evts = np.unique(np.concatenate(np.array([filt_wfs_dict[ch].T[0] for ch in normal_chs])))
filt_evts_dict = {ch: filt_wfs_dict[ch].T[0]
                  if len(filt_wfs_dict[ch])!=0 else []
                  for ch in normal_chs}

## Baseline subtraction
subt_wfs_dict = {ch: np.array([pf.subtract_baseline_std_lim(fwf,
                                                            mode=False,
                                                            wf_range_bsl=(0, max_smpl_bsl),
                                                            std_lim=3*std_thr_dict[ch])
                                                            for _, fwf in filt_wfs_dict[ch]])
                 if len(filt_wfs_dict[ch])!=0 else []
                 for ch in normal_chs}

## Apply the Savitzky-Golay filter to smooth the wf
sg_filt_swfs_dict = {ch: savgol_filter(subt_wfs_dict[ch],
                                       window_length=sg_filter_window,
                                       polyorder=sg_filter_polyorder)
                     if len(subt_wfs_dict[ch]) != 0 else []
                     for ch in normal_chs}

## Noise suppression
zs_sg_filt_swfs_dict = {ch: pf.noise_suppression(sg_filt_swfs_dict[ch],
                                                 threshold=thr_ADC)
                        if len(sg_filt_swfs_dict[ch]) != 0 else []
                        for ch in normal_chs}

## Get peaks above thr_ADC
partial_get_peaks_peakutils = partial(pf.get_peaks_peakutils, thres=thr_ADC, min_dist=min_dist, thres_abs=True)

## Indices of the max of the peak
idx_peaks_max = {ch: np.array(list(map(partial_get_peaks_peakutils, zs_sg_filt_swfs_dict[ch])), dtype=object)
                 for ch in normal_chs}

## Height of the max of the peak after SG filter
height_peaks_sg = {ch: np.array([pf.peak_height(wf, idx_peaks_max[ch][i])
                                 for i,wf in enumerate(zs_sg_filt_swfs_dict[ch])], dtype=object)
                   for ch in normal_chs}

integral_peaks_sg = {ch: np.array([pf.integrate_and_get_len_peaks_fast(wf, idx_peaks_max[ch][i])[0]
                                   for i,wf in enumerate(zs_sg_filt_swfs_dict[ch])], dtype=object)
                     for ch in normal_chs}

len_peaks_sg = {ch: np.array([pf.integrate_and_get_len_peaks_fast(wf, idx_peaks_max[ch][i])[1]
                              for i, wf in enumerate(zs_sg_filt_swfs_dict[ch])], dtype=object)
                for ch in normal_chs}


#### TRIGGER SIPMS
## In the deconvolution the baseline is already subtracted from the waveform!!!
trigg_cwfs_dict = {ch: np.array([blr.deconvolver(wf, wf_range_bsl=(0, max_smpl_bsl), std_lim=3*std_thr_dict[ch])
                                 for wf in pf.wfs_from_rawtree(RawTree, ch)])
                   for ch in trigger_chs}

sg_filt_trigg_dict = {ch: savgol_filter(trigg_cwfs_dict[ch],
                                        window_length=sg_filter_window,
                                        polyorder=sg_filter_polyorder)
                      for ch in trigger_chs}

zs_sg_filt_trigg_dict = {ch: pf.noise_suppression(sg_filt_trigg_dict[ch],
                                                  threshold=thr_ADC_trigg)
                         for ch in trigger_chs}

idx_peaks_max_trigg = {ch: np.array(list(map(partial_get_peaks_peakutils, zs_sg_filt_trigg_dict[ch])), dtype=object)
                       for ch in trigger_chs}


height_peaks_sg_trigg = {ch: np.array([pf.peak_height(wf, idx_peaks_max_trigg[ch][i])
                                       for i,wf in enumerate(zs_sg_filt_trigg_dict[ch])], dtype=object)
                         for ch in trigger_chs}

integral_peaks_sg_trigg = {ch: np.array([pf.integrate_and_get_len_peaks_fast(wf, idx_peaks_max_trigg[ch][i])[0]
                                        for i,wf in enumerate(zs_sg_filt_trigg_dict[ch])], dtype=object)
                           for ch in trigger_chs}

len_peaks_sg_trigg = {ch: np.array([pf.integrate_and_get_len_peaks_fast(wf, idx_peaks_max_trigg[ch][i])[1]
                                   for i, wf in enumerate(zs_sg_filt_trigg_dict[ch])], dtype=object)
                      for ch in trigger_chs}

np.savez(outfile,
         filt_evts_dict=filt_evts_dict,
         idx_peaks_max=idx_peaks_max,
         height_peaks_sg=height_peaks_sg,
         integral_peaks_sg=integral_peaks_sg,
         len_peaks_sg=len_peaks_sg,
         idx_peaks_max_trigg=idx_peaks_max_trigg,
         height_peaks_sg_trigg=height_peaks_sg_trigg,
         integral_peaks_sg_trigg=integral_peaks_sg_trigg,
         len_peaks_sg_trigg=len_peaks_sg_trigg)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} s, {elapsed_time/60} mins")