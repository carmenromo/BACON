
import sys
import uproot

import numpy as np

import peak_functions as pf
import blr_functions  as blr

from scipy.signal import savgol_filter


arguments = pf.parse_args(sys.argv)
in_path   = arguments.in_path
file_name = arguments.file_name
out_path  = arguments.out_path

if file_name.endswith(".root"):
    file_name = file_name[:-5]
filename = f"{in_path}/{file_name}.root"
infile   = uproot.open(filename)
RawTree  = infile['RawTree']

outfile = f"{out_path}/BACoN_hits_and_times_{file_name}"

## Parameters
max_smpl_bsl        = 650
std_bsl_thr         = 15
sg_filter_window    = 50
sg_filter_polyorder = 3
thr_ADC             = 20 #ths for the noise suppression and peak finder after SG filter

normal_chs  = range(9)
trigger_chs = [9, 10, 11]


#### CHANNELS FROM 0 TO 8:
## In the normal SiPMs, a filter is performed to remove the baseline evts
filt_wfs_dict = {ch: np.array([(evt, wf)
                               for evt, wf in enumerate(pf.wfs_from_rawtree(RawTree, ch)) if np.std(wf) > std_bsl_thr], dtype=object)
                 for ch in normal_chs}

#filt_evts      = np.unique(np.concatenate(np.array([filt_wfs_dict[ch].T[0] for ch in normal_chs])))
filt_evts_dict = {ch: filt_wfs_dict[ch].T[0]
                  if len(filt_wfs_dict[ch])!=0 else []
                  for ch in normal_chs}

## Baseline subtraction
subt_wfs_dict = {ch: np.array([pf.subtract_baseline(fwf,
                                          mode=True,
                                          wf_range_bsl=(0, max_smpl_bsl))
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
idx_peaks_ch_dict = {ch: np.array(list(map(pf.get_peaks_peakutils, zs_sg_filt_swfs_dict[ch])), dtype=object)
                     for ch in normal_chs}

height_peaks_ch_dict = {ch: np.array([pf.peak_height(wf, idx_peaks_ch_dict[ch][i])
                                      for i,wf in enumerate(zs_sg_filt_swfs_dict[ch])], dtype=object)
                        for ch in normal_chs}


#### TRIGGER SIPMS
## In the deconvolution the baseline is already subtracted from the waveform!!!
trigg_cwfs_dict = {ch: np.array([blr.pmt_deconvolver(wf, wf_range_bsl=(0, max_smpl_bsl))
                                 for wf in pf.wfs_from_rawtree(RawTree, ch)])
                   for ch in trigger_chs}

sg_filt_trigg_dict = {ch: savgol_filter(trigg_cwfs_dict[ch],
                                        window_length=sg_filter_window,
                                        polyorder=sg_filter_polyorder)
                      for ch in trigger_chs}

zs_sg_filt_trigg_dict = {ch: pf.noise_suppression(sg_filt_trigg_dict[ch],
                                                  threshold=thr_ADC)
                         for ch in trigger_chs}

idx_peaks_ch_trigg_dict = {ch: np.array(list(map(pf.get_peaks_peakutils, zs_sg_filt_trigg_dict[ch])), dtype=object)
                           for ch in trigger_chs}

height_peaks_ch_trigg_dict = {ch: np.array([pf.peak_height(wf, idx_peaks_ch_trigg_dict[ch][i])
                                            for i,wf in enumerate(zs_sg_filt_trigg_dict[ch])], dtype=object)
                              for ch in trigger_chs}

height_peaks_deconv_ch_trigg_dict = {ch: np.array([pf.peak_height_deconv(wf,
                                                                        idx_peaks_ch_trigg_dict   [ch][i],
                                                                        height_peaks_ch_trigg_dict[ch][i].copy())
                                     for i, wf in enumerate(zs_sg_filt_trigg_dict[ch])], dtype=object)
                                     for ch in trigger_chs}


np.savez(outfile,
         filt_evts_dict=filt_evts_dict,
         idx_peaks_ch_dict=idx_peaks_ch_dict,
         height_peaks_ch_dict=height_peaks_ch_dict,
         idx_peaks_ch_trigg_dict=idx_peaks_ch_trigg_dict,
         height_peaks_ch_trigg_dict=height_peaks_ch_trigg_dict,
         height_peaks_deconv_ch_trigg_dict=height_peaks_deconv_ch_trigg_dict)