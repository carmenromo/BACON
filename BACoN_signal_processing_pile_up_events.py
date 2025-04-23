import uproot

import numpy as np

import peak_functions as pf
import blr_functions  as blr

from functools import partial

from scipy.signal import savgol_filter


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

python3 BACoN_signal_processing_hits_and_times.py /wherever/your/files/are my_file.root /output/data/"""

## Parameters
max_smpl_bsl        = 650
#std_bsl_thr         = 15
sg_filter_window    =  30
sg_filter_polyorder =   3
thr_ADC             =  80 #ths for the noise suppression and peak finder after SG filter
thr_ADC_trigg       = 200 #ths for the noise suppression and peak finder after SG filter for the trigger SiPMs
min_dist            =  50 #min distance in time samples (ns/2) between peaks for peakutils

## Thr values valid from 9/10/2024 since the bas voltage was changed
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

outfile = f"{out_path}/BACoN_run3_pile_up_evts_thr{thr_ADC}_mean_w{sg_filter_window}_dist{min_dist}_{file_name}"

normal_chs  = [8]#range(9)
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

unique_evt_ids = np.sort(np.unique(np.concatenate([filt_evts_dict[ch] for ch in normal_chs])))

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
idx_peaks_ch_dict = {ch: np.array(list(map(partial_get_peaks_peakutils, zs_sg_filt_swfs_dict[ch])), dtype=object)
                     for ch in normal_chs}

idx_peaks_thr_ch_dict = {ch: np.array([pf.get_values_thr_from_zswf(wf, idx_peaks_ch_dict[ch][i])
                                      for i,wf in enumerate(zs_sg_filt_swfs_dict[ch])], dtype=object)
                         for ch in normal_chs}

height_peaks_sg_ch_dict = {ch: np.array([pf.peak_height(wf, idx_peaks_ch_dict[ch][i])
                                      for i,wf in enumerate(zs_sg_filt_swfs_dict[ch])], dtype=object)
                        for ch in normal_chs}

idx_peaks_sg_deconv_sec_ch_dict = {ch: np.array([pf.peak_height_deconv_save_secondary_peaks(wf,
                                                                      idx_peaks_ch_dict      [ch][i],
                                                                      height_peaks_sg_ch_dict[ch][i].copy())[0]
                               for i, wf in enumerate(zs_sg_filt_swfs_dict[ch])], dtype=object)
                               for ch in normal_chs}

height_peaks_sg_deconv_sec_abs_ch_dict = {ch: np.array([pf.peak_height_deconv_save_secondary_peaks(wf,
                                                                      idx_peaks_ch_dict      [ch][i],
                                                                      height_peaks_sg_ch_dict[ch][i].copy())[1]
                               for i, wf in enumerate(zs_sg_filt_swfs_dict[ch])], dtype=object)
                               for ch in normal_chs}

height_peaks_sg_deconv_sec_rel_ch_dict = {ch: np.array([pf.peak_height_deconv_save_secondary_peaks(wf,
                                                                      idx_peaks_ch_dict      [ch][i],
                                                                      height_peaks_sg_ch_dict[ch][i].copy())[2]
                               for i, wf in enumerate(zs_sg_filt_swfs_dict[ch])], dtype=object)
                               for ch in normal_chs}



#### TRIGGER SIPMS
## In the deconvolution the baseline is already subtracted from the waveform!!!
trigg_cwfs_dict = {ch: np.array([blr.pmt_deconvolver(wf, wf_range_bsl=(0, max_smpl_bsl), std_lim=std_thr_dict[ch])
                                 for wf in pf.wfs_from_rawtree(RawTree, ch)])
                   for ch in trigger_chs}

sg_filt_trigg_dict = {ch: savgol_filter(trigg_cwfs_dict[ch],
                                        window_length=sg_filter_window,
                                        polyorder=sg_filter_polyorder)
                      for ch in trigger_chs}

trigger_samp_dict = {ch: np.array([pf.get_evt_trigger_t(wf, thr_ADC_trigg=thr_ADC_trigg, rng=(1400/2, 1500/2))
                                   for wf in sg_filt_trigg_dict[ch]])
                     for ch in trigger_chs}

zs_sg_filt_trigg_dict = {ch: pf.noise_suppression(sg_filt_trigg_dict[ch],
                                                  threshold=thr_ADC_trigg)
                         for ch in trigger_chs}

idx_peaks_ch_trigg_dict = {ch: np.array(list(map(partial_get_peaks_peakutils, zs_sg_filt_trigg_dict[ch])), dtype=object)
                           for ch in trigger_chs}

height_peaks_sg_ch_trigg_dict = {ch: np.array([pf.peak_height(wf, idx_peaks_ch_trigg_dict[ch][i])
                                               for i,wf in enumerate(zs_sg_filt_trigg_dict[ch])], dtype=object)
                                 for ch in trigger_chs}

idx_peaks_sg_deconv_sec_ch_trigg_dict = {ch: np.array([pf.peak_height_deconv_save_secondary_peaks(wf,
                                                                      idx_peaks_ch_trigg_dict      [ch][i],
                                                                      height_peaks_sg_ch_trigg_dict[ch][i].copy())[0]
                               for i, wf in enumerate(zs_sg_filt_trigg_dict[ch])], dtype=object)
                               for ch in trigger_chs}

height_peaks_sg_deconv_sec_abs_ch_trigg_dict = {ch: np.array([pf.peak_height_deconv_save_secondary_peaks(wf,
                                                                      idx_peaks_ch_trigg_dict      [ch][i],
                                                                      height_peaks_sg_ch_trigg_dict[ch][i].copy())[1]
                               for i, wf in enumerate(zs_sg_filt_trigg_dict[ch])], dtype=object)
                               for ch in trigger_chs}

height_peaks_sg_deconv_sec_rel_ch_trigg_dict = {ch: np.array([pf.peak_height_deconv_save_secondary_peaks(wf,
                                                                      idx_peaks_ch_trigg_dict      [ch][i],
                                                                      height_peaks_sg_ch_trigg_dict[ch][i].copy())[2]
                               for i, wf in enumerate(zs_sg_filt_trigg_dict[ch])], dtype=object)
                               for ch in trigger_chs}


np.savez(outfile,
         filt_evts_dict=filt_evts_dict,
         trigger_samp_dict=trigger_samp_dict,
         idx_peaks_sg_deconv_sec_ch_dict=idx_peaks_sg_deconv_sec_ch_dict,
         height_peaks_sg_deconv_sec_abs_ch_dict=height_peaks_sg_deconv_sec_abs_ch_dict,
         height_peaks_sg_deconv_sec_rel_ch_dict=height_peaks_sg_deconv_sec_rel_ch_dict,
         idx_peaks_sg_deconv_sec_ch_trigg_dict=idx_peaks_sg_deconv_sec_ch_trigg_dict,
         height_peaks_sg_deconv_sec_abs_ch_trigg_dict=height_peaks_sg_deconv_sec_abs_ch_trigg_dict,
         height_peaks_sg_deconv_sec_rel_ch_trigg_dict=height_peaks_sg_deconv_sec_rel_ch_trigg_dict)