
import argparse
import numpy as np

from scipy     import stats   as st
from functools import partial

import peakutils

import blr_functions as blr

def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('in_path'  , help = "input files path"   )
    parser.add_argument('file_name', help = "name of input files")
    parser.add_argument('out_path' , help = "output files path"  )
    return parser.parse_args()

def wfs_from_rawtree(RawTree, channel):
    return np.array(RawTree[f'chan{channel}/rdigi'].array())

def compute_baseline(wf, mode=True, wf_range_bsl=(0, None)):
    """
    Compute the baseline for a waveform in the input
    with a specific algorithm (mode or mean) and a 
    specific range.
    """
    if mode:
        baseline = st.mode(wf[wf_range_bsl[0]:wf_range_bsl[1]], keepdims=False).mode.astype(np.float32)
    else:
        baseline = np.mean(wf[wf_range_bsl[0]:wf_range_bsl[1]])
    return baseline

def subtract_baseline(wfs, mode=True, wf_range_bsl=(0, None)):
    """
    Subtract the baseline to one or multiple waveforms in the input
    with a specific algorithm (mode or mean).
    """
    if len(wfs.shape)==1: ## Only one waveform
        baseline = compute_baseline(wfs, mode=mode, wf_range_bsl=wf_range_bsl)
    elif len(wfs.shape)==2: ## Multiple wfs
        baseline = np.mean([compute_baseline(wf, mode=mode, wf_range_bsl=wf_range_bsl) for wf in wfs])
    return wfs - baseline

def suppress_wf(waveform, threshold):
    """Put zeros where the waveform is below some threshold.
    """
    wf = np.copy(waveform)
    below_thr = wf <= threshold
    wf[below_thr] = 0
    return wf

def noise_suppression(waveforms, threshold=0):
    """Put zeros where the waveform is below some threshold.
    """
    thresholds     = np.tile(threshold, len(waveforms))
    suppressed_wfs = list(map(suppress_wf, waveforms, thresholds))
    return np.array(suppressed_wfs)

def flatten_list(arr):
    flattened_list = []
    for item in arr:
        if isinstance(item, list):
            flattened_list.extend(flatten_list(item))
        else:
            flattened_list.append(item)
    return flattened_list

def split_in_peaks_vals(wfs, stride, len_peak=5):
    indices_above_zero = np.where(wfs > 0)[0]
    where = np.where(np.diff(indices_above_zero) > stride)[0]
    ind_peaks_splitted = np.split(indices_above_zero, where + 1)
    if len(ind_peaks_splitted)>len_peak:
        return [wfs[peak] for peak in ind_peaks_splitted]
    else:
        return []
    
def peak_height(waveform, peaks):
    return np.array([waveform[peak] for peak in peaks])

def peak_height_deconv(zs_waveform, peaks, heights):
    for idx_peak in range(len(peaks)-1):
        zswf_sel = zs_waveform[peaks[idx_peak]:peaks[idx_peak+1]]
        if np.any(zswf_sel == 0):
            continue
        else:
            heights[idx_peak+1] -= np.min(zswf_sel)
    return heights

def integrate_peaks(waveform, peaks):

    peaks_indxs = []
    for p_wk in peaks:
        peak_indxs     = []
        next_val_left  = p_wk - 1
        next_val_right = p_wk + 1
        peak_indxs.append(p_wk)
        
        while next_val_left >= 0 and waveform[next_val_left] > 0:
            peak_indxs.append(next_val_left)
            next_val_left -= 1
        
        while next_val_right < len(waveform) and waveform[next_val_right] > 0:
            peak_indxs.append(next_val_right)
            next_val_right += 1
        
        peaks_indxs.append(np.array(peak_indxs))
    
    return np.array([np.sum(waveform[idx]) for idx in peaks_indxs])

def find_wfs_above_thr(wfs, thr):
    indices_above_thr = [idx for idx, wf in enumerate(wfs) if len(wf[wf>thr]) > 0]
    return np.array(indices_above_thr)

def get_saturating_evts_using_pmt_signal(RawTree, num_wfs=None, pmt_channel=12, pmt_thr=1000):
    ## Get saturating events using PMT info
    pmt_raw_wfs     = wfs_from_rawtree(RawTree, pmt_channel)[:num_wfs]
    pmt_cwfs        = np.array([blr.pmt_deconvolver(wf, wf_range_bsl=(0, 700)) for wf in pmt_raw_wfs])
    saturating_evts = find_wfs_above_thr(pmt_cwfs, thr=pmt_thr)
    return saturating_evts

def remove_waveforms_by_indices(waveforms, indices_to_remove):
    if len(indices_to_remove)==0:
        return waveforms
    filtered_waveforms = np.delete(waveforms, indices_to_remove, axis=0)
    return filtered_waveforms

def get_peaks_peakutils(waveform, thres=0.35, min_dist=100, thres_abs=False):
    return peakutils.indexes(waveform, thres=thres, min_dist=min_dist, thres_abs=thres_abs)

def get_peaks_using_peakutils(RawTree, channel, num_wfs=None, sipm_thr=50, pmt_thr=1000, peak_range=(650,850)):
    all_raw_wfs       = wfs_from_rawtree(RawTree, channel)[:num_wfs]
    
    ## Subtract baseline
    subt_raw_wfs      = list(map(subtract_baseline, all_raw_wfs))
    
    ## Get and remove saturated events from PMTs
    saturated_evts    = get_saturating_evts_using_pmt_signal(RawTree, num_wfs=num_wfs, pmt_thr=pmt_thr)
    filt_wfs          = remove_waveforms_by_indices(subt_raw_wfs, saturated_evts)
    
    ## Zero suppression
    zs_raw_wfs        = noise_suppression(filt_wfs, threshold=sipm_thr)
    
    ## Remove events with no signal in the ROI
    empty_evts        = np.array([idx for idx, zwf in enumerate(zs_raw_wfs) if np.sum(zwf[peak_range[0]:peak_range[1]])==0])
    filter_empty_zwfs = remove_waveforms_by_indices(zs_raw_wfs, empty_evts)
    subt_raw_wfs_filt = remove_waveforms_by_indices(filt_wfs,   empty_evts)
    
    ## Get the peaks found in the ROI
    all_peaks         = list(map(get_peaks_peakutils, filter_empty_zwfs))
    return filter_empty_zwfs, subt_raw_wfs_filt, all_peaks

def get_peaks_using_peakutils_no_PMT(RawTree, channel, num_wfs=None, sipm_thr=50, peak_range=(650,850), wf_range_bsl=(0, None)):
    all_raw_wfs = wfs_from_rawtree(RawTree, channel)[:num_wfs]
    #all_raw_wfs = np.array([wf for wf in all_raw_wfs if np.std(wf) > 15]) #Way of removing the baseline wfs

    if channel in [9, 10, 11]: #trigger SiPMs
        all_raw_wfs = np.array([blr.pmt_deconvolver(wf, wf_range_bsl=wf_range_bsl) for wf in all_raw_wfs])

    ## Subtract baseline
    partial_subtract_baseline = partial(subtract_baseline, wf_range_bsl=wf_range_bsl)
    subt_raw_wfs = list(map(partial_subtract_baseline, all_raw_wfs))
    #subt_raw_wfs = list(map(subtract_baseline, all_raw_wfs))

    ## Zero suppression
    zs_raw_wfs   = noise_suppression(subt_raw_wfs, threshold=sipm_thr)

    ## Remove events with no signal in the ROI
    empty_evts        = np.array([idx for idx, zwf in enumerate(zs_raw_wfs) if np.sum(zwf[peak_range[0]:peak_range[1]])==0])
    filter_empty_zwfs = remove_waveforms_by_indices(zs_raw_wfs,   empty_evts)
    subt_raw_wfs_filt = remove_waveforms_by_indices(subt_raw_wfs, empty_evts)

    ## Get the peaks found in the ROI
    all_peaks         = list(map(get_peaks_peakutils, filter_empty_zwfs))
    return filter_empty_zwfs, subt_raw_wfs_filt, all_peaks

def height_of_peaks(waveforms, peaks):
    all_heights = np.concatenate(list(map(peak_height, waveforms, peaks)))
    return all_heights

def area_of_peaks(waveforms, peaks):
    all_areas = np.concatenate(list(map(integrate_peaks, waveforms, peaks)))
    return all_areas

def area_zs(zs_waveforms, waveforms, peak_sep=10):
    peaks_splitted_vals = list(map(split_in_peaks_vals, zs_waveforms, np.tile(peak_sep, len(waveforms))))
    all_areas           = np.array(list(map(sum, flatten_list(peaks_splitted_vals))))
    return all_areas
