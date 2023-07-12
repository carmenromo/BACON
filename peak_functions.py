
import argparse
import uproot
import numpy as np

from scipy import stats as st
import scipy

import peakutils

def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('in_path'  ,  help = "input files path"   )
    parser.add_argument('file_name',  help = "name of input files")
    parser.add_argument('out_path' ,  help = "output files path"  )
    return parser.parse_args()


def subtract_baseline(wfs, mode=True):
    """
    Subtract the baseline to all waveforms in the input
    with a specific algorithm (mode or mean).
    """
    if mode:
        baseline = st.mode(wfs, keepdims=False).mode.astype(np.float32)
    else:
        baseline = np.mean(wfs)
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

def split_in_peaks_indices(wfs, stride):
    indices_above_zero = np.where(wfs > 0)[0]
    where = np.where(np.diff(indices_above_zero) > stride)[0]
    return np.split(indices_above_zero, where + 1)

def split_in_peaks_vals(wfs, stride, len_peak=5):
    indices_above_zero = np.where(wfs > 0)[0]
    where = np.where(np.diff(indices_above_zero) > stride)[0]
    ind_peaks_splitted = np.split(indices_above_zero, where + 1)
    if len(ind_peaks_splitted)>len_peak:
        return [wfs[peak] for peak in ind_peaks_splitted]
    else:
        return []

def get_peaks_peakutils(waveform):
    return peakutils.indexes(waveform, thres=0.35, min_dist=100)
    
def peak_height(waveform, peaks):
    return np.array([waveform[peak] for peak in peaks])


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


def get_peaks_using_peakutils(RawTree, channel, sipm_thr=50):
    all_raw_wfs       = np.array(RawTree[f'chan{channel}/rdigi'].array())
    subt_raw_wfs      = list(map(subtract_baseline, all_raw_wfs))
    zs_raw_wfs        = noise_suppression(subt_raw_wfs, threshold=sipm_thr)
    filter_empty_wfs  = zs_raw_wfs[np.any(zs_raw_wfs != 0, axis=1)]
    subt_raw_wfs_filt = np.array(subt_raw_wfs)[np.any(zs_raw_wfs != 0, axis=1)]
    all_peaks         = list(map(get_peaks_peakutils, filter_empty_wfs))
    return filter_empty_wfs, subt_raw_wfs_filt, all_peaks

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
