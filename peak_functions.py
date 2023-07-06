
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

def split_in_peaks_indices(wfs, stride):
    indices_above_zero = np.where(wfs > 0)[0]
    where = np.where(np.diff(indices_above_zero) > stride)[0]
    return np.split(indices_above_zero, where + 1)

def split_in_peaks_vals(wfs, stride, len_peak):
    indices_above_zero = np.where(wfs > 0)[0]
    where = np.where(np.diff(indices_above_zero) > stride)[0]
    ind_peaks_splitted = np.split(indices_above_zero, where + 1)
    if len(ind_peaks_splitted)>len_peak:
    #print(len(ind_peaks_splitted))
        return [wfs[peak] for peak in ind_peaks_splitted]
    else:
        return []

def get_height_charge_for_a_channel(RawTree, channel, sipm_thr=50):
    all_raw_wfs            = np.array(RawTree[f'chan{channel}/rdigi'].array())
    subt_raw_wfs           = list(map(subtract_baseline, all_raw_wfs))
    zs_raw_wfs             = noise_suppression(subt_raw_wfs, sipm_thr)
    filter_empty_wfs       = zs_raw_wfs[np.any(zs_raw_wfs != 0, axis=1)]
    peaks_splitted_vals    = list(map(split_in_peaks_vals,    filter_empty_wfs, np.tile(peak_sep, len(subt_raw_wfs))))
    #peaks_splitted_indices = list(map(split_in_peaks_indices, filter_empty_wfs, np.tile(peak_sep, len(subt_raw_wfs))))
    peaks_splitted_vals_flattened = flatten_list(peaks_splitted_vals)
    max_peak_channel       = list(map(max, peaks_splitted_vals_flattened))
    return max_peak_channel

def get_peaks_peakutils(subt_wf):
    return peakutils.indexes(subt_wf, thres=0.35, min_dist=100)
    
def peak_height(wf_bs_subt, peaks):
    return np.array([wf_bs_subt[peak] for peak in peaks])

def peak_height_using_peakutils(RawTree, channel, sipm_thr=50):
    all_raw_wfs       = np.array(RawTree[f'chan{channel}/rdigi'].array())
    subt_raw_wfs      = list(map(subtract_baseline, all_raw_wfs))
    zs_raw_wfs        = noise_suppression(subt_raw_wfs, threshold=sipm_thr)
    filter_empty_wfs  = zs_raw_wfs[np.any(zs_raw_wfs != 0, axis=1)]
    subt_raw_wfs_filt = np.array(subt_raw_wfs)[np.any(zs_raw_wfs != 0, axis=1)]
    all_peaks         = list(map(get_peaks_peakutils, filter_empty_wfs))
    all_heights       = np.concatenate(list(map(peak_height, subt_raw_wfs_filt, all_peaks)))
    return all_heights
