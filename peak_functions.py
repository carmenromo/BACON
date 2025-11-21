
import argparse
import numpy as np

from scipy        import stats   as st
from functools    import partial
from scipy.signal import savgol_filter

import peakutils

import blr_functions as blr

def parse_args():
    """
    Parse command-line arguments for the analysis scripts.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('in_path'  , help = "input files path"   )
    parser.add_argument('file_name', help = "name of input files")
    parser.add_argument('out_path' , help = "output files path"  )
    return parser.parse_args()


def wfs_from_rawtree(RawTree, channel):
    """
    Extract waveform samples for a specific channel from a RAW ROOT tree.

    Parameters
    ----------
    RawTree : uproot.behaviors.TBranch.TBranch or similar
        ROOT tree containing digitized waveform data.
    channel : int or str
        Channel number identifying the waveform branch.

    Returns
    -------
    numpy.ndarray
        Array of waveform samples for the selected channel.
    """
    return np.array(RawTree[f'chan{channel}/rdigi'].array())


def compute_baseline(wf, mode=True, wf_range_bsl=(0, None)):
    """
    Compute the baseline for a single waveform in the
    input with a specific algorithm (mode or mean) and
    a specific range.
    """
    if mode:
        try:
            baseline = st.mode(wf[wf_range_bsl[0]:wf_range_bsl[1]], keepdims=False).mode.astype(np.float32)
        except TypeError:
            baseline = st.mode(wf[wf_range_bsl[0]:wf_range_bsl[1]], axis=None).mode[0].astype(np.float32)
    else:
        baseline = np.mean(wf[wf_range_bsl[0]:wf_range_bsl[1]])
    return baseline

def compute_baseline_std_lim(wf, mode=True, wf_range_bsl=(0, None), std_lim=50):
    """
    Compute the baseline for a single waveform in the input
    with a specific algorithm (mode or mean), a specific range
    and a certain limit on the amplitude.
    """
    wf_sel_region = wf[wf_range_bsl[0]:wf_range_bsl[1]]
    try:
        mode_val = st.mode(wf, keepdims=False).mode
    except TypeError:
        mode_val = st.mode(wf, axis=None).mode[0]

    low_bnd = float(mode_val) - std_lim
    upp_bnd = float(mode_val) + std_lim
    filt_wf = wf_sel_region[(wf_sel_region >= low_bnd) & (wf_sel_region <= upp_bnd)]
    if len(filt_wf)==0:
        if mode:
            baseline = st.mode(wf, keepdims=False).mode.astype(np.float32)
        else:
            baseline = np.mean(wf)
    else:
        if mode:
            baseline = st.mode(filt_wf, keepdims=False).mode.astype(np.float32)
        else:
            baseline = np.mean(filt_wf)
    return baseline

def fast_mode(arr):
    """
    Compute the mode (most frequent value) of an array.

    Parameters
    ----------
    arr : array-like
        Input array of values (assumed to be integer-like or discrete).

    Returns
    -------
    value
        The most frequently occurring value in the array.
    """
    values, counts = np.unique(arr, return_counts=True)
    return values[np.argmax(counts)]

def compute_baseline_std_lim_fast(wf, mode=True, wf_range_bsl=(0, None), std_lim=50):
    """
    Compute the waveform baseline using a fast mode-based method.

    Parameters
    ----------
    wf : array-like
        Input waveform.
    mode : bool, default True
        If True, estimate baseline using the mode; otherwise use the mean.
    wf_range_bsl : tuple(int, int or None), default (0, None)
        Index range of the waveform used for baseline estimation.
    std_lim : float, default 50
        Amplitude window around the central value used to filter samples.

    Returns
    -------
    float
        Estimated baseline value.
    """
    wf_sel_region = wf[wf_range_bsl[0]:wf_range_bsl[1]]

    # Use whole waveform for central value estimation
    central_val = fast_mode(wf.astype(np.int32)) #Use mode to avoid including peaks in the average

    low_bnd = central_val - std_lim
    upp_bnd = central_val + std_lim
    filt_wf = wf_sel_region[(wf_sel_region >= low_bnd) & (wf_sel_region <= upp_bnd)]

    if len(filt_wf) == 0:
        baseline = central_val
    else:
        baseline = fast_mode(filt_wf.astype(np.int32)) if mode else np.mean(filt_wf)

    return float(baseline)

def subtract_baseline(wfs, mode=True, wf_range_bsl=(0, None), mean_bsl=True):
    """
    Subtract the baseline to one or multiple waveforms in the input
    with a specific algorithm (mode or mean).
    """
    if len(wfs.shape)==1: ## Only one waveform
        baseline = compute_baseline(wfs, mode=mode, wf_range_bsl=wf_range_bsl)
    elif len(wfs.shape)==2: ## Multiple wfs
        if mean_bsl:
            baseline = np.mean([compute_baseline(wf, mode=mode, wf_range_bsl=wf_range_bsl) for wf in wfs])
        else:
            baseline = np.array([compute_baseline(wf, mode=mode, wf_range_bsl=wf_range_bsl) for wf in wfs])

    return wfs - baseline

def subtract_baseline_std_lim(wf, mode=True, wf_range_bsl=(0, None), std_lim=50):
    """
    Subtract the estimated baseline from a waveform.

    Parameters
    ----------
    wf : array-like
        Input waveform.
    mode : bool, default True
        If True, estimate baseline using the mode; otherwise use the mean.
    wf_range_bsl : tuple(int, int or None), default (0, None)
        Index range of the waveform used for baseline estimation.
    std_lim : float, default 50
        Amplitude window around the central value used for baseline filtering.

    Returns
    -------
    array-like
        Waveform with baseline subtracted.
    """
    baseline = compute_baseline_std_lim_fast(wf, mode=mode, wf_range_bsl=wf_range_bsl, std_lim=std_lim)
    return wf - baseline

def suppress_wf(waveform, threshold):
    """
    Set values from a single waveform below a threshold to zero.

    Parameters
    ----------
    waveform : array-like
        Input waveform.
    threshold : float
        Values below this threshold will be set to zero.

    Returns
    -------
    array-like
        Waveform with values below the threshold suppressed.
    """
    wf = np.copy(waveform)
    below_thr = wf <= threshold
    wf[below_thr] = 0
    return wf


def noise_suppression(waveforms, threshold=0):
    """
    Set values from a set of waveforms below a threshold to zero.

    Parameters
    ----------
    waveforms : array-like
        Input waveforms.
    threshold : float
        Values below this threshold will be set to zero.

    Returns
    -------
    array-like
        Waveform array with values below the threshold suppressed.
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

def peak_height_deconv(zs_waveform, peaks, heights, thr=50):
    if len(peaks)==1:
        return heights
    for idx_peak in range(len(peaks)-1):
        zswf_sel = zs_waveform[peaks[idx_peak]:peaks[idx_peak+1]]
        if np.any(zswf_sel == 0):
            continue
        else:
            heights[idx_peak+1] -= np.min(zswf_sel)
    return heights[heights > thr]

def peak_height_deconv_reject_second_peak(zs_waveform, peaks, heights, thr=50):
    # If there is only one peak, no need to process, return heights and peaks as is
    if len(peaks)==1:
        return heights, peaks
    
    indxs_to_remove = []
    for idx_peak in range(len(peaks) - 1):
        zswf_sel = zs_waveform[peaks[idx_peak]:peaks[idx_peak + 1]]
        if np.any(zswf_sel == 0):
            continue
        else:
            indxs_to_remove.append(idx_peak + 1)  # Mark the second peak for removal

    # If there are peaks to remove, update heights and peaks
    if len(indxs_to_remove) != 0:
        # Ensure that indxs_to_remove is a boolean mask
        mask = np.ones(len(heights), dtype=bool)
        mask[indxs_to_remove] = False

        heights = heights[mask]
        peaks   = peaks  [mask]

    return heights[heights > thr], peaks[heights > thr]


def peak_height_deconv_save_secondary_peaks(zs_waveform, peaks, heights):
    """This function saves only the secondary peaks to study pile-up.
       It saves the absolute and relative height of the second peak."""
    if len(peaks)==1:
        return np.array([]), np.array([]), np.array([])

    indxs_to_save   = []
    abs_sec_heights = []
    rel_sec_heights = []
    for idx_peak in range(len(peaks) - 1):
        zswf_sel = zs_waveform[peaks[idx_peak]:peaks[idx_peak + 1]]
        if np.any(zswf_sel == 0):
            continue
        else:
            indxs_to_save  .append(peaks  [idx_peak + 1])  # Mark the second peak for saving
            abs_sec_heights.append(heights[idx_peak + 1])
            rel_sec_heights.append(heights[idx_peak + 1] - np.min(zswf_sel))

    return np.array(indxs_to_save), np.array(abs_sec_heights), np.array(rel_sec_heights)


def peak_height_deconv_indexes(zs_waveform, peaks, heights, thr=50):
    if len(peaks)==1:
        return peaks
    for idx_peak in range(len(peaks)-1):
        zswf_sel = zs_waveform[peaks[idx_peak]:peaks[idx_peak+1]]
        if np.any(zswf_sel == 0):
            continue
        else:
            heights[idx_peak+1] -= np.min(zswf_sel)
    return peaks[heights > thr]

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

def integrate_and_get_len_peaks(waveform, peaks):
    if np.isscalar(peaks):  # Handles single integer or float
        peaks = [peaks]

    peaks_indxs = []
    peaks_lens  = []
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
        peaks_lens .append(len(peak_indxs)) # Num of timesamples of the ZS peak
    
    return np.array([np.sum(waveform[idx]) for idx in peaks_indxs]), np.array(peaks_lens)

def peak_height(waveform, peaks):
    """
    Return the values of a waveform at given peak positions.

    Parameters
    ----------
    waveform : array-like
        Input waveform.
    peaks : array-like
        Indices of the peaks in the waveform.

    Returns
    -------
    numpy.ndarray
        Array of waveform values at the specified peak indices.
    """
    return np.array([waveform[peak] for peak in peaks])


def integrate_and_get_len_peaks_fast(waveform, peaks):
    """
    Compute the area and width of peaks in a waveform.

    For each peak index, the function expands left and right while the
    waveform remains positive, then integrates (sum) the values in that
    region and returns its length.

    Parameters
    ----------
    waveform : array-like
        Input waveform.
    peaks : int or array-like
        Peak index or list/array of peak indices.

    Returns
    -------
    areas : numpy.ndarray
        Integrated values (sum) of each peak region.
    lengths : numpy.ndarray
        Number of samples in each peak region.
    """
    if np.isscalar(peaks):
        peaks = [peaks]
    elif isinstance(peaks, np.ndarray) and peaks.ndim == 0:
        peaks = [int(peaks)]

    areas   = []
    lengths = []

    for p in peaks:
        # Expand left
        left = p
        while left > 0 and waveform[left - 1] > 0:
            left -= 1

        # Expand right
        right = p
        while right < len(waveform) - 1 and waveform[right + 1] > 0:
            right += 1

        inds   = np.arange(left, right + 1)
        values = waveform[inds]

        areas  .append(np.sum(values))
        lengths.append(len(inds))

    return np.array(areas), np.array(lengths)


def get_peaks_peakutils(waveform, thres=0.35, min_dist=100, thres_abs=False):
    """
    Detect peak positions in a waveform using peakutils.

    Parameters
    ----------
    waveform : array-like
        Input waveform.
    thres : float, optional
        Threshold for peak detection. Interpreted as a fraction of the
        maximum value unless `thres_abs=True`.
    min_dist : int, optional
        Minimum number of samples separating adjacent peaks.
    thres_abs : bool, optional
        If True, `thres` is treated as an absolute amplitude threshold.

    Returns
    -------
    numpy.ndarray
        Indices of detected peaks.
    """
    return peakutils.indexes(waveform, thres=thres, min_dist=min_dist, thres_abs=thres_abs)

def area_and_len_of_peaks(waveforms, peaks):
    all_areas_and_lens = [integrate_and_get_len_peaks(wf, pk) for wf, pk in zip(waveforms, peaks)]
    all_areas          = np.concatenate([areas for areas, _ in all_areas_and_lens])
    all_lens           = np.concatenate([lens  for _, lens  in all_areas_and_lens])
    return all_areas, all_lens

def area_and_len_of_peaks_no_concat(waveforms, peaks):
    all_areas_and_lens = [integrate_and_get_len_peaks(wf, pk) for wf, pk in zip(waveforms, peaks)]
    all_areas          = np.array([areas for areas, _ in all_areas_and_lens], dtype=object)
    all_lens           = np.array([lens  for _, lens  in all_areas_and_lens], dtype=object)
    return all_areas, all_lens

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

def get_peaks_using_peakutils_no_PMT(RawTree, channel, num_wfs=None, sipm_thr=50, peak_range=(650,850), wf_range_bsl=(0, None), sg_filter=False, window_length=None, polyorder=None):
    all_raw_wfs = wfs_from_rawtree(RawTree, channel)[:num_wfs]
    #all_raw_wfs = np.array([wf for wf in all_raw_wfs if np.std(wf) > 15]) #Way of removing the baseline wfs

    if channel in [9, 10, 11]: #trigger SiPMs
        all_raw_wfs = np.array([blr.pmt_deconvolver(wf, wf_range_bsl=wf_range_bsl) for wf in all_raw_wfs])

    ## Subtract baseline
    partial_subtract_baseline = partial(subtract_baseline, wf_range_bsl=wf_range_bsl)
    subt_raw_wfs = list(map(partial_subtract_baseline, all_raw_wfs))

    if sg_filter:
        if window_length==None or polyorder==None:
            raise ValueError
        subt_raw_wfs = savgol_filter(subt_raw_wfs, window_length=window_length, polyorder=polyorder)

    ## Zero suppression
    zs_raw_wfs = noise_suppression(subt_raw_wfs, threshold=sipm_thr)

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

def get_values_thr_from_zswf(waveform, idx_peaks):
    """Get the indices when the peaks cross the set threshold,
       that is set in a previous step to compute the ZS wfs
    """
    ## The wfs should be zeros except for the peaks
    vals_thr = np.ones(len(idx_peaks))
    for i, peak in enumerate(idx_peaks):
        if i==0:
            if len(np.where(waveform[:peak]>0)[0])==0:
                vals_thr[i] = peak
            else:
                vals_thr[i] = np.where(waveform[:peak]>0)[0][0]
        else:
            zeros_in_range = np.where(waveform[idx_peaks[i-1]:peak]==0)[0]
            if len(zeros_in_range)==0:
                vals_thr[i] = np.argmin(waveform[idx_peaks[i-1]:peak]) + idx_peaks[i-1]
            else:
                vals_thr[i] = idx_peaks[i-1] + zeros_in_range[-1] + 1
    return vals_thr

def find_crossings(wf_x, waveform, level):
    """Find where the waveform crosses the level
    """
    crosses = []
    for i in range(1, len(waveform)):
        if (waveform[i-1] < level and waveform[i] >= level) or (waveform[i-1] > level and waveform[i] <= level):
            # Linear interpolation to get a more accurate crossing point
            x_cross = wf_x[i-1] + (wf_x[i] - wf_x[i-1]) * (level - waveform[i-1]) / (waveform[i] - waveform[i-1])
            crosses.append(x_cross)
    return crosses

def get_evt_trigger_t(wf, thr_ADC_trigg=200, rng=(1400, 1500)):
    """Get trigger time in SAMPLE NUMBER from waveform
    """
    wf_x        = np.arange(len(wf))
    sel_wf_x    = wf_x[(wf_x > rng[0]) & (wf_x < rng[1])]
    sel_wf_y    = wf  [(wf_x > rng[0]) & (wf_x < rng[1])]
    crossing_pt = find_crossings(sel_wf_x, sel_wf_y, thr_ADC_trigg)
    if len(crossing_pt)==0:
        return 0
    else:
        return crossing_pt[0]
