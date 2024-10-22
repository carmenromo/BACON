
import sys
import glob
import uproot

import numpy as np

import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from scipy          import stats   as st


data_path = '/mnt/Data2/BaconRun3Data/rootData/' #'/Users/romoluque_c/LEGEND/BACON/new_setup/datatest/' #sys.argv[1]
date_data = '09_02_2024'
file_num  = 6
filename  = np.sort(glob.glob(data_path + f'/run-{date_data}-*.root'))[file_num]
infile    = uproot.open(filename)
RawTree   = infile['RawTree']

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

def gaussian(x, amplitude, mean, stddev):
    return amplitude * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2))


### Sum waveforms
fig, axs = plt.subplots(4, 3, figsize=(16, 14))
for row, ich in enumerate(reversed(np.arange(0, 12, 3))):
    for col in range(3):
        ch = ich + col
        all_wfs = np.array(RawTree[f'chan{ch}/rdigi'].array())
        sum_wfs = np.sum(all_wfs, axis=0)
        axs[row][col].plot(2*np.arange(len(sum_wfs)), sum_wfs, linewidth=0.5, color='#418094', alpha=0.9)
        axs[row][col].set_xlabel('Time window (ns)', fontsize=14)
        axs[row][col].set_ylabel('Amplitude (ADC)',  fontsize=14)
        axs[row][col].set_title(f"Channel {ch}",     fontsize=16)
        axs[row][col].tick_params(axis='x', labelsize=10)
        axs[row][col].tick_params(axis='y', labelsize=10)
        
fname = filename.split("/")[-1]
fig.suptitle(f'Sum of all waveforms with no cuts for file: {fname}')
plt.tight_layout()
#plt.show()


### RMS of waveforms
fig, axs   = plt.subplots(4, 3, figsize=(16, 14))
for row, ich in enumerate(reversed(np.arange(0, 12, 3))):
    for col in range(3):
        ch = ich + col
        all_wfs  = np.array(RawTree[f'chan{ch}/rdigi'].array())
        subt_wfs = subtract_baseline(all_wfs, mode=True, wf_range_bsl=(0, 500))
        rms_vals = np.std(all_wfs, axis=1)
        
        if ich in [9, 10, 11]:
            axs[row][col].hist(rms_vals, bins=100, range=(0, 300), color='#418094', log=True)
        else:
            axs[row][col].hist(rms_vals, bins=100, range=(8, 22), color='#418094', log=True)
        axs[row][col].set_xlabel('Std wfs (ADC)', fontsize=14)
        axs[row][col].set_ylabel('Entries/bin',  fontsize=14)
        axs[row][col].set_title(f"Channel {ch}",     fontsize=16)
        axs[row][col].tick_params(axis='x', labelsize=10)
        axs[row][col].tick_params(axis='y', labelsize=10)
        
plt.suptitle(f'Waveforms RMS for file {fname}')
plt.tight_layout()
#plt.show()


fig, axs   = plt.subplots(3, 3, figsize=(16, 11))
for row, ich in enumerate(reversed(np.arange(0, 9, 3))):
    for col in range(3):
        ch = ich + col
        all_wfs  = np.array(RawTree[f'chan{ch}/rdigi'].array())
        subt_wfs = subtract_baseline(all_wfs, mode=True, wf_range_bsl=(0, 500))
        rms_vals = np.std(subt_wfs, axis=1)

        #y, x, _ = axs[row][col].hist(rms_vals, bins=100, range=(8, 18), color=fav_col, log=True)
        y, x = np.histogram(rms_vals, bins=100, range=(8, 18))

        initial_guess = [np.max(y), x[np.argmax(y)], 0.15]  # stddev is initialized to 1
        (_, mean_fit, std_fit), _ = curve_fit(gaussian, x[1:], y, p0=initial_guess)

        rms_thr  = mean_fit + 5*std_fit
        filt_wfs = subt_wfs[rms_vals > rms_thr]
        sum_fwfs = np.sum(filt_wfs, axis=0)
        axs[row][col].plot(2*np.arange(len(sum_fwfs)), sum_fwfs, linewidth=0.5, color='#418094', alpha=0.9)
        axs[row][col].set_title(f"Channel {ch}",  fontsize=16)
        axs[row][col].set_xlabel('Std wfs (ADC)', fontsize=14)
        axs[row][col].set_ylabel('Entries/bin',   fontsize=14)
        axs[row][col].tick_params(axis='x',       labelsize=10)
        axs[row][col].tick_params(axis='y',       labelsize=10)
        
plt.suptitle(f'Sum Waveforms with signal for file {fname}')
plt.tight_layout()
plt.show()