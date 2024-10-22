import sys
import uproot
import numpy          as np
import peak_functions as pf

from functools    import partial


arguments = pf.parse_args(sys.argv)
in_path   = arguments.in_path
file_name = arguments.file_name
out_path  = arguments.out_path

if file_name.endswith(".root"):
    file_name = file_name[:-5]
filename = f"{in_path}/{file_name}.root"
infile   = uproot.open(filename)
RawTree  = infile['RawTree']

outfile = f"{out_path}/BACoN_event_rates_{file_name}"


def get_peaks(wfs, sipm_thr=50, peak_range=(650,850), wf_range_bsl=(0, None), sg_filter=False, window_length=None, polyorder=None):
    ## Subtract baseline
    partial_subtract_baseline = partial(pf.subtract_baseline, wf_range_bsl=wf_range_bsl)
    subt_raw_wfs = list(map(partial_subtract_baseline, wfs))

    if sg_filter:
        if window_length==None or polyorder==None:
            raise ValueError
        subt_raw_wfs = savgol_filter(subt_raw_wfs, window_length=window_length, polyorder=polyorder)

    ## Zero suppression
    zs_raw_wfs = pf.noise_suppression(subt_raw_wfs, threshold=sipm_thr)

    ## Remove events with no signal in the ROI
    empty_evts        = np.array([idx for idx, zwf in enumerate(zs_raw_wfs) if np.sum(zwf[peak_range[0]:peak_range[1]])==0])
    filter_empty_zwfs = pf.remove_waveforms_by_indices(zs_raw_wfs,   empty_evts)
    subt_raw_wfs_filt = pf.remove_waveforms_by_indices(subt_raw_wfs, empty_evts)

    ## Get the peaks found in the ROI
    all_peaks         = list(map(pf.get_peaks_peakutils, filter_empty_zwfs))
    return filter_empty_zwfs, subt_raw_wfs_filt, all_peaks



all_chs      = range(9) # we're not interested in the rate of trigger SiPMs (an event starts when they trigger)
max_smpl_bsl = 650
sipm_thr     = 50
peak_range   = (0, 7500)

peak_height_all_chs = [[] for i in all_chs]
peak_index_all_chs  = [[] for i in all_chs]

## Thr values valid from 9/10/2024 since the bas voltage was changed
std_thr_dict = {0: 13,
                1: 13,
                2: 13,
                3: 13,
                4: 14,
                5: 13,
                6: 12,
                7: 13,
                8: 13}

for ch in all_chs:
    print(f'Channel {ch}')
    ## 0) Get waveforms:
    all_wfs = pf.wfs_from_rawtree(RawTree, ch)

    ## 1) Filter waveforms
    std_devs = np.std(all_wfs, axis=1)
    filt_wfs = all_wfs[std_devs > std_thr_dict[ch]]
    evt_numb = np.arange(len(all_wfs))[std_devs > std_thr_dict[ch]]

    ## 1) Get the peaks
    _, subt_wfs_filt, all_peaks = get_peaks(filt_wfs, sipm_thr=sipm_thr, peak_range=peak_range)
    heights                     = pf.height_of_peaks(subt_wfs_filt, all_peaks)
    peak_height_all_chs[ch].append(heights)
    peak_index_all_chs [ch].append(all_peaks)

peak_height_all_chs = np.array(peak_height_all_chs, dtype=object)
peak_index_all_chs  = np.array(peak_index_all_chs,  dtype=object)

np.savez(outfile, 
         peak_height_all_chs=peak_height_all_chs,
         peak_index_all_chs=peak_index_all_chs)