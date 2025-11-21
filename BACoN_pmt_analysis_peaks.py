import time
import uproot
import numpy          as np
import peak_functions as pf
import blr_functions  as blr

from functools import partial

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

outfile = f"{out_path}/BACoN_pmt_analysis_peaks_{file_name}"

## Parameters
ch           = 12
std_thr      = 4
max_smpl_bsl = 650
thr_ADC_pmt  = 20
min_dist_pmt = 15

## In the deconvolution the baseline is already subtracted from the waveform
swfs = np.array([blr.deconvolver(wf, wf_range_bsl=(0, max_smpl_bsl), baseline_mode=False, std_lim=3*std_thr)
                 for wf in pf.wfs_from_rawtree(RawTree, ch)])

## Noise suppression
zswfs = pf.noise_suppression(swfs, threshold=thr_ADC_pmt)

## Get peaks above the chosen threshold thr_ADC_pmt
partial_get_peaks_peakutils = partial(pf.get_peaks_peakutils, thres=thr_ADC_pmt, min_dist=min_dist_pmt, thres_abs=True)

## Indices of the maximum of the peak
idx_peaks_max = np.array([partial_get_peaks_peakutils(wf) for wf in zswfs], dtype=object)

## Height of the max of the peak after SG filter
height_peaks = np.array([pf.peak_height(wf, idx_peaks_max[i])
                         for i,wf in enumerate(zswfs)], dtype=object)

## Integral and length of the integrated peaks
integral_results = [pf.integrate_and_get_len_peaks_fast(wf, peaks)
                    for wf, peaks in zip(zswfs, idx_peaks_max)]

integral_peaks = np.array([r[0] for r in integral_results], dtype=object)
len_peaks      = np.array([r[1] for r in integral_results], dtype=object)

np.savez(outfile,
         h_peaks_pmt=height_peaks,
         i_peaks_pmt=integral_peaks,
         l_peaks_pmt=len_peaks)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} s, {elapsed_time/60} mins")