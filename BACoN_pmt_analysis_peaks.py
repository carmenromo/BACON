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

all_chs      = range(13)
max_smpl_bsl = 650
thr_ADC_pmt  = 20
min_dist_pmt = 15

## Thr values valid from 9/10/2024 since the bias voltage was changed
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
                11: 40,
                12: 4}

ch    = 12
wfs   = pf.wfs_from_rawtree(RawTree, ch)
swfs  = np.array([blr.pmt_deconvolver(wf, wf_range_bsl=(0, max_smpl_bsl), baseline_mode=False, std_lim=3*std_thr_dict[ch]) for wf in wfs])
zswfs = pf.noise_suppression(swfs, threshold=thr_ADC_pmt)
partial_get_peaks_peakutils = partial(pf.get_peaks_peakutils, thres=thr_ADC_pmt, min_dist=min_dist_pmt, thres_abs=True)
idx_peaks_max               = np.array(list(map(partial_get_peaks_peakutils, zswfs)), dtype=object)
height_peaks                = pf.height_of_peaks(zswfs, idx_peaks_max)       ## Heights and integrals concatenated
integral_peaks, len_peaks   = pf.area_and_len_of_peaks(zswfs, idx_peaks_max) ## Heights and integrals concatenated

np.savez(outfile,
         h_peaks_pmt=height_peaks,
         i_peaks_pmt=integral_peaks,
         l_peaks_pmt=len_peaks)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} s, {elapsed_time/60} mins")