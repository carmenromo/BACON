import os
import sys
import time
import uproot
import numpy  as np

# Add the parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import peak_functions as pf


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

"""to run this script:

python3 BACoN_data_baseline.py /wherever/your/files/are my_file.root /output/data/"""

outfile = f"{out_path}/BACoN_data_baseline_{file_name}"

all_chs      = range(13) # Includes the PMT
max_smpl_bsl = 650

## Thr values valid for run3 starting on 9/10/2024 since the bias voltage was changed
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

bsl_mean_i_dict = {}
bsl_mode_i_dict = {}
bsl_mean_f_dict = {}
bsl_mode_f_dict = {}
bsl_mean_w_dict = {} #Check baseline for the whole waveform
bsl_mode_w_dict = {}
std_all_dict    = {}
max_all_dict    = {}
for ch in all_chs:
    print(f'Channel {ch}')
    ## 0) Get waveforms:
    all_wfs = pf.wfs_from_rawtree(RawTree, ch)

    ## 1) Compute the baseline for each channel (mean and mode at the beginning and at the end of each wf)
    bsl_mode_i_dict[ch] = np.array([pf.compute_baseline_std_lim(wf, mode=True,  wf_range_bsl=(0, max_smpl_bsl),         std_lim=3*std_thr_dict[ch]) for wf in all_wfs])
    bsl_mean_i_dict[ch] = np.array([pf.compute_baseline_std_lim(wf, mode=False, wf_range_bsl=(0, max_smpl_bsl),         std_lim=3*std_thr_dict[ch]) for wf in all_wfs])
    bsl_mode_f_dict[ch] = np.array([pf.compute_baseline_std_lim(wf, mode=True,  wf_range_bsl=(7500-max_smpl_bsl, 7500), std_lim=3*std_thr_dict[ch]) for wf in all_wfs])
    bsl_mean_f_dict[ch] = np.array([pf.compute_baseline_std_lim(wf, mode=False, wf_range_bsl=(7500-max_smpl_bsl, 7500), std_lim=3*std_thr_dict[ch]) for wf in all_wfs])
    bsl_mode_w_dict[ch] = np.array([pf.compute_baseline_std_lim(wf, mode=True,                                          std_lim=3*std_thr_dict[ch]) for wf in all_wfs])
    bsl_mean_w_dict[ch] = np.array([pf.compute_baseline_std_lim(wf, mode=False,                                         std_lim=3*std_thr_dict[ch]) for wf in all_wfs])

    ## 2) Std of the waveforms
    std_all_dict[ch] = np.std(all_wfs, axis=1)

    ## 3) Max value of each waveform
    max_all_dict[ch] = np.max(all_wfs, axis=1)


np.savez(outfile,
         bsl_mean_i_dict=bsl_mean_i_dict,
         bsl_mode_i_dict=bsl_mode_i_dict,
         bsl_mean_f_dict=bsl_mean_f_dict,
         bsl_mode_f_dict=bsl_mode_f_dict,
         bsl_mean_w_dict=bsl_mean_w_dict,
         bsl_mode_w_dict=bsl_mode_w_dict,
         std_all_dict=std_all_dict,
         max_all_dict=max_all_dict)


end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} s, {elapsed_time/60} mins")