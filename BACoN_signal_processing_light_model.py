
import sys
import uproot
import time

import numpy as np

import peak_functions as pf
import blr_functions  as blr

start_time = time.time()

arguments = pf.parse_args(sys.argv)
in_path   = arguments.in_path
file_name = arguments.file_name
out_path  = arguments.out_path

total_SiPMs = 9
dead_SiPMs  = []

sum_wfs_all_chs = [[] for i in range(total_SiPMs)]

if file_name.endswith(".root"):
    file_name = file_name[:-5]
filename = f"{in_path}/{file_name}.root"
infile   = uproot.open(filename)
RawTree  = infile['RawTree']

outfile = f"{out_path}/BACoN_sig_processing_light_model_{file_name}"

pretrigg_range = (0, 600)
max_smpl_bsl   = 650
sipm_thr       = 50 #ADCs

def wfs_from_rawtree(RawTree, channel):
    return np.array(RawTree[f'chan{channel}/rdigi'].array())

##Trigger SiPMs
trigger_chs     = [9, 10, 11]
trigg_cwfs_dict = {ch: np.array([blr.pmt_deconvolver(wf, wf_range_bsl=(0, max_smpl_bsl))
                                 for wf in wfs_from_rawtree(RawTree, ch)])
                                 for ch in trigger_chs}

subt_wfs_dict = {ch: pf.subtract_baseline(wfs_from_rawtree(RawTree, ch),
                                          mode=True,
                                          wf_range_bsl=(0, max_smpl_bsl))
                 for ch in range(9)}

all_subt_wfs_dict = {**trigg_cwfs_dict, **subt_wfs_dict}

good_evts = []
for evt in range(len(all_subt_wfs_dict[0])):
    wfs_trigg_reg = np.concatenate([all_subt_wfs_dict[ch][evt][pretrigg_range[0]:pretrigg_range[1]]
                                    for ch in all_subt_wfs_dict.keys()])
    if np.max(wfs_trigg_reg)>100:
        continue
        
    wfs_sig_reg = np.concatenate([subt_wfs_dict[ch][evt][pretrigg_range[1]:]
                                  for ch in subt_wfs_dict.keys()])
    ## Zero suppression
    zs_raw_wfs = pf.noise_suppression(wfs_sig_reg, threshold=sipm_thr)
    if np.sum(zs_raw_wfs)==0:
        continue

    good_evts.append(evt)
    if evt%100==0:
        print(evt)

for ch in all_subt_wfs_dict.keys():
    sum_wf = np.sum(np.array([all_subt_wfs_dict[ch][evt] for evt in good_evts]), axis=0)
    sum_wfs_all_chs[ch].append(sum_wf)

sum_wfs_all_chs = np.array(sum_wfs_all_chs)
good_evts       = np.array(good_evts)

np.savez(outfile, 
         sum_wfs_all_chs=sum_wfs_all_chs,
         good_evts=good_evts)

end_time = time.time()
execution_time = start_time - end_time
print("Execution time:",execution_time)