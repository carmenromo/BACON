import sys
import uproot
import numpy          as np
import peak_functions as pf


arguments = pf.parse_args(sys.argv)
in_path   = arguments.in_path
file_name = arguments.file_name
out_path  = arguments.out_path

if file_name.endswith(".root"):
    file_name = file_name[:-5]
filename = f"{in_path}/{file_name}.root"
infile   = uproot.open(filename)
RawTree  = infile['RawTree']

outfile = f"{out_path}/BACoN_data_basic_run_info_{file_name}"

all_chs      = range(13)
max_smpl_bsl = 650

bsl_mean_i_dict = {}
bsl_mode_i_dict = {}
bsl_mean_f_dict = {}
bsl_mode_f_dict = {}
std_all_dict    = {}
max_all_dict    = {}
for ch in all_chs:
    print(f'Channel {ch}')
    ## 0) Get waveforms:
    all_wfs = pf.wfs_from_rawtree(RawTree, ch)

    ## 1) Compute the baseline for each channel (mean and mode at the beginning and at the end of each wf)
    bsl_mean_i_dict[ch] = pf.compute_baseline(all_wfs[:, :max_smpl_bsl],  mode=False)
    bsl_mode_i_dict[ch] = pf.compute_baseline(all_wfs[:, :max_smpl_bsl],  mode=True)
    bsl_mean_f_dict[ch] = pf.compute_baseline(all_wfs[:, -max_smpl_bsl:], mode=False)
    bsl_mode_f_dict[ch] = pf.compute_baseline(all_wfs[:, -max_smpl_bsl:], mode=True)

    ## 2) Std of the waveforms
    std_all_dict[ch] = np.std(all_wfs, axis=1)

    ## 3) Max value of each waveform
    max_all_dict[ch] = np.max(all_wfs, axis=1)


np.savez(outfile,
         bsl_mean_i_dict=bsl_mean_i_dict,
         bsl_mode_i_dict=bsl_mode_i_dict,
         bsl_mean_f_dict=bsl_mean_f_dict,
         bsl_mode_f_dict=bsl_mode_f_dict,
         std_all_dict=std_all_dict,
         max_all_dict=max_all_dict)