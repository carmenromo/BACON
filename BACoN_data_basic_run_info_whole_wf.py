import uproot
import numpy          as np
import peak_functions as pf


arguments = pf.parse_args()
in_path   = arguments.in_path
file_name = arguments.file_name
out_path  = arguments.out_path

if file_name.endswith(".root"):
    file_name = file_name[:-5]
filename = f"{in_path}/{file_name}.root"
infile   = uproot.open(filename)
RawTree  = infile['RawTree']

outfile = f"{out_path}/BACoN_data_basic_info_whole_wf_{file_name}"

all_chs = range(13)

bsl_mode_dict0 = {}
bsl_mean_dict0 = {}
bsl_mode_dict1 = {}
bsl_mean_dict1 = {}
for ch in all_chs:
    print(f'Channel {ch}')
    ## 0) Get waveforms:
    all_wfs = pf.wfs_from_rawtree(RawTree, ch)

    ## 1) Compute the baseline for each channel (mode and mean for the whole wf)
    bsl_mode_dict0[ch] = np.array([pf.compute_baseline(wf, mode=True)                         for wf in all_wfs])
    bsl_mean_dict0[ch] = np.array([pf.compute_baseline(wf, mode=False)                        for wf in all_wfs])
    bsl_mode_dict1[ch] = np.array([pf.compute_baseline(wf, mode=True,  wf_range_bsl=(0, 650)) for wf in all_wfs])
    bsl_mean_dict1[ch] = np.array([pf.compute_baseline(wf, mode=False, wf_range_bsl=(0, 650)) for wf in all_wfs])

np.savez(outfile,
         bsl_mode0=bsl_mode_dict0,
         bsl_mean0=bsl_mean_dict0,
         bsl_mode1=bsl_mode_dict1,
         bsl_mean1=bsl_mean_dict1)