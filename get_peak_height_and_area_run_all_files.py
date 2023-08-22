
import os
import sys
import glob
import subprocess
import uproot

import numpy as np

files_dir         = sys.argv[1]
files_prefix_name = sys.argv[2]
out_path          = sys.argv[3]

#for filename in os.listdir(files_dir):
all_files = glob.glob(f"{files_dir}/{files_prefix_name}" + '*')
for file_name in np.sort(all_files):

    filename = file_name.split("/")[-1]

    print('--------------------------------------------')
    print("Analyzing file: ", filename)
    print('--------------------------------------------')

    # Call get_peak_height.py script using subprocess
    # try:
    #     subprocess.call(['python', 'get_peak_height_and_area.py', files_dir, filename, out_path])
    # except FileNotFoundError:
    #     subprocess.call(['python3', 'get_peak_height_and_area.py', files_dir, filename, out_path])
    # except uproot.exceptions.KeyInFileError:
    #     continue

    try:
        subprocess.call(['python', 'BACoN_signal_processing_get_peak_height_and_area.py', files_dir, filename, out_path])
    except FileNotFoundError:
        subprocess.call(['python3', 'BACoN_signal_processing_get_peak_height_and_area.py', files_dir, filename, out_path])
    except uproot.exceptions.KeyInFileError:
        continue