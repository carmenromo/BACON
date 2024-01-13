
import sys
import glob
import subprocess
import uproot

import numpy as np

script            = sys.argv[1]
files_dir         = sys.argv[2]
files_prefix_name = sys.argv[3]
out_path          = sys.argv[4]

all_files = glob.glob(f"{files_dir}/{files_prefix_name}" + '*')
for file_name in np.sort(all_files):

    filename = file_name.split("/")[-1]

    print('--------------------------------------------')
    print("Analyzing file: ", filename)
    print('--------------------------------------------')

    try:
        subprocess.call(['python3', script, files_dir, filename, out_path])
    except FileNotFoundError:
        "FileNotFoundError"
        continue
    except uproot.exceptions.KeyInFileError:
        "uproot.exceptions.KeyInFileError"
        continue