
import os
import sys
import glob
import subprocess
import uproot

files_dir         = sys.argv[1]
files_prefix_name = sys.argv[2]
out_path          = sys.argv[3]

#for filename in os.listdir(files_dir):
for file_name in glob.glob(f"{files_dir}/{files_prefix_name}" + '*'):

    filename = file_name.split("/")[-1]

    print('--------------------------------------------')
    print("Analyzing file: ", filename)
    print('--------------------------------------------')

    # Call get_peak_height.py script using subprocess
    try:
        subprocess.call(['python', 'get_peak_height.py', files_dir, filename, out_path])
    except FileNotFoundError:
        subprocess.call(['python3', 'get_peak_height.py', files_dir, filename, out_path])
    except uproot.exceptions.KeyInFileError:
        continue