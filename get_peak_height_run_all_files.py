
import os
import sys
import subprocess

files_dir = sys.argv[1]
out_path  = sys.argv[2]

for filename in os.listdir(files_dir):
    filepath = os.path.join(files_dir, filename)

    print('--------------------------------------------')
    print("Analyzing file: ", filename)
    print('--------------------------------------------')

    # Call get_peak_height.py script using subprocess
    subprocess.call(['python', 'get_peak_height.py', files_dir, filename, out_path])