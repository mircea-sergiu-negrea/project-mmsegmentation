# Script to extract depth channel from .npy input files and save as .npy depth maps

import os
import numpy as np
import glob
import time
from multiprocessing import Pool


# Number of worker processes (set this to the number of CPU cores you want to use)
NUM_WORKERS = 2  # <-- Change this value as needed

# Path to the directory containing the .npy input files
input_path = '/home/negreami/datasets/ap4ad/AP-for-AD_data'
# Directory to save the output depth .npy files
save_dir = '/home/negreami/datasets/ap4ad/depth'
os.makedirs(save_dir, exist_ok=True)

# Find all .npy files matching the pattern inputs_*_*.npy
all_input_files = sorted(glob.glob(os.path.join(input_path, 'inputs_*_*.npy')))

# Prepare a list of (sequence_id, frame_id) pairs
tasks = []
for input_file in all_input_files:
    filename = os.path.basename(input_file)
    parts = filename.split('_')
    sequence_id = parts[1]
    frame_id = parts[2].split('.')[0]
    tasks.append((sequence_id, frame_id))

def process_depth(args):
    sequence_id, frame_id = args
    sequence_dir = os.path.join(save_dir, sequence_id)
    os.makedirs(sequence_dir, exist_ok=True)
    input_file = os.path.join(input_path, f'inputs_{sequence_id}_{frame_id}.npy')
    output_file = os.path.join(sequence_dir, f'{sequence_id}_{frame_id}.npy')
    start_time = time.time()
    if os.path.exists(output_file):
        elapsed = time.time() - start_time
        print(f"Skipping frame {sequence_id}_{frame_id}, already exists. (skipped in {elapsed:.4f} seconds)")
        return
    if not os.path.exists(input_file):
        elapsed = time.time() - start_time
        print(f'Missing input file: {input_file} (checked in {elapsed:.4f} seconds)')
        return
    # Load the .npy file (assumed to be an image with at least 8 channels)
    input_data = np.load(input_file)
    # Extract the depth channel (channel 7)
    depth = input_data[:, :, 7:8]  # Shape [H, W, 1]
    # Save as .npy file (preserves full precision)
    np.save(output_file, depth)
    elapsed = time.time() - start_time
    print(f"frame {sequence_id}_{frame_id} to depth done in {elapsed:.4f} seconds")

if __name__ == '__main__':
    print(f"Processing {len(tasks)} files using {NUM_WORKERS} workers...")
    with Pool(NUM_WORKERS) as pool:
        pool.map(process_depth, tasks)
    print("All done.")
