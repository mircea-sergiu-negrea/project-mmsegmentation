#!/usr/bin/env python3
"""
Extract the 'para' modality from AP4AD .npy input files and save per-frame .npy files.

This script mirrors the style of the existing depth extractor. It:
 - scans the source folder for inputs_<seq>_<frame>.npy
 - extracts channel 12 (indices 12:13) which corresponds to 'para'
 - saves outputs to /home/negreami/datasets/ap4ad/para/<seq>/<seq>_<frame>.npy
 - uses multiprocessing to speed up conversion

Run as a standalone script. Configure NUM_WORKERS at the top as needed.
"""

import os
import numpy as np
import glob
import time
from multiprocessing import Pool


# Number of worker processes (set this to the number of CPU cores you want to use)
NUM_WORKERS = 2  # <-- Change this value as needed

# Source and destination paths
INPUT_PATH = '/home/negreami/datasets/ap4ad/AP-for-AD_data'
SAVE_DIR = '/home/negreami/datasets/ap4ad/para'
os.makedirs(SAVE_DIR, exist_ok=True)

# Find all .npy input files
all_input_files = sorted(glob.glob(os.path.join(INPUT_PATH, 'inputs_*_*.npy')))

# Prepare tasks list of (sequence_id, frame_id)
tasks = []
for input_file in all_input_files:
    filename = os.path.basename(input_file)
    parts = filename.split('_')
    sequence_id = parts[1]
    frame_id = parts[2].split('.')[0]
    tasks.append((sequence_id, frame_id))


def process_para(args):
    sequence_id, frame_id = args
    sequence_dir = os.path.join(SAVE_DIR, sequence_id)
    os.makedirs(sequence_dir, exist_ok=True)
    input_file = os.path.join(INPUT_PATH, f'inputs_{sequence_id}_{frame_id}.npy')
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

    # Load source .npy and extract para channel (index 12)
    data = np.load(input_file)
    # para is channels 12:13 -> shape [H, W, 1]
    para = data[:, :, 12:13]
    # Save as .npy
    np.save(output_file, para)
    elapsed = time.time() - start_time
    print(f"frame {sequence_id}_{frame_id} to para done in {elapsed:.4f} seconds")


if __name__ == '__main__':
    print(f"Processing {len(tasks)} files using {NUM_WORKERS} workers...")
    with Pool(NUM_WORKERS) as pool:
        pool.map(process_para, tasks)
    print("All done.")
