#!/usr/bin/env python3
"""
Extract the 'flow' modality from AP4AD .npy input files and save per-frame .npy files.

This script extracts channels 21:23 (two-channel optical flow) and saves them under
/home/negreami/datasets/ap4ad/flow/<seq>/<seq>_<frame>.npy.

It uses multiprocessing for speed and mirrors the existing depth extractor behavior.
"""

import os
import numpy as np
import glob
import time
from multiprocessing import Pool


# Number of worker processes
NUM_WORKERS = 2

# Paths
INPUT_PATH = '/home/negreami/datasets/ap4ad/AP-for-AD_data'
SAVE_DIR = '/home/negreami/datasets/ap4ad/flow'
os.makedirs(SAVE_DIR, exist_ok=True)

all_input_files = sorted(glob.glob(os.path.join(INPUT_PATH, 'inputs_*_*.npy')))

tasks = []
for input_file in all_input_files:
    filename = os.path.basename(input_file)
    parts = filename.split('_')
    sequence_id = parts[1]
    frame_id = parts[2].split('.')[0]
    tasks.append((sequence_id, frame_id))


def process_flow(args):
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

    # Load and extract flow channels (21:23)
    data = np.load(input_file)
    flow = data[:, :, 21:23]  # shape [H, W, 2]
    np.save(output_file, flow)
    elapsed = time.time() - start_time
    print(f"frame {sequence_id}_{frame_id} to flow done in {elapsed:.4f} seconds")


if __name__ == '__main__':
    print(f"Processing {len(tasks)} files using {NUM_WORKERS} workers...")
    with Pool(NUM_WORKERS) as pool:
        pool.map(process_flow, tasks)
    print("All done.")
