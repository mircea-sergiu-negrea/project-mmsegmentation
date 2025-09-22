#!/usr/bin/env python3
"""
Extract the 'seg' modality from AP4AD .npy input files and save per-frame .npy files.

This script extracts channels 63:73 (10 segmentation channels) and saves them under
/home/negreami/datasets/ap4ad/seg/<seq>/<seq>_<frame>.npy.

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
SAVE_DIR = '/home/negreami/datasets/ap4ad/seg'
os.makedirs(SAVE_DIR, exist_ok=True)

all_input_files = sorted(glob.glob(os.path.join(INPUT_PATH, 'inputs_*_*.npy')))

tasks = []
for input_file in all_input_files:
    filename = os.path.basename(input_file)
    parts = filename.split('_')
    sequence_id = parts[1]
    frame_id = parts[2].split('.')[0]
    tasks.append((sequence_id, frame_id))


def process_seg(args):
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

    # Load and extract segmentation channels (63:73)
    data = np.load(input_file)
    seg = data[:, :, 63:73]  # shape [H, W, 10]
    np.save(output_file, seg)
    elapsed = time.time() - start_time
    print(f"frame {sequence_id}_{frame_id} to seg done in {elapsed:.4f} seconds")


if __name__ == '__main__':
    print(f"Processing {len(tasks)} files using {NUM_WORKERS} workers...")
    with Pool(NUM_WORKERS) as pool:
        pool.map(process_seg, tasks)
    print("All done.")
