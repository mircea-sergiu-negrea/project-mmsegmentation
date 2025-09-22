# Script to convert .npy image arrays to JPEG RGB images, organized by sequence and frame
import os
import cv2
import numpy as np
import glob


# Path to the directory containing the .npy input files
path = '/home/negreami/datasets/ap4ad/AP-for-AD_data'

# Find all .npy files matching the pattern inputs_*_*.npy
all_input_files = sorted(glob.glob(os.path.join(path, 'inputs_*_*.npy')))

# Extract unique sequence IDs from filenames
sequence_ids = set()
for input_file in all_input_files:
    # Example filename: inputs_0001_0123.npy
    # input_file.split('/')[-1] -> 'inputs_0001_0123.npy'
    # .split('_')[1] -> '0001' (sequence_id)
    sequence_id = input_file.split('/')[-1].split('_')[1]
    sequence_ids.add(sequence_id)

# Extract unique frame IDs from filenames
frame_ids = set()
for input_file in all_input_files:
    # Example filename: inputs_0001_0123.npy
    # input_file.split('/')[-1] -> 'inputs_0001_0123.npy'
    # .split('_')[2] -> '0123.npy'
    # .split('.')[0] -> '0123' (frame_id)
    frame_id = input_file.split('/')[-1].split('_')[2].split('.')[0]
    frame_ids.add(frame_id)

# Directory to save the output RGB JPEG images
save_dir = '/home/negreami/datasets/ap4ad/only_rgb'
os.makedirs(save_dir, exist_ok=True)

# Loop over each sequence and frame to process and save images
for sequence_id in sequence_ids:
    sequence_dir = os.path.join(save_dir, sequence_id)
    os.makedirs(sequence_dir, exist_ok=True)
    import time
    for frame_id in frame_ids:
        # Construct the expected input file path
        input_file = os.path.join(path, f'inputs_{sequence_id}_{frame_id}.npy')
        output_file = os.path.join(sequence_dir, f'{sequence_id}_{frame_id}.jpg')
        start_time = time.time()
        if os.path.exists(output_file):
            elapsed = time.time() - start_time
            print(f"Skipping frame {sequence_id}_{frame_id}, already exists. (skipped in {elapsed:.4f} seconds)")
            continue
        if not os.path.exists(input_file):
            elapsed = time.time() - start_time
            print(f'Missing input file: {input_file} (checked in {elapsed:.4f} seconds)')
            continue

        # Load the .npy file (assumed to be an image with at least 3 channels)
        input_data = np.load(input_file)
        # Extract the first 3 channels as RGB
        rgb = input_data[:, :, :3]  # Assuming RGB is the first 3 channels
        # Scale values from [-0.5, 0.5] to [0, 255]
        rgb = (rgb + 0.5) * 255.0
        # (Optional) Get unique RGB values for inspection
        values = np.unique(rgb.reshape(-1, rgb.shape[2]), axis=0)
        # Convert to uint8 for saving as image
        rgb = rgb.astype(np.uint8)
        # Save the RGB image as JPEG
        cv2.imwrite(output_file, rgb)
        elapsed = time.time() - start_time
        # Keep track of progress in screen terminal
        print(f"frame {sequence_id}_{frame_id} to rgb done in {elapsed:.4f} seconds")
    print(f"sequence {sequence_id} completed with {len(frame_ids)} / 127 frames saved.")