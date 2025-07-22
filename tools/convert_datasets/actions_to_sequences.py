import os  # For file and directory operations
import shutil  # For copying files

# all .npy files (both input and action files)
SRC_DIR = '/home/negreami/datasets/ap4ad/AP-for-AD_data'
# Destination for actions organized by sequence
DST_DIR = '/home/negreami/datasets/ap4ad_local/actions'

os.makedirs(DST_DIR, exist_ok=True)

for root, _, files in os.walk(SRC_DIR):
    for fname in files:
        # Only process 'action_' files
        if fname.startswith('action_') and fname.endswith('.npy'):
            # Extract sequence
            # 'action_0111_0126.npy' -> seq = '0111'
            seq = fname.split('_')[1]
            # Build destination subfolder path
            # '/home/negreami/datasets/ap4ad_local/actions/0111'
            seq_dst_dir = os.path.join(DST_DIR, seq)
            # Create the sequence subfolder if it doesn't exist
            os.makedirs(seq_dst_dir, exist_ok=True)
            # Full path to the source file
            # Example: '/home/negreami/datasets/ap4ad/AP-for-AD-data/action_0111_0126.npy'
            src_path = os.path.join(root, fname)
            # Remove 'action_' prefix for the destination filename
            stripped_fname = fname[len('action_'):]
            # Full path to the destination file
            # Example: '/home/negreami/datasets/ap4ad_local/actions/0111/0111_0126.npy'
            dst_path = os.path.join(seq_dst_dir, stripped_fname)
            # Copy the file from source to destination
            shutil.copy2(src_path, dst_path)
            print(f'Copied {src_path} -> {dst_path}')

# Print when done
print('Done copying action files into sequence subfolders.')
