# script that counts the number of .npy files in each subfolder of a given directory
# checks if there are 112 subfolders and if each has 127 .npy files

import os

def check_only_rgb_structure(directory):
    subfolders = [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]
    print()
    print(f"Found {len(subfolders)} / 112 folders in '{directory}'")
    print(f"Out of the {len(subfolders)} folders...")
    all_ok = True
    for folder in subfolders:
        folder_path = os.path.join(directory, folder)
        npys = [f for f in os.listdir(folder_path) if f.lower().endswith('.npy')]
        if len(npys) != 127:
            print(f"Sequence '{folder}' has only {len(npys)} / 127 frames.")
            all_ok = False
    if all_ok:
        print("All folders have exactly 127 .npy files.")
    else:
        print("Some folders are missing some .npy files.")

if __name__ == "__main__":
    check_only_rgb_structure("/home/negreami/datasets/ap4ad_local/depth")  # Change this to the directory you want to check