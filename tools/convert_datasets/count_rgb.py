# script that counts the number of .jpg files in each subfolder of a given directory
# checks if there are 112 subfolders and if each has 127 .jpg files

import os

def check_only_rgb_structure(directory):
    subfolders = [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]
    print()
    print(f"Found {len(subfolders)} / 112 folders in '{directory}'")
    print(f"Out of the {len(subfolders)} folders...")
    all_ok = True
    for folder in subfolders:
        folder_path = os.path.join(directory, folder)
        jpgs = [f for f in os.listdir(folder_path) if f.lower().endswith('.jpg')]
        if len(jpgs) != 127:
            print(f"Sequence '{folder}' has only {len(jpgs)} / 127 frames.")
            all_ok = False
    if all_ok:
        print("All folders have exactly 127 .jpg files.")
    else:
        print("Some folders are missing some .jpg files.")

if __name__ == "__main__":
    check_only_rgb_structure("/home/negreami/datasets/ap4ad_local/only_rgb")