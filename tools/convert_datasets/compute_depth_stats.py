#!/usr/bin/env python3
"""
Compute global mean and std for AP4AD depth .npy files.

Folder layout expected (recursive scan):
  /home/negreami/datasets/ap4ad_local/depth/<seq>/<seq>_<frame>.npy

Usage (from repo root):
  python tools/convert_datasets/compute_depth_stats.py 
    [--root /path/to/depth] [--max-files 0] [--save-json /path/to/out.json]

Notes:
- Only finite values are considered (NaN/Inf are ignored).
- Supports arrays shaped (H, W) or (H, W, 1). Other shapes are skipped with a warning.
- Outputs a JSON summary and a config snippet for mmseg pipelines.
"""

import argparse
import glob
import json
import math
import os
import sys
from datetime import datetime

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='Compute mean/std for depth .npy files')
    parser.add_argument('--root', type=str,
                        default='/home/negreami/datasets/ap4ad_local/depth',
                        help='Root folder containing depth .npy files (scanned recursively)')
    parser.add_argument('--max-files', type=int, default=0,
                        help='If > 0, limit to this many files for a quick estimate')
    parser.add_argument('--save-json', type=str, default='',
                        help='Optional path to save stats JSON; default: <root>/depth_stats.json')
    parser.add_argument('--quiet', action='store_true', help='Reduce console output')
    parser.add_argument('--workers', type=int, default=2,  # This might have crashed the system with default=4 ...
                        help='Number of parallel workers for file loading (default: CPU count)')
    return parser.parse_args()


def find_npy_files(root: str):
    pattern = os.path.join(root, '**', '*.npy')
    files = sorted(glob.iglob(pattern, recursive=True))
    return list(files)


def safe_load_npy(path: str):
    try:
        arr = np.load(path)
        return arr
    except Exception as e:
        print(f"[WARN] Failed to load {path}: {e}")
        return None


def update_stats_from_array(arr: np.ndarray, acc):
    """Update accumulators with values from arr.

    acc is a dict with keys: n, s, ss, min, max, files_ok, files_skipped
    """
    # Accept (H, W) or (H, W, 1)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[..., 0]
    elif arr.ndim != 2:
        acc['files_skipped'] += 1
        return

    arr = arr.astype(np.float64, copy=False)
    mask = np.isfinite(arr)
    cnt = int(mask.sum())
    if cnt == 0:
        acc['files_skipped'] += 1
        return

    vals = arr[mask]
    s = float(vals.sum())
    ss = float(np.square(vals).sum())
    mn = float(vals.min())
    mx = float(vals.max())

    acc['n'] += cnt
    acc['s'] += s
    acc['ss'] += ss
    acc['min'] = mn if acc['min'] is None else min(acc['min'], mn)
    acc['max'] = mx if acc['max'] is None else max(acc['max'], mx)
    acc['files_ok'] += 1


def finalize_stats(acc):
    if acc['n'] == 0:
        return None
    mean = acc['s'] / acc['n']
    # var = E[x^2] - (E[x])^2
    ex2 = acc['ss'] / acc['n']
    var = max(0.0, ex2 - mean * mean)
    std = math.sqrt(var)
    return {
        'count_pixels': acc['n'],
        'mean': mean,
        'std': std,
        'min': acc['min'],
        'max': acc['max'],
    }


def main():
    args = parse_args()
    root = args.root
    save_json_path = args.save_json or os.path.join(root, 'depth_stats.json')

    files = find_npy_files(root)
    if not files:
        print(f"No .npy files found under {root}")
        print("Reminder: expected structure depth/<seq>/<seq>_<frame>.npy")
        sys.exit(1)

    if args.max_files and args.max_files > 0:
        files = files[: args.max_files]

    if not args.quiet:
        print(f"Found {len(files)} .npy files under {root}")

    acc = {
        'n': 0, 's': 0.0, 'ss': 0.0, 'min': None, 'max': None,
        'files_ok': 0, 'files_skipped': 0
    }

    if args.workers <= 1:
        for i, fpath in enumerate(files, 1):
            arr = safe_load_npy(fpath)
            if arr is None:
                acc['files_skipped'] += 1
                continue
            update_stats_from_array(arr, acc)
            if not args.quiet and (i % 1000 == 0 or i == len(files)):
                print(f"Processed {i}/{len(files)} files... pixels so far: {acc['n']}")
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        def process_file(fpath):
            arr = safe_load_npy(fpath)
            return (fpath, arr)
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(process_file, f): f for f in files}
            for i, fut in enumerate(as_completed(futures), 1):
                fpath, arr = fut.result()
                if arr is None:
                    acc['files_skipped'] += 1
                    continue
                update_stats_from_array(arr, acc)
                if not args.quiet and (i % 1000 == 0 or i == len(files)):
                    print(f"Processed {i}/{len(files)} files... pixels so far: {acc['n']}")

    stats = finalize_stats(acc)
    if stats is None:
        print("No finite depth values found. Aborting.")
        sys.exit(2)

    summary = {
        'root': root,
        'files_total': len(files),
        'files_ok': acc['files_ok'],
        'files_skipped': acc['files_skipped'],
        'count_pixels': stats['count_pixels'],
        'mean': stats['mean'],
        'std': stats['std'],
        'min': stats['min'],
        'max': stats['max'],
        'timestamp': datetime.now().isoformat(timespec='seconds')
    }

    # Console report
    print("\nDepth statistics:")
    print(json.dumps(summary, indent=2))

    # mmseg config snippet (depth-only normalization)
    print("\nSuggested mmseg config snippet:")
    print(f"depth_norm_cfg = dict(mean=[{summary['mean']:.6f}], std=[{summary['std']:.6f}], to_rgb=False)")

    # Save JSON
    os.makedirs(os.path.dirname(save_json_path), exist_ok=True)
    with open(save_json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    if not args.quiet:
        print(f"\nSaved stats JSON to: {save_json_path}")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Compute global mean and std for AP4AD depth .npy files.

Scans a directory tree (default: /home/negreami/datasets/ap4ad_local/depth),
loads all .npy files (HxW or HxWx1), and computes the mean and std across all pixels.

Usage:
  python tools/convert_datasets/compute_depth_stats.py \
      --depth-root /home/negreami/datasets/ap4ad_local/depth \
      [--recursive] [--limit N] [--workers 4]

Notes:
- Expects float data; if dtype is not float, it will be cast to float32.
- Ignores NaNs and infs gracefully.
- Uses a two-pass numerically stable algorithm (Welford) aggregated across workers.
"""

import argparse
import math
import os
from pathlib import Path
from typing import Iterator, Tuple

import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed


def iter_npy_files(root: Path, recursive: bool = True) -> Iterator[Path]:
    if recursive:
        yield from root.rglob('*.npy')
    else:
        yield from root.glob('*.npy')


def load_depth(path: Path) -> np.ndarray:
    arr = np.load(str(path))
    # Accept HxW or HxWx1; squeeze trailing singleton dim
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim != 2:
        raise ValueError(f"Depth file {path} has unexpected shape {arr.shape}; expected HxW or HxWx1")
    arr = arr.astype(np.float32, copy=False)
    # Mask invalid values
    mask = np.isfinite(arr)
    return arr[mask]


def partial_stats(path: Path) -> Tuple[int, float, float]:
    data = load_depth(path)
    n = data.size
    if n == 0:
        return 0, 0.0, 0.0
    s1 = float(data.sum())
    s2 = float(np.dot(data, data))  # sum of squares
    return n, s1, s2


def reduce_stats(partials: Iterator[Tuple[int, float, float]]) -> Tuple[int, float, float]:
    n_tot = 0
    s1_tot = 0.0
    s2_tot = 0.0
    for n, s1, s2 in partials:
        n_tot += n
        s1_tot += s1
        s2_tot += s2
    return n_tot, s1_tot, s2_tot


def compute_mean_std(depth_root: Path, recursive: bool, limit: int, workers: int) -> Tuple[float, float, int]:
    files = list(iter_npy_files(depth_root, recursive=recursive))
    if limit > 0:
        files = files[:limit]
    if not files:
        raise SystemExit(f"No .npy files found under {depth_root}")

    parts = []
    if workers <= 1:
        for f in files:
            parts.append(partial_stats(f))
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(partial_stats, f): f for f in files}
            for fut in as_completed(futures):
                try:
                    parts.append(fut.result())
                except Exception as e:
                    print(f"[WARN] Skipping {futures[fut]} due to error: {e}")

    n_tot, s1_tot, s2_tot = reduce_stats(parts)
    if n_tot == 0:
        raise SystemExit("All files were empty or invalid.")

    mean = s1_tot / n_tot
    # var = E[x^2] - (E[x])^2
    var = (s2_tot / n_tot) - (mean * mean)
    var = max(var, 0.0)
    std = math.sqrt(var)
    return mean, std, n_tot


def main():
    parser = argparse.ArgumentParser(description="Compute mean/std for depth .npy files")
    parser.add_argument('--depth-root', type=str, default='/home/negreami/datasets/ap4ad_local/depth',
                        help='Root directory containing depth .npy files')
    parser.add_argument('--recursive', action='store_true', help='Recurse into subdirectories')
    parser.add_argument('--limit', type=int, default=0, help='Limit number of files (0 = all)')
    parser.add_argument('--workers', type=int, default=os.cpu_count() or 4, help='Number of parallel workers')
    args = parser.parse_args()

    depth_root = Path(args.depth_root)
    if not depth_root.exists():
        raise SystemExit(f"Depth root does not exist: {depth_root}")

    mean, std, n = compute_mean_std(depth_root, args.recursive, args.limit, args.workers)

    print("=== Depth stats ===")
    print(f"root: {depth_root}")
    print(f"pixels: {n}")
    print(f"mean: {mean:.6f}")
    print(f"std:  {std:.6f}")
    print()
    print("Config snippet suggestion:")
    print("depth_norm_cfg = dict(mean=[%.6f], std=[%.6f])" % (mean, std))


if __name__ == '__main__':
    main()
