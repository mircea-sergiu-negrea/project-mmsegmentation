import matplotlib
matplotlib.use('Agg')
import json
import matplotlib.pyplot as plt
import os
import sys
import argparse
import numpy as np
import glob
from collections import deque


def parse_loss_entry(entry):
    # Try multiple formats for 'loss'
    if 'loss' in entry:
        lf = entry['loss']
        if isinstance(lf, (int, float)):
            return float(lf)
        if isinstance(lf, dict):
            # prefer a 'loss' key inside, else sum numeric components
            if 'loss' in lf and isinstance(lf['loss'], (int, float)):
                return float(lf['loss'])
            s = 0.0
            found = False
            for v in lf.values():
                if isinstance(v, (int, float)):
                    s += float(v)
                    found = True
            if found:
                return s
    # fallback: sum top-level keys that start with 'loss'
    s = 0.0
    found = False
    for k, v in entry.items():
        if isinstance(k, str) and k.startswith('loss') and isinstance(v, (int, float)):
            s += float(v)
            found = True
    if found:
        return s
    return None


def extract_throttle_steer(entry):
    # Try several places for throttle/steer losses
    thr = None
    st = None
    # 1) inside 'loss' dict
    lf = entry.get('loss')
    if isinstance(lf, dict):
        thr = lf.get('loss_throttle') if isinstance(lf.get('loss_throttle'), (int, float)) else lf.get('throttle')
        st = lf.get('loss_steer') if isinstance(lf.get('loss_steer'), (int, float)) else lf.get('steer')
    # 2) top-level keys
    if thr is None and isinstance(entry.get('loss_throttle'), (int, float)):
        thr = float(entry.get('loss_throttle'))
    if st is None and isinstance(entry.get('loss_steer'), (int, float)):
        st = float(entry.get('loss_steer'))
    # 3) numeric coercion if numpy types
    try:
        if thr is not None:
            thr = float(thr)
        if st is not None:
            st = float(st)
    except Exception:
        pass
    return thr, st


def main(log_path=None, out_png=None, data_root=None, npy_glob='**/*.npy', max_samples=200, smooth_window=None):
    if log_path is None:
        log_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), '..', 'work_dirs', 'ap4ad_rgb_test', 'None.log.json'))
    if out_png is None:
        out_png = os.path.join(os.path.dirname(__file__), 'loss_curve.png')

    if not os.path.exists(log_path):
        print(f'ERROR: log file not found: {log_path}', file=sys.stderr)
        return 2

    losses = []
    iters = []
    throttle_vals = []
    steer_vals = []
    line_no = 0
    with open(log_path, 'r') as f:
        for line in f:
            line_no += 1
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except Exception as e:
                # skip non-json lines but report occasionally
                if line_no <= 5:
                    print(f'WARN: failed to parse json on line {line_no}: {e}', file=sys.stderr)
                continue
            if entry.get('mode') != 'train':
                continue
            loss_val = parse_loss_entry(entry)
            thr, st = extract_throttle_steer(entry)
            if thr is None and st is None and loss_val is None:
                # nothing useful
                continue
            iter_idx = entry.get('iter') if isinstance(entry.get('iter'), int) else len(iters)
            iters.append(iter_idx)
            losses.append(loss_val if loss_val is not None else float('nan'))
            throttle_vals.append(thr if thr is not None else float('nan'))
            steer_vals.append(st if st is not None else float('nan'))

    if not losses:
        print('ERROR: no training loss entries found in log.', file=sys.stderr)
        return 3

    # Sort by iteration if not already
    paired = sorted(zip(iters, losses), key=lambda x: x[0])
    iters, losses = zip(*paired)

    # helper: moving average that ignores NaNs
    def moving_average(arr, window):
        a = np.array(arr, dtype=float)
        n = len(a)
        if n == 0:
            return a
        if window <= 1:
            return a
        res = np.full(n, np.nan)
        # simple cumulative approach to handle NaNs
        s = 0.0
        cnt = 0
        q = deque()
        for i in range(n):
            v = a[i]
            q.append(v)
            if not np.isnan(v):
                s += v
                cnt += 1
            if len(q) > window:
                old = q.popleft()
                if not np.isnan(old):
                    s -= old
                    cnt -= 1
            if cnt > 0:
                res[i] = s / cnt
        return res

    # determine smoothing window adaptively
    def smoothing_window(n_points):
        # aim for a window ~1-2% of points, min 5, max 200
        if n_points <= 10:
            return 1
        w = max(5, int(max(1, n_points * 0.02)))
        return min(w, 200)

    fig, ax = plt.subplots(figsize=(12, 5))
    plotted = 0
    n_points = len(iters)
    win = smoothing_window(n_points)

    def safe_array(vals):
        return np.array([v if (v is not None and not (isinstance(v, float) and np.isnan(v))) else np.nan for v in vals], dtype=float)

    throttle_arr = safe_array(throttle_vals)
    steer_arr = safe_array(steer_vals)
    loss_arr = safe_array(losses)

    if np.any(~np.isnan(throttle_arr)):
        ma_thr = moving_average(throttle_arr, win)
        ax.plot(iters, throttle_arr, color='C0', alpha=0.12, label='_nolegend_')
        ax.plot(iters, ma_thr, color='C0', alpha=1.0, linewidth=2.0, label=f'loss_throttle (ma{win})')
        plotted += 1
    if np.any(~np.isnan(steer_arr)):
        ma_st = moving_average(steer_arr, win)
        ax.plot(iters, steer_arr, color='C1', alpha=0.12, label='_nolegend_')
        ax.plot(iters, ma_st, color='C1', alpha=1.0, linewidth=2.0, label=f'loss_steer (ma{win})')
        plotted += 1
    if plotted == 0:
        ma_loss = moving_average(loss_arr, win)
        ax.plot(iters, loss_arr, color='C2', alpha=0.12, label='_nolegend_')
        ax.plot(iters, ma_loss, color='C2', alpha=1.0, linewidth=2.0, label=f'loss (ma{win})')

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss Curve')
    ax.legend()
    ax.grid(True)
    fig.tight_layout()

    try:
        out_dir = os.path.dirname(out_png)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        plt.savefig(out_png)
    except Exception as e:
        print(f'ERROR: failed to save plot to {out_png}: {e}', file=sys.stderr)
        return 4

    print(f'Saved loss curve to: {out_png}  (entries={len(losses)})')

    # Minimal second loss-only plot (faded raw + vibrant moving mean) saved to loss_hist.png
    hist_png = out_png
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    plotted2 = 0
    if np.any(~np.isnan(throttle_arr)):
        ma_thr2 = moving_average(throttle_arr, win)
        ax2.plot(iters, throttle_arr, color='C0', alpha=0.12)
        ax2.plot(iters, ma_thr2, color='C0', alpha=1.0, linewidth=2.0, label=f'loss_throttle (ma{win})')
        plotted2 += 1
    if np.any(~np.isnan(steer_arr)):
        ma_st2 = moving_average(steer_arr, win)
        ax2.plot(iters, steer_arr, color='C1', alpha=0.12)
        ax2.plot(iters, ma_st2, color='C1', alpha=1.0, linewidth=2.0, label=f'loss_steer (ma{win})')
        plotted2 += 1
    if plotted2 == 0:
        ma_loss2 = moving_average(loss_arr, win)
        ax2.plot(iters, loss_arr, color='C2', alpha=0.12)
        ax2.plot(iters, ma_loss2, color='C2', alpha=1.0, linewidth=2.0, label=f'loss (ma{win})')

    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Loss')
    ax2.set_title('Training Loss Curve')
    ax2.legend()
    ax2.grid(True)
    fig2.tight_layout()
    try:
        plt.savefig(hist_png)
        print(f'Saved loss curve to: {hist_png}')
    except Exception as e:
        print(f'ERROR: failed to save loss curve to {hist_png}: {e}', file=sys.stderr)

    return 0


if __name__ == '__main__':
    lp = None
    op = None
    # optional args: [log_path] [out_png]
    if len(sys.argv) >= 2:
        lp = sys.argv[1]
    if len(sys.argv) >= 3:
        op = sys.argv[2]
    # optional third arg: data_root (path to action .npy files)
    default_data_root = '/home/negreami/datasets/ap4ad_local/actions/'
    data_root = None
    if len(sys.argv) >= 4:
        data_root = sys.argv[3]
    else:
        data_root = default_data_root

    rc = main(lp, op, data_root)
    sys.exit(rc)
