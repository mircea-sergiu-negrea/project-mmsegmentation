"""RGB+Depth AP4AD sanity inspection script.

Usage (default config path embedded):
    python demo/inspect_ap4ad_sample_depth.py \
        --config configs/encoder_decoder_action/ap4ad_rgb-d.py \
        --device cuda:0

What it does:
 1. Loads the RGB+Depth config.
 2. Builds the train dataset (expects modalities ['rgb','depth']).
 3. Prints dataset length and one random sample (shapes + per-channel stats).
 4. Builds a DataLoader, fetches one batch, prints tensor shapes.
 5. Builds the model and runs a single forward_train to print loss dict.

Safe to run before launching a full training job to confirm nothing breaks
with 4-channel inputs.
"""

import argparse
import random
import copy
import torch
import contextlib
from mmcv import Config
from mmseg.datasets import build_dataset, build_dataloader
from mmseg.models import build_segmentor
import mmseg.datasets.ap4ad  # register AP4ADDataset


def per_channel_stats(t):
    """Return list of (mean,std,min,max) per channel for a CHW or HWC tensor/ndarray."""
    import numpy as np
    if torch.is_tensor(t):
        arr = t.detach().cpu().float().numpy()
    else:
        arr = t
    if arr.ndim == 3 and arr.shape[0] in (3,4,5,10) and arr.shape[0] < arr.shape[2]:
        # likely CHW
        ch_axis = 0
    elif arr.ndim == 3:
        # assume HWC
        arr = arr.transpose(2,0,1)
        ch_axis = 0
    else:
        return []
    stats = []
    for c in range(arr.shape[ch_axis]):
        ch = arr[c]
        stats.append((float(ch.mean()), float(ch.std()), float(ch.min()), float(ch.max())))
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/encoder_decoder_action/ap4ad_rgb-d.py')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--no-cudnn', action='store_true', help='Disable cuDNN for this run')
    parser.add_argument('--cpu', action='store_true', help='Force CPU even if CUDA is available')
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)

    # --- Build a CHECK dataset with LoadDepthFromFile injected to sanity-check depth ---
    cfg_check = copy.deepcopy(cfg)
    train_cfg_check = cfg_check.data['train']
    pipe = train_cfg_check['pipeline']
    # Insert LoadDepthFromFile after LoadImageFromFile if not present
    def has_step(p, t):
        return any(isinstance(s, dict) and s.get('type') == t for s in p)
    if not has_step(pipe, 'LoadDepthFromFile'):
        inserted = False
        for i, step in enumerate(pipe):
            if isinstance(step, dict) and step.get('type') == 'LoadImageFromFile':
                pipe.insert(i+1, dict(type='LoadDepthFromFile'))
                inserted = True
                break
        if not inserted:
            pipe.insert(0, dict(type='LoadDepthFromFile'))
    # Ensure Collect includes 'depth' so we can inspect it at sample level
    for step in pipe:
        if isinstance(step, dict) and step.get('type') == 'Collect':
            keys = step.get('keys', [])
            if 'depth' not in keys:
                step['keys'] = list(keys) + ['depth']
            # include meta for depth path if available
            meta_keys = step.get('meta_keys', (
                'filename', 'ori_filename', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'img_norm_cfg'))
            # convert to list to append safely
            if not isinstance(meta_keys, (list, tuple)):
                meta_keys = [meta_keys]
            meta_keys = list(meta_keys)
            for mk in ['depth_filename', 'depth_shape']:
                if mk not in meta_keys:
                    meta_keys.append(mk)
            step['meta_keys'] = tuple(meta_keys)
            break

    dataset_check = build_dataset(train_cfg_check)
    print(f"[CHECK] Dataset length: {len(dataset_check)}")
    idx = random.randrange(len(dataset_check))
    sample_check = dataset_check[idx]

    # Pull out items
    img_chk = sample_check['img']
    action_chk = sample_check['action']
    depth_chk = sample_check.get('depth', None)

    print(f"Random sample index: {idx}")
    print(f"Sample img tensor shape: {getattr(img_chk,'shape',None)} dtype: {getattr(img_chk,'dtype',None)}")
    stats = per_channel_stats(img_chk)
    if stats:
        for i,(m,s,mi,ma) in enumerate(stats):
            print(f"  Channel {i}: mean={m:.3f} std={s:.3f} min={mi:.3f} max={ma:.3f}")
    print(f"Action shape: {getattr(action_chk,'shape',None)} value: {action_chk}")
    print("Note: 'Depth channel matches file' compares the normalized img channel vs raw file, so False is expected when normalization is applied.")

    # Validate depth presence and consistency against file and img 4th channel
    try:
        import numpy as np
        # Get meta for computing expected path
        meta = sample_check['img_metas'].data
        if isinstance(meta, list) and meta:
            meta = meta[0]
        ori_filename = meta.get('ori_filename') if isinstance(meta, dict) else None
        # Compute depth path from dataset fields
        ds = dataset_check
        # If dataset provides properties
        depth_root = getattr(ds, 'depth_dir', None)
        data_root = getattr(ds, 'data_root', None)
        depth_base = None
        if depth_root is not None and data_root is not None:
            import os.path as osp
            depth_base = osp.join(data_root, depth_root)
            if ori_filename is not None:
                seq = ori_filename.split('/')[0]
                name = ori_filename.split('/')[-1].rsplit('.', 1)[0]
                depth_expected_path = osp.join(depth_base, seq, name + '.npy')
            else:
                depth_expected_path = None
        else:
            depth_expected_path = None

        if depth_expected_path is not None:
            depth_disk = np.load(depth_expected_path)
            if depth_disk.ndim == 2:
                depth_disk = depth_disk[..., None]
            # Compare with img 4th channel when modalities include depth
            if torch.is_tensor(img_chk):
                arr = img_chk.detach().cpu().numpy()
                # CHW expected after formatting
                if arr.ndim == 3 and arr.shape[0] >= 4:
                    depth_from_img = arr[3]
                    # transpose depth_disk HWC->HW if needed
                    if depth_disk.ndim == 3 and depth_disk.shape[2] == 1:
                        depth_disk_hw = depth_disk[..., 0]
                    else:
                        depth_disk_hw = depth_disk
                    same = np.allclose(depth_from_img, depth_disk_hw, equal_nan=True)
                    print(f"Depth channel matches file: {bool(same)}")
            # If depth key was collected, compare as well
            if depth_chk is not None:
                if torch.is_tensor(depth_chk):
                    depth_arr = depth_chk.detach().cpu().numpy()
                else:
                    depth_arr = np.asarray(depth_chk)
                if depth_arr.ndim == 3 and depth_arr.shape[2] == 1:
                    depth_arr_hw = depth_arr[..., 0]
                elif depth_arr.ndim == 2:
                    depth_arr_hw = depth_arr
                else:
                    depth_arr_hw = depth_arr
                if depth_disk.ndim == 3 and depth_disk.shape[2] == 1:
                    depth_disk_hw = depth_disk[..., 0]
                else:
                    depth_disk_hw = depth_disk
                same2 = np.allclose(depth_arr_hw, depth_disk_hw, equal_nan=True)
                print(f"Loaded depth (pipeline) matches file: {bool(same2)}")
                print(f"Depth array shape from pipeline: {depth_arr.shape}, dtype: {depth_arr.dtype}")
    except Exception as e:
        print(f"[WARN] Depth consistency check failed with error: {e}")

    # --- Build the ORIGINAL dataset for downstream DataLoader/model smoke test ---
    train_cfg = cfg.data['train']
    dataset = build_dataset(train_cfg)
    print(f"Dataset length (original pipeline): {len(dataset)}")

    # Build DataLoader
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=args.batch_size,
        workers_per_gpu=args.num_workers,
        dist=False,
        shuffle=False,
        persistent_workers=False,
        drop_last=False)

    first_batch = next(iter(data_loader))
    # After collate: 'img' is a Tensor (B,C,H,W); 'action' is Tensor (B,2)
    batch_imgs = first_batch['img']
    batch_actions = first_batch['action']
    # img_metas remains a DataContainer -> extract underlying list
    batch_img_metas = first_batch['img_metas'].data[0]
    print(f"Batch img tensor shape: {batch_imgs.shape} dtype: {batch_imgs.dtype}")
    print(f"Batch action tensor shape: {batch_actions.shape} dtype: {batch_actions.dtype}")

    # Build model
    # cfg.model already includes test_cfg; avoid passing duplicate test_cfg arg
    model = build_segmentor(cfg.model)
    # Device/cuDNN setup
    device = 'cpu' if args.cpu else args.device
    if device.startswith('cuda') and not torch.cuda.is_available():
        print('[WARN] CUDA not available, falling back to CPU')
        device = 'cpu'
    if device.startswith('cuda') and args.no_cudnn:
        try:
            import torch.backends.cudnn as cudnn
            cudnn.enabled = False
            cudnn.benchmark = False
            cudnn.deterministic = True
            print('[INFO] cuDNN disabled by --no-cudnn')
        except Exception:
            pass
    model.to(device)
    model.train()

    # Forward train once
    imgs_device = batch_imgs.to(device).contiguous()
    # Ensure action tensor is float32 for the model (convert if double)
    actions_device = batch_actions.to(device).float()
    # Forward with required img_metas argument
    def run_forward():
        losses_inner = model.forward_train(img=imgs_device, img_metas=batch_img_metas, action=actions_device)
        print('Loss keys:', list(losses_inner.keys()))
        if isinstance(losses_inner, dict):
            total = 0.0
            for k, v in losses_inner.items():
                if torch.is_tensor(v):
                    val = float(v.mean())
                elif isinstance(v, (list, tuple)):
                    val = float(sum(x.mean() for x in v) / len(v))
                else:
                    continue
                print(f"  {k}: {val:.6f}")
                total += val
            print(f"Total (approx) loss: {total:.6f}")

    try:
        run_forward()
    except RuntimeError as e:
        msg = str(e)
        if 'Unable to find a valid cuDNN algorithm' in msg and device.startswith('cuda'):
            print('[WARN] cuDNN algo selection failed; retrying with cuDNN disabled...')
            with contextlib.suppress(Exception):
                import torch.backends.cudnn as cudnn
                cudnn.enabled = False
                cudnn.benchmark = False
                cudnn.deterministic = True
            # retry on GPU without cuDNN
            try:
                run_forward()
            except RuntimeError as e2:
                print('[WARN] Retry without cuDNN also failed; moving to CPU and retrying once...')
                model.cpu()
                imgs_cpu = batch_imgs.contiguous().cpu()
                actions_cpu = batch_actions.float().cpu()
                with torch.no_grad():
                    losses_cpu = model.forward_train(img=imgs_cpu, img_metas=batch_img_metas, action=actions_cpu)
                    print('Loss keys (CPU):', list(losses_cpu.keys()))
        else:
            raise

    print('\n[OK] Sanity pass complete.')


if __name__ == '__main__':
    main()
