# Demo script to inspect a sample from the Cityscapes dataset using mmsegmentation
import os
from mmseg.datasets import build_dataset
from mmcv import Config

# Path to Cityscapes config file
config_path = "/home/negreami/project/mmsegmentation/configs/_base_/datasets/cityscapes.py"

cfg = Config.fromfile(config_path)

# Build the dataset (train set by default)
dataset = build_dataset(cfg.data.train)

# Inspect the first sample
sample = dataset[0]

print()
print('Sample keys:', sample.keys())
print('Image shape:', sample['img'].data.shape)
if 'gt_semantic_seg' in sample:
    print('Label shape:', sample['gt_semantic_seg'].data.shape)
print()
print('Metadata:', sample.get('img_metas', {}))
print()
