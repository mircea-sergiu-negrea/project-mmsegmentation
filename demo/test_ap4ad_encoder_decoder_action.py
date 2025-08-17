"""
Test EncoderDecoderAction + ActionHead using the AP4AD config file.
Loads the config, builds the model, loads a sample from the AP4AD (rgb) dataset,
and runs a forward pass to print predictions and losses.
"""
import numpy as np
import torch
from mmcv import Config
from mmseg.models.segmentors.encoder_decoder_action import EncoderDecoderAction
from mmseg.models.decode_heads import action_head  # This ensures ActionHead is registered
from mmseg.datasets.ap4ad import AP4ADDataset

# Load config
cfg = Config.fromfile('/home/negreami/project/mmsegmentation/configs/encoder_decoder_action/ap4ad_rgb.py')

# Instantiate dataset using config
train_cfg = cfg.data['train']
dataset = AP4ADDataset(
    data_root=train_cfg['data_root'],
    img_dir=train_cfg['img_dir'],
    action_dir=train_cfg['action_dir'],
    pipeline=train_cfg['pipeline']
)

# Instantiate model using config
model_cfg = dict(cfg.model)  # make a copy
model_cfg.pop('type', None)  # remove 'type' if present
model = EncoderDecoderAction(**model_cfg)
model.eval()

# --- Load a real sample from the dataset ---
random_datapoint = np.random.randint(0, len(dataset))
sample = dataset[random_datapoint]

img = sample['img']
if not isinstance(img, torch.Tensor):
    img = torch.from_numpy(img)
img = img.float()
if img.shape[-1] == 3:
    img = img.permute(2, 0, 1)
if img.ndim == 3:
    img = img.unsqueeze(0)

action = sample['gt_semantic_seg']
if not isinstance(action, torch.Tensor):
    action = torch.from_numpy(action)
action = action.float()
if action.ndim == 1:
    action = action.unsqueeze(0)

img_metas = [{
    'img_shape': img.shape,
    'ori_shape': img.shape,
    'pad_shape': img.shape,
    'filename': None,
    'scale_factor': 1.0,
    'flip': False,
    'flip_direction': None
}]

with torch.no_grad():
    pred = model.decode_head(model.extract_feat(img))
    losses = model.forward_train(img, img_metas, action)
print("\nPredicted actions:", pred)
print("Ground truth actions:", action)
print("Loss dict:", losses, "\n")