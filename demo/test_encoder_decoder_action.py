"""
This tests the full EncoderDecoderAction + ActionHead.
It loads a sample from the dataset, passes it through our model EncoderDecoderAction,
then through our ActionHead, and then prints the loss dict.
"""
import numpy as np
import torch
from mmseg.models.segmentors.encoder_decoder_action import EncoderDecoderAction
from mmseg.models.decode_heads.action_head import ActionHead
from mmseg.datasets.ap4ad import AP4ADDataset

# Instantiate dataset
dataset = AP4ADDataset(
    data_root='/home/negreami/datasets/ap4ad_local',
    img_dir='only_rgb',
    action_dir='actions',
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='Collect', keys=['img', 'gt_semantic_seg'])
    ]
)

# Dummy config for model
model_cfg = dict(
    backbone=dict(
        type='ResNet',
        depth=18,
        in_channels=3,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        style='pytorch'
    ),
    decode_head=dict(
        type='ActionHead',
        in_channels=512,  # match backbone output channels
    ),
    train_cfg=None,
    test_cfg=dict(mode='whole')
)

# Instantiate model
model = EncoderDecoderAction(**model_cfg)
model.eval()

# --- Load a real sample from the dataset ---
# 1. Pick a random datapoint from the dataset
random_datapoint = np.random.randint(0, len(dataset))
sample = dataset[random_datapoint]

# 2. Get the image
img = sample['img']
# If the image is a numpy array, convert to torch tensor
if not isinstance(img, torch.Tensor):
    img = torch.from_numpy(img)
# Ensure image is float (required by model)
img = img.float()
# If channels are last (H, W, C), permute to (C, H, W)
if img.shape[-1] == 3:
    img = img.permute(2, 0, 1)
# Add batch dimension to get (B, C, H, W)
if img.ndim == 3:
    img = img.unsqueeze(0)

# 3. Get the action (target) and ensure it's a float tensor with batch dim
action = sample['gt_semantic_seg']  # shape: (2,)
if not isinstance(action, torch.Tensor):
    action = torch.from_numpy(action)
action = action.float()
if action.ndim == 1:
    action = action.unsqueeze(0)

# 4. Build img_metas dict (required by mmsegmentation model)
#    Contains metadata about the image for resizing, flipping, etc.
img_metas = [{
    'img_shape': img.shape,
    'ori_shape': img.shape,
    'pad_shape': img.shape,
    'filename': None,
    'scale_factor': 1.0,
    'flip': False,
    'flip_direction': None
}]

# Forward train
with torch.no_grad():
    # Get model prediction (output of ActionHead)
    pred = model.decode_head(model.extract_feat(img))
    losses = model.forward_train(img, img_metas, action)
print("\nPredicted actions:", pred)
print("Ground truth actions:", action)
print("Loss dict:", losses, "\n")
