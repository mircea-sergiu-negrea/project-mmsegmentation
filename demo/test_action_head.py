"""
This tests the ActionHead in isolation.
It loads a sample from the dataset, passes it through some backbone (ResNet),
then feeds the backbone features directly to ActionHead and prints the predicted actions and loss.
"""
# Test script for ActionHead
import numpy as np
import torch
from mmseg.models.decode_heads.action_head import ActionHead
from mmseg.datasets.ap4ad import AP4ADDataset
from mmseg.models.builder import build_backbone

# 1. Instantiate dataset
dataset = AP4ADDataset(
    data_root='/home/negreami/datasets/ap4ad_local',
    img_dir='only_rgb',
    action_dir='actions',
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='Collect', keys=['img', 'gt_semantic_seg'])
    ]
)

# 2. Get a random sample
random_datapoint = np.random.randint(0, len(dataset))
sample = dataset[random_datapoint]
img = sample['img']  # shape: (C, H, W)
action = sample['gt_semantic_seg']  # shape: (2,)

# 3. Convert image to tensor and float
if not isinstance(img, torch.Tensor):
    img = torch.from_numpy(img)
img = img.float()
if img.shape[-1] == 3:  # if channels are last
    img = img.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
img = img.unsqueeze(0)  # add batch dimension

# 4. Build backbone (ResNet-18 for demo)
backbone_cfg = dict(
    type='ResNet',
    depth=18,
    in_channels=3,
    norm_cfg=dict(type='BN', requires_grad=True),
    norm_eval=False,
    style='pytorch'
)
backbone = build_backbone(backbone_cfg)
backbone.eval()

# 5. Forward pass through backbone
with torch.no_grad():
    feats = backbone(img)

# 6. Instantiate ActionHead with correct in_channels
in_channels = feats[-1].shape[1]
head = ActionHead(in_channels=in_channels)

# 7. Forward pass through ActionHead
with torch.no_grad():
    pred = head(feats)
print("\nPredicted actions:", pred)
print("Ground truth actions:", action)

# 8. Compute loss (match batch size)
target = torch.tensor(action).float().unsqueeze(0)  # shape [1, 2]
losses = head.loss(pred, target)
print("Loss dict:", losses, "\n")