import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import BaseModule
from ..builder import HEADS


@HEADS.register_module()
class ActionHead(BaseModule):
	"""
	Regression head for throttle (0-1, sigmoid) and steer (-1 to 1, tanh).
	Architecture:
	- Flatten pooled features
	- FC(2048) -> ReLU -> FC(1024) -> ReLU
	- Two output heads:
		- throttle: FC(1024, 1) + sigmoid
		- steer: FC(1024, 1) + tanh
	"""

	def __init__(self, in_channels, hidden_dim1=2048, hidden_dim2=1024, init_cfg=None, weight_throttle_loss=50.0):
		super().__init__(init_cfg=init_cfg)
		self.fc1 = nn.Linear(in_channels, hidden_dim1)
		self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
		self.throttle_out = nn.Linear(hidden_dim2, 1)
		self.steer_out = nn.Linear(hidden_dim2, 1)
		self.loss_fn = nn.MSELoss(reduction='mean')
		self.weight_throttle_loss = weight_throttle_loss

	def forward(self, feats):
		x = feats[-1]
		if x.dim() == 4:
			x = F.adaptive_avg_pool2d(x, 1).view(x.size(0), -1)
		elif x.dim() == 2:
			x = x
		else:
			raise ValueError(f'Unexpected feature tensor shape: {x.shape}')
		x = F.relu(self.fc1(x))
		x = self.fc2(x)  # linear activation
		throttle = torch.sigmoid(self.throttle_out(x))  # (B, 1)
		steer = torch.tanh(self.steer_out(x))           # (B, 1)
		return torch.cat([throttle, steer], dim=1)      # (B, 2)

	def loss(self, pred, target):
		if isinstance(target, (list, tuple)):
			target = target[0]
		t = target.float().to(pred.device)
		l0 = self.weight_throttle_loss * self.loss_fn(pred[:, 0], t[:, 0])  # throttle loss
		l1 = self.loss_fn(pred[:, 1], t[:, 1])  # steer loss
		return dict(loss_throttle=l0, loss_steer=l1)

