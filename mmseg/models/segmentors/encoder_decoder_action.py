# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.core import add_prefix
from mmseg.ops import resize
from .. import builder
from ..builder import SEGMENTORS
from .base import BaseSegmentor
from .encoder_decoder import EncoderDecoder

@SEGMENTORS.register_module()
class EncoderDecoderAction(EncoderDecoder):
    """Encoder Decoder segmentors for action recognition.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """

    def _init_decode_head(self, decode_head):
        # Only set decode_head, skip align_corners and segmentation-specific logic
        self.decode_head = builder.build_head(decode_head)

    def encode_decode(self, img, img_metas):
        """Encode images with backbone and decode into action predictions."""
        x = self.extract_feat(img)
        out = self._decode_head_forward_test(x, img_metas)
        return out  # shape (B, 2)

    def _decode_head_forward_train(self, x, img_metas, action):
        """Custom forward for ActionHead: forward + loss."""
        pred = self.decode_head(x)
        losses = self.decode_head.loss(pred, action)
        return losses
    
    def forward_train(self, img, img_metas, action):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        x = self.extract_feat(img)

        losses = dict()

        loss_decode = self._decode_head_forward_train(x, img_metas,
                                                     action)
        losses.update(loss_decode)

        return losses

    def whole_inference(self, img, img_meta, rescale):
        """Inference with full image for action prediction."""
        action_pred = self.encode_decode(img, img_meta)
        return action_pred  # shape (B, 2)

    def inference(self, img, img_meta, rescale):
        """Inference for action prediction (no flipping or segmentation logic)."""
        action_pred = self.whole_inference(img, img_meta, rescale)
        return action_pred  # shape (B, 2)

    def simple_test(self, img, img_meta, rescale=True):
        """Simple test with single image for action prediction."""
        action_pred = self.inference(img, img_meta, rescale)
        if torch.onnx.is_in_onnx_export():
            action_pred = action_pred.unsqueeze(0)
            return action_pred
        action_pred = action_pred.cpu().numpy()
        return action_pred  # shape (B, 2)

    def aug_test(self, imgs, img_metas, rescale=True):
        """Test with augmentations for action prediction.

        Only rescale=True is supported.
        """
        assert rescale
        action_pred = self.inference(imgs[0], img_metas[0], rescale)
        for i in range(1, len(imgs)):
            cur_action_pred = self.inference(imgs[i], img_metas[i], rescale)
            action_pred += cur_action_pred
        action_pred /= len(imgs)
        action_pred = action_pred.cpu().numpy()
        return action_pred  # shape (B, 2)
