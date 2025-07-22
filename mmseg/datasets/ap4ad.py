# Dataset Class for AP-for-AD
import os.path as osp

import mmcv
import numpy as np
from mmcv.utils import print_log
from PIL import Image

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class AP4ADDataset(CustomDataset):
    """AP4AD dataset"""

    def __init__(self, img_dir, action_dir, img_suffix='.jpg', action_suffix='.npy', pipeline=None, **kwargs):
        super(AP4ADDataset, self).__init__(
            img_dir=img_dir,
            img_suffix=img_suffix,
            pipeline=pipeline,
            **kwargs)
        self.action_dir = action_dir
        self.action_suffix = action_suffix

    def __getitem__(self, idx):
        return super(AP4ADDataset, self).__getitem__(idx)
    

    # CustomDataset expects 'ann' to be defined,
    # but we don't have annotations for action prediction

    def get_ann_info(self, idx):
        # No annotation info for action prediction
        return None
    
    def prepare_train_img(self, idx):
        # Only prepare image and action, no annotation
        img_info = self.img_infos[idx]
        results = dict(img_info=img_info)
        self.pre_pipeline(results)

        # Add action to results before pipeline
        img_path = osp.join(self.img_dir, img_info['filename'])
        img_filename = osp.basename(img_path)
        img_name = osp.splitext(img_filename)[0]
        seq = osp.basename(osp.dirname(img_path))
        action_path = osp.join(self.data_root, self.action_dir, seq, img_name + self.action_suffix)
        results['action'] = np.load(action_path)
        results = self.pipeline(results)
        return results

    # pipeline expects a 'flip' key

    def pre_pipeline(self, results):
        results['img_prefix'] = self.img_dir
        results['flip'] = False
        results['flip_direction'] = None