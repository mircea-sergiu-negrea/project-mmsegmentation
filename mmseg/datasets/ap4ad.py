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
    
    def evaluate(self, results, metric='mse', logger=None, **kwargs):
        """Evaluate regression results using mean squared error (MSE)."""
        import numpy as np
        if isinstance(metric, str):
            metric = [metric]
        if 'mse' not in metric:
            raise KeyError(f"metric {metric} is not supported. Only 'mse' is supported for AP4ADDataset.")
        # results: list of np.ndarray or torch.Tensor, shape (2,) per sample (throttle, steer)
        # ground truth: get from self.img_infos and self.action_dir
        gt_actions = []
        for idx in range(len(self.img_infos)):
            img_info = self.img_infos[idx]
            img_path = osp.join(self.img_dir, img_info['filename'])
            img_filename = osp.basename(img_path)
            img_name = osp.splitext(img_filename)[0]
            seq = osp.basename(osp.dirname(img_path))
            action_path = osp.join(self.data_root, self.action_dir, seq, img_name + self.action_suffix)
            gt_action = np.load(action_path)
            gt_actions.append(gt_action)
        gt_actions = np.stack(gt_actions)
        preds = np.array([r if isinstance(r, np.ndarray) else r.cpu().numpy() for r in results])
        # Ensure shapes match
        if preds.shape != gt_actions.shape:
            raise ValueError(f"Prediction shape {preds.shape} does not match ground truth shape {gt_actions.shape}")
        mse = np.mean((preds - gt_actions) ** 2)
        return {'mse': mse}

    def __init__(self, img_dir, action_dir, img_suffix='.jpg', action_suffix='.npy', pipeline=None,
                 modalities=None, depth_dir=None, para_dir=None, flow_dir=None, seg_dir=None, **kwargs):
        """
        Initialize AP4ADDataset. Supports modular modalities.

        modalities: list of modality names to load in addition to 'rgb'.
                    e.g. ['rgb'] or ['rgb','depth'] or ['rgb','depth','para']
        depth_dir/para_dir/flow_dir/seg_dir: directory names under data_root where
                    the per-modality .npy files live.
        """
        if modalities is None:
            modalities = ['rgb']

        super(AP4ADDataset, self).__init__(
            img_dir=img_dir,
            img_suffix=img_suffix,
            pipeline=pipeline,
            **kwargs)
        self.action_dir = action_dir
        self.action_suffix = action_suffix
        # modality configuration
        self.modalities = list(modalities)
        self.depth_dir = depth_dir
        self.para_dir = para_dir
        self.flow_dir = flow_dir
        self.seg_dir = seg_dir

    def __getitem__(self, idx):
        return super(AP4ADDataset, self).__getitem__(idx)
    

    # CustomDataset expects 'ann' to be defined,
    # but we don't have annotations for action prediction

    def get_ann_info(self, idx):
        # No annotation info for action prediction
        return None
    
    def prepare_train_img(self, idx):

        # Prepare image, optional modalities, and action, no annotation
        img_info = self.img_infos[idx]
        results = dict(img_info=img_info)
        self.pre_pipeline(results)

        img_path = osp.join(self.img_dir, img_info['filename'])
        img_filename = osp.basename(img_path)
        img_name = osp.splitext(img_filename)[0]
        seq = osp.basename(osp.dirname(img_path))

        # For pipeline-based loading, we only initialize results here; modalities are loaded by pipeline

        # Add action to results before pipeline
        action_path = osp.join(self.data_root, self.action_dir, seq, img_name + self.action_suffix)
        if not osp.exists(action_path):
            raise FileNotFoundError(f"Missing action file: {action_path} for image {img_path}")
        results['action'] = np.load(action_path)

        # Hand over to pipeline; LoadImageFromFile and other modality loaders will populate img and meta
        results = self.pipeline(results)
        return results

    def prepare_test_img(self, idx):
        # Prepare image, optional modalities, and action, no annotation
        img_info = self.img_infos[idx]
        results = dict(img_info=img_info)
        self.pre_pipeline(results)

        img_path = osp.join(self.img_dir, img_info['filename'])
        img_filename = osp.basename(img_path)
        img_name = osp.splitext(img_filename)[0]
        seq = osp.basename(osp.dirname(img_path))

        # For pipeline-based loading, we only initialize results here; modalities are loaded by pipeline

        action_path = osp.join(self.data_root, self.action_dir, seq, img_name + self.action_suffix)
        if not osp.exists(action_path):
            raise FileNotFoundError(f"Missing action file: {action_path} for image {img_path}")
        results['action'] = np.load(action_path)

        # Hand over to pipeline; LoadImageFromFile and other modality loaders will populate img and meta
        results = self.pipeline(results)
        return results

    # pipeline expects a 'flip' key

    def pre_pipeline(self, results):
        results['img_prefix'] = self.img_dir
        results['flip'] = False
        results['flip_direction'] = None
        # Provide modality prefixes for pipeline-based loaders (e.g., LoadDepthFromFile)
        if getattr(self, 'depth_dir', None) is not None:
            results['depth_prefix'] = osp.join(self.data_root, self.depth_dir)
        if getattr(self, 'para_dir', None) is not None:
            results['para_prefix'] = osp.join(self.data_root, self.para_dir)
        if getattr(self, 'flow_dir', None) is not None:
            results['flow_prefix'] = osp.join(self.data_root, self.flow_dir)
        if getattr(self, 'seg_dir', None) is not None:
            results['seg_prefix'] = osp.join(self.data_root, self.seg_dir)

    #
    #add eval