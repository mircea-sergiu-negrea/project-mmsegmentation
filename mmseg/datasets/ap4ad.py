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

        # --- modality loading blocks ---
        # RGB (required by default)
        if 'rgb' in self.modalities:
            rgb_img = mmcv.imread(img_path)
            # Convert BGR (OpenCV default) to RGB for consistent channel order
            if rgb_img is not None and rgb_img.ndim == 3 and rgb_img.shape[2] >= 3:
                rgb_img = rgb_img[:, :, ::-1]
            if rgb_img is None:
                raise FileNotFoundError(f"Missing or unreadable RGB image: {img_path}")
        else:
            rgb_img = None

        # Depth (single-channel .npy expected)
        if 'depth' in self.modalities:
            if self.depth_dir is None:
                raise ValueError('depth_dir must be specified if using depth modality')
            depth_path = osp.join(self.data_root, self.depth_dir, seq, img_name + '.npy')
            if not osp.exists(depth_path):
                raise FileNotFoundError(f"Missing depth file: {depth_path} for image {img_path}")
            depth = np.load(depth_path)
            # normalize shape to HWC
            if depth.ndim == 2:
                depth = depth[..., None]
            elif depth.ndim == 3 and depth.shape[2] == 1:
                pass
            else:
                # allow multi-channel depth-like arrays but cast to float32
                pass
            depth = depth.astype(np.float32)
        else:
            depth = None

        # Para (expect single-channel .npy)
        if 'para' in self.modalities:
            if self.para_dir is None:
                raise ValueError('para_dir must be specified if using para modality')
            para_path = osp.join(self.data_root, self.para_dir, seq, img_name + '.npy')
            if not osp.exists(para_path):
                raise FileNotFoundError(f"Missing para file: {para_path} for image {img_path}")
            para = np.load(para_path)
            if para.ndim == 2:
                para = para[..., None]
            para = para.astype(np.float32)
        else:
            para = None

        # Flow (expect 2-channel .npy or similar)
        if 'flow' in self.modalities:
            if self.flow_dir is None:
                raise ValueError('flow_dir must be specified if using flow modality')
            flow_path = osp.join(self.data_root, self.flow_dir, seq, img_name + '.npy')
            if not osp.exists(flow_path):
                raise FileNotFoundError(f"Missing flow file: {flow_path} for image {img_path}")
            flow = np.load(flow_path)
            if flow.ndim == 2:
                flow = flow[..., None]
            flow = flow.astype(np.float32)
        else:
            flow = None

        # Seg (expect multi-channel one-hot or soft labels .npy)
        if 'seg' in self.modalities:
            if self.seg_dir is None:
                raise ValueError('seg_dir must be specified if using seg modality')
            seg_path = osp.join(self.data_root, self.seg_dir, seq, img_name + '.npy')
            if not osp.exists(seg_path):
                raise FileNotFoundError(f"Missing seg file: {seg_path} for image {img_path}")
            seg = np.load(seg_path)
            if seg.ndim == 2:
                seg = seg[..., None]
            seg = seg.astype(np.float32)
        else:
            seg = None

        # --- Combine modalities in the order requested by self.modalities ---
        hwc_parts = []
        for mod in self.modalities:
            if mod == 'rgb' and rgb_img is not None:
                # ensure RGB is HWC
                if rgb_img.ndim == 2:
                    hwc_parts.append(rgb_img[..., None])
                else:
                    hwc_parts.append(rgb_img)
            elif mod == 'depth' and depth is not None:
                hwc_parts.append(depth)
            elif mod == 'para' and para is not None:
                hwc_parts.append(para)
            elif mod == 'flow' and flow is not None:
                hwc_parts.append(flow)
            elif mod == 'seg' and seg is not None:
                hwc_parts.append(seg)
            else:
                # If modality requested but file missing, the explicit loaders above
                # already raised FileNotFoundError. If modality not requested skip.
                pass

        if len(hwc_parts) == 0:
            raise ValueError('No valid input modalities specified')
        elif len(hwc_parts) == 1:
            inputs = hwc_parts[0]
        else:
            try:
                inputs = np.concatenate(hwc_parts, axis=2)
            except Exception as e:
                raise RuntimeError(f"Failed to concatenate modalities for index {idx}: {e}")

        results['img'] = inputs

        # Add action to results before pipeline
        action_path = osp.join(self.data_root, self.action_dir, seq, img_name + self.action_suffix)
        if not osp.exists(action_path):
            raise FileNotFoundError(f"Missing action file: {action_path} for image {img_path}")
        results['action'] = np.load(action_path)

        # Populate meta keys expected by Collect (since we don't use LoadImageFromFile)
        h, w = inputs.shape[:2]
        c = inputs.shape[2] if inputs.ndim == 3 else 1
        results['filename'] = img_path
        results['ori_filename'] = img_info['filename']
        results['ori_shape'] = (h, w, c)
        results['img_shape'] = (h, w, c)
        results['pad_shape'] = (h, w, c)
        results['scale_factor'] = 1.0  # no resizing yet
        # 'img_norm_cfg' will be inserted by Normalize pipeline step
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

        # --- modality loading blocks (same as train) ---
        if 'rgb' in self.modalities:
            rgb_img = mmcv.imread(img_path)
            if rgb_img is not None and rgb_img.ndim == 3 and rgb_img.shape[2] >= 3:
                rgb_img = rgb_img[:, :, ::-1]
            if rgb_img is None:
                raise FileNotFoundError(f"Missing or unreadable RGB image: {img_path}")
        else:
            rgb_img = None

        if 'depth' in self.modalities:
            if self.depth_dir is None:
                raise ValueError('depth_dir must be specified if using depth modality')
            depth_path = osp.join(self.data_root, self.depth_dir, seq, img_name + '.npy')
            if not osp.exists(depth_path):
                raise FileNotFoundError(f"Missing depth file: {depth_path} for image {img_path}")
            depth = np.load(depth_path)
            if depth.ndim == 2:
                depth = depth[..., None]
            depth = depth.astype(np.float32)
        else:
            depth = None

        if 'para' in self.modalities:
            if self.para_dir is None:
                raise ValueError('para_dir must be specified if using para modality')
            para_path = osp.join(self.data_root, self.para_dir, seq, img_name + '.npy')
            if not osp.exists(para_path):
                raise FileNotFoundError(f"Missing para file: {para_path} for image {img_path}")
            para = np.load(para_path)
            if para.ndim == 2:
                para = para[..., None]
            para = para.astype(np.float32)
        else:
            para = None

        if 'flow' in self.modalities:
            if self.flow_dir is None:
                raise ValueError('flow_dir must be specified if using flow modality')
            flow_path = osp.join(self.data_root, self.flow_dir, seq, img_name + '.npy')
            if not osp.exists(flow_path):
                raise FileNotFoundError(f"Missing flow file: {flow_path} for image {img_path}")
            flow = np.load(flow_path)
            if flow.ndim == 2:
                flow = flow[..., None]
            flow = flow.astype(np.float32)
        else:
            flow = None

        if 'seg' in self.modalities:
            if self.seg_dir is None:
                raise ValueError('seg_dir must be specified if using seg modality')
            seg_path = osp.join(self.data_root, self.seg_dir, seq, img_name + '.npy')
            if not osp.exists(seg_path):
                raise FileNotFoundError(f"Missing seg file: {seg_path} for image {img_path}")
            seg = np.load(seg_path)
            if seg.ndim == 2:
                seg = seg[..., None]
            seg = seg.astype(np.float32)
        else:
            seg = None

        hwc_parts = []
        for mod in self.modalities:
            if mod == 'rgb' and rgb_img is not None:
                if rgb_img.ndim == 2:
                    hwc_parts.append(rgb_img[..., None])
                else:
                    hwc_parts.append(rgb_img)
            elif mod == 'depth' and depth is not None:
                hwc_parts.append(depth)
            elif mod == 'para' and para is not None:
                hwc_parts.append(para)
            elif mod == 'flow' and flow is not None:
                hwc_parts.append(flow)
            elif mod == 'seg' and seg is not None:
                hwc_parts.append(seg)
            else:
                pass

        if len(hwc_parts) == 0:
            raise ValueError('No valid input modalities specified')
        elif len(hwc_parts) == 1:
            inputs = hwc_parts[0]
        else:
            try:
                inputs = np.concatenate(hwc_parts, axis=2)
            except Exception as e:
                raise RuntimeError(f"Failed to concatenate modalities for index {idx}: {e}")

        results['img'] = inputs

        action_path = osp.join(self.data_root, self.action_dir, seq, img_name + self.action_suffix)
        if not osp.exists(action_path):
            raise FileNotFoundError(f"Missing action file: {action_path} for image {img_path}")
        results['action'] = np.load(action_path)

        # Populate meta keys expected by Collect
        h, w = inputs.shape[:2]
        c = inputs.shape[2] if inputs.ndim == 3 else 1
        results['filename'] = img_path
        results['ori_filename'] = img_info['filename']
        results['ori_shape'] = (h, w, c)
        results['img_shape'] = (h, w, c)
        results['pad_shape'] = (h, w, c)
        results['scale_factor'] = 1.0
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

    #
    #add eval