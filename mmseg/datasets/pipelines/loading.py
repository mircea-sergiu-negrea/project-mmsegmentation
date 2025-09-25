# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import mmcv
import numpy as np

from ..builder import PIPELINES


@PIPELINES.register_module()
class LoadImageFromFile(object):
    """Load an image from file.

    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename"). Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`), "pad_shape" (same as `img_shape`),
    "scale_factor" (1.0) and "img_norm_cfg" (means=0 and stds=1).

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'cv2'
    """

    def __init__(self,
                 to_float32=False,
                 color_type='color',
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='cv2'):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results.get('img_prefix') is not None:
            filename = osp.join(results['img_prefix'],
                                results['img_info']['filename'])
        else:
            filename = results['img_info']['filename']
        img_bytes = self.file_client.get(filename)
        img = mmcv.imfrombytes(
            img_bytes, flag=self.color_type, backend=self.imdecode_backend)
        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = img
        # results['depth'] = depth  # for depth modality
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32},'
        repr_str += f"color_type='{self.color_type}',"
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str


@PIPELINES.register_module()
class LoadAnnotations(object):
    """Load annotations for semantic segmentation.

    Args:
        reduce_zero_label (bool): Whether reduce all label value by 1.
            Usually used for datasets where 0 is background label.
            Default: False.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'pillow'
    """

    def __init__(self,
                 reduce_zero_label=False,
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='pillow'):
        self.reduce_zero_label = reduce_zero_label
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results.get('seg_prefix', None) is not None:
            filename = osp.join(results['seg_prefix'],
                                results['ann_info']['seg_map'])
        else:
            filename = results['ann_info']['seg_map']
        img_bytes = self.file_client.get(filename)
        gt_semantic_seg = mmcv.imfrombytes(
            img_bytes, flag='unchanged',
            backend=self.imdecode_backend).squeeze().astype(np.uint8)
        # modify if custom classes
        if results.get('label_map', None) is not None:
            # Add deep copy to solve bug of repeatedly
            # replace `gt_semantic_seg`, which is reported in
            # https://github.com/open-mmlab/mmsegmentation/pull/1445/
            gt_semantic_seg_copy = gt_semantic_seg.copy()
            for old_id, new_id in results['label_map'].items():
                gt_semantic_seg[gt_semantic_seg_copy == old_id] = new_id
        # reduce zero_label
        if self.reduce_zero_label:
            # avoid using underflow conversion
            gt_semantic_seg[gt_semantic_seg == 0] = 255
            gt_semantic_seg = gt_semantic_seg - 1
            gt_semantic_seg[gt_semantic_seg == 254] = 255
        results['gt_semantic_seg'] = gt_semantic_seg
        results['seg_fields'].append('gt_semantic_seg')
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(reduce_zero_label={self.reduce_zero_label},'
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str

#
@PIPELINES.register_module()
class LoadActionsGT(object):
    """Load annotations for semantic segmentation.

    Args:
        reduce_zero_label (bool): Whether reduce all label value by 1.
            Usually used for datasets where 0 is background label.
            Default: False.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'pillow'
    """

    def __init__(self,
                 reduce_zero_label=False,
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='pillow'):
        self.reduce_zero_label = reduce_zero_label
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results.get('seg_prefix', None) is not None:
            filename = osp.join(results['seg_prefix'],
                                results['ann_info']['seg_map'])
        else:
            filename = results['ann_info']['seg_map']

        gt_semantic_seg = np.load(filename, allow_pickle=True)
        results['gt_semantic_seg'] = gt_semantic_seg
        results['seg_fields'].append('gt_semantic_seg')
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(reduce_zero_label={self.reduce_zero_label},'
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str


@PIPELINES.register_module()
class LoadDepthFromFile(object):
    """Load a depth map stored as a .npy file using image filename context.

    This transform expects to find a corresponding depth file for the current
    image using the pattern:

        depth_path = os.path.join(depth_prefix, seq, img_name + file_suffix)

    where `seq` is the parent directory name of the image relative path and
    `img_name` is the stem (filename without extension).

    Inputs/required keys:
    - results['img_info']['filename']: relative image path like 'SEQ/frame.jpg'
    - results['depth_prefix'] or depth_prefix argument: base folder for depth files

    Added keys:
    - results['depth']: HxWx1 float32 numpy array
    - results['depth_shape'], results['depth_filename']

    Args:
        depth_prefix (str | None): Base directory where depth .npy files live.
            If None, will use results['depth_prefix'].
        file_suffix (str): Suffix for depth files. Default: '.npy'.
        to_float32 (bool): Cast loaded array to float32. Default: True.
    """

    def __init__(self, depth_prefix=None, file_suffix='.npy', to_float32=True):
        self.depth_prefix = depth_prefix
        self.file_suffix = file_suffix
        self.to_float32 = to_float32

    def __call__(self, results):
        # Derive relative components from img_info
        rel_path = results['img_info']['filename']
        img_name = osp.splitext(osp.basename(rel_path))[0]
        seq = osp.basename(osp.dirname(rel_path))

        depth_root = results.get('depth_prefix') or self.depth_prefix
        if depth_root is None:
            raise KeyError('depth_prefix is not set in results and no depth_prefix provided to LoadDepthFromFile')

        depth_path = osp.join(depth_root, seq, img_name + self.file_suffix)
        if not osp.exists(depth_path):
            raise FileNotFoundError(f'Missing depth file: {depth_path} for image {rel_path}')

        depth = np.load(depth_path, allow_pickle=False)
        # Ensure HWC with single channel
        if depth.ndim == 2:
            depth = depth[..., None]
        elif depth.ndim == 3 and depth.shape[2] == 1:
            pass
        # else: keep as is (e.g., multi-channel depth-like data)

        if self.to_float32:
            depth = depth.astype(np.float32, copy=False)

        results['depth'] = depth
        results['depth_shape'] = depth.shape
        results['depth_filename'] = depth_path
        return results

    def __repr__(self):
        return (f"{self.__class__.__name__}(depth_prefix={self.depth_prefix}, "
                f"file_suffix='{self.file_suffix}', to_float32={self.to_float32})")


@PIPELINES.register_module()
class LoadFlowFromFile(object):
    """Load optical flow stored as .npy using image filename context.

    Expects file layout: flow_prefix/SEQ/NAME.npy and stores under results['flow'].

    Added keys:
    - flow (HxWx2 float32), flow_shape, flow_filename
    """

    def __init__(self, flow_prefix=None, file_suffix='.npy', to_float32=True):
        self.flow_prefix = flow_prefix
        self.file_suffix = file_suffix
        self.to_float32 = to_float32

    def __call__(self, results):
        rel_path = results['img_info']['filename']
        img_name = osp.splitext(osp.basename(rel_path))[0]
        seq = osp.basename(osp.dirname(rel_path))

        flow_root = results.get('flow_prefix') or self.flow_prefix
        if flow_root is None:
            raise KeyError('flow_prefix is not set in results and no flow_prefix provided to LoadFlowFromFile')

        flow_path = osp.join(flow_root, seq, img_name + self.file_suffix)
        if not osp.exists(flow_path):
            raise FileNotFoundError(f'Missing flow file: {flow_path} for image {rel_path}')

        flow = np.load(flow_path, allow_pickle=False)
        # Ensure HWC with 2 channels
        if flow.ndim == 2:
            # Promote to single channel if input is HW, but typical flow is HWx2
            flow = flow[..., None]
        if self.to_float32:
            flow = flow.astype(np.float32, copy=False)

        results['flow'] = flow
        results['flow_shape'] = flow.shape
        results['flow_filename'] = flow_path
        return results

    def __repr__(self):
        return (f"{self.__class__.__name__}(flow_prefix={self.flow_prefix}, "
                f"file_suffix='{self.file_suffix}', to_float32={self.to_float32})")


@PIPELINES.register_module()
class LoadParaFromFile(object):
    """Load para modality stored as .npy using image filename context.

    Expects file layout: para_prefix/SEQ/NAME.npy and stores under results['para'].

    Added keys:
    - para (HxWx1 float32), para_shape, para_filename
    """

    def __init__(self, para_prefix=None, file_suffix='.npy', to_float32=True):
        self.para_prefix = para_prefix
        self.file_suffix = file_suffix
        self.to_float32 = to_float32

    def __call__(self, results):
        rel_path = results['img_info']['filename']
        img_name = osp.splitext(osp.basename(rel_path))[0]
        seq = osp.basename(osp.dirname(rel_path))

        para_root = results.get('para_prefix') or self.para_prefix
        if para_root is None:
            raise KeyError('para_prefix is not set in results and no para_prefix provided to LoadParaFromFile')

        para_path = osp.join(para_root, seq, img_name + self.file_suffix)
        if not osp.exists(para_path):
            raise FileNotFoundError(f'Missing para file: {para_path} for image {rel_path}')

        para = np.load(para_path, allow_pickle=False)
        if para.ndim == 2:
            para = para[..., None]
        if self.to_float32:
            para = para.astype(np.float32, copy=False)

        results['para'] = para
        results['para_shape'] = para.shape
        results['para_filename'] = para_path
        return results

    def __repr__(self):
        return (f"{self.__class__.__name__}(para_prefix={self.para_prefix}, "
                f"file_suffix='{self.file_suffix}', to_float32={self.to_float32})")


@PIPELINES.register_module()
class LoadSegFromFile(object):
    """Load segmentation channels stored as .npy using image filename context.

    Expects file layout: seg_prefix/SEQ/NAME.npy and stores under results['seg'].

    Added keys:
    - seg (HxWxC float32), seg_shape, seg_filename
    """

    def __init__(self, seg_prefix=None, file_suffix='.npy', to_float32=True):
        self.seg_prefix = seg_prefix
        self.file_suffix = file_suffix
        self.to_float32 = to_float32

    def __call__(self, results):
        rel_path = results['img_info']['filename']
        img_name = osp.splitext(osp.basename(rel_path))[0]
        seq = osp.basename(osp.dirname(rel_path))

        seg_root = results.get('seg_prefix') or self.seg_prefix
        if seg_root is None:
            raise KeyError('seg_prefix is not set in results and no seg_prefix provided to LoadSegFromFile')

        seg_path = osp.join(seg_root, seq, img_name + self.file_suffix)
        if not osp.exists(seg_path):
            raise FileNotFoundError(f'Missing seg file: {seg_path} for image {rel_path}')

        seg = np.load(seg_path, allow_pickle=False)
        if seg.ndim == 2:
            seg = seg[..., None]
        if self.to_float32:
            seg = seg.astype(np.float32, copy=False)

        results['seg'] = seg
        results['seg_shape'] = seg.shape
        results['seg_filename'] = seg_path
        return results

    def __repr__(self):
        return (f"{self.__class__.__name__}(seg_prefix={self.seg_prefix}, "
                f"file_suffix='{self.file_suffix}', to_float32={self.to_float32})")


@PIPELINES.register_module()
class NormalizeByKey(object):
    """Normalize a modality stored under a given key in the results dict.

    This applies per-channel normalization: (x - mean) / std. The transform is
    no-op if the key is missing. Supports HxW and HxWxC (C>=1). Optionally
    converts BGR to RGB for the 'img' key via to_rgb semantics to align with
    mmseg's Normalize.

    Args:
        key (str): Key in results to normalize (e.g., 'img', 'depth').
        mean (Sequence[float]): Per-channel mean.
        std (Sequence[float]): Per-channel std.
        to_rgb (bool): If True and key == 'img', convert BGR->RGB before
            normalization to match torchvision/ImageNet conventions.
    """

    def __init__(self, key, mean, std, to_rgb=False):
        self.key = key
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, results):
        if self.key not in results:
            return results
        arr = results[self.key]
        if arr.ndim == 2:
            arr = arr[..., None]
        if arr.ndim != 3:
            raise ValueError(f'{self.__class__.__name__}: expected HxW or HxWxC for key {self.key}, got {arr.shape}')

        # Optionally convert BGR to RGB for standard image inputs
        if self.to_rgb and self.key == 'img' and arr.shape[2] >= 3:
            arr = arr[..., ::-1]

        # Ensure float32
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32)

        # Broadcast mean/std to channels
        if self.mean.size == 1 and arr.shape[2] > 1:
            mean = np.repeat(self.mean, arr.shape[2])
        else:
            mean = self.mean
        if self.std.size == 1 and arr.shape[2] > 1:
            std = np.repeat(self.std, arr.shape[2])
        else:
            std = self.std

        if mean.size != arr.shape[2] or std.size != arr.shape[2]:
            raise ValueError(f'{self.__class__.__name__}: mean/std channel mismatch for key {self.key}: '
                             f'{mean.size}/{std.size} vs C={arr.shape[2]}')

        arr = (arr - mean) / std
        results[self.key] = arr
        # Update norm cfg hints if normalizing the main image
        if self.key == 'img':
            results['img_norm_cfg'] = dict(mean=mean, std=std, to_rgb=self.to_rgb)
        return results

    def __repr__(self):
        return (f"{self.__class__.__name__}(key='{self.key}', mean={self.mean.tolist()}, "
                f"std={self.std.tolist()}, to_rgb={self.to_rgb})")


@PIPELINES.register_module()
class ConcatModalities(object):
    """Concatenate multiple modality arrays along channel dimension.

    Expects each modality key in `keys` to be present in results. Each array is
    coerced to HxWxC (C>=1) with float32 dtype. All must share the same H and W.
    The concatenated result is written to `out_key` (default: 'img') and shape
    metadata is updated. This transform should be placed after individual
    Normalize steps for each modality.

    Args:
        keys (Sequence[str]): Modality keys to concatenate in order.
        out_key (str): Results key to write concatenated array to. Default: 'img'.
        ensure_match (bool): If True, raise on spatial mismatch. Default: True.
    """

    def __init__(self, keys, out_key='img', ensure_match=True):
        assert isinstance(keys, (list, tuple)) and len(keys) >= 2, 'ConcatModalities needs >=2 keys'
        self.keys = list(keys)
        self.out_key = out_key
        self.ensure_match = ensure_match

    def __call__(self, results):
        arrays = []
        H = W = None
        for k in self.keys:
            if k not in results:
                raise KeyError(f'ConcatModalities requires results["{k}"] to be present')
            a = results[k]
            if a.ndim == 2:
                a = a[..., None]
            if a.ndim != 3:
                raise ValueError(f'ConcatModalities expects HxW or HxWxC for key {k}, got {a.shape}')
            if a.dtype != np.float32:
                a = a.astype(np.float32)
            h, w, _ = a.shape
            if H is None:
                H, W = h, w
            elif self.ensure_match and (H != h or W != w):
                raise ValueError(f'Spatial mismatch: {k} has {a.shape}, expected HxW={H}x{W}')
            arrays.append(a)

        merged = np.concatenate(arrays, axis=2)
        results[self.out_key] = merged
        # Update shape metadata for main image output key
        if self.out_key == 'img':
            results['img_shape'] = merged.shape
            results['pad_shape'] = merged.shape
            # img_norm_cfg no longer meaningful as combined; keep placeholder
            results['img_norm_cfg'] = dict(
                mean=np.zeros(merged.shape[2], dtype=np.float32),
                std=np.ones(merged.shape[2], dtype=np.float32),
                to_rgb=False)
        return results

    def __repr__(self):
        return (f"{self.__class__.__name__}(keys={self.keys}, out_key='{self.out_key}', "
                f"ensure_match={self.ensure_match})")
