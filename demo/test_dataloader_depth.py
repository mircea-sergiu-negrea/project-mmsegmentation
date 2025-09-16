# Minimal dataset-only smoke test for AP4ADDataset + depth
import os
import numpy as np


def dataset_smoke_test():
    from mmseg.datasets.ap4ad import AP4ADDataset

    data_root = '/home/negreami/datasets/ap4ad_local/temp_test_for_depth'
    img_dir = os.path.join(data_root, 'rgb')
    action_dir = 'actions'
    depth_dir = 'depth'

    dataset = AP4ADDataset(
        img_dir=img_dir,
        action_dir=action_dir,
        depth_dir=depth_dir,
        data_root=data_root,
        modalities=['rgb', 'depth'],
        pipeline=[],
    )

    # load up to 3 samples and print concise diagnostics
    n = min(3, len(dataset))
    if n == 0:
        print('Dataset is empty')
        return

    for i in range(n):
        item = dataset[i]
        img = item['img']
        action = item.get('action')
        print(f'{i}: img {getattr(img, "shape", None)} dtype={getattr(img, "dtype", None)}', end='')
        if isinstance(img, np.ndarray) and img.ndim == 3:
            mins = img.min(axis=(0, 1))
            maxs = img.max(axis=(0, 1))
            print(f' min={mins} max={maxs}', end='')
        if action is not None:
            print(f' | action {getattr(action, "shape", None)} dtype={getattr(action, "dtype", None)}')
        else:
            print('')


if __name__ == '__main__':
    dataset_smoke_test()


