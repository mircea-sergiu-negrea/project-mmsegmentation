from mmcv import Config
from mmseg.datasets import build_dataset
import mmseg.datasets.ap4ad  # Ensure AP4ADDataset is registered

# Load your config file
cfg = Config.fromfile('/home/negreami/project/mmsegmentation/configs/_base_/datasets/ap4ad.py')

# Build the dataset (e.g., train split)
dataset = build_dataset(cfg.data['train'])

# Print the length and a sample
print()
print(f"Dataset length: {len(dataset)}")
sample = dataset[0]
print("Sample keys:", sample.keys())
print("Image shape:", sample['img'].shape)
print("Action:", sample['action'])
print()
print('Metadata:', sample.get('img_metas', {}))
print()