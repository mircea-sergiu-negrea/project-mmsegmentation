from mmcv import Config
from mmseg.apis import set_random_seed, train_segmentor
from mmseg.models import build_segmentor
from mmseg.datasets import build_dataset
import mmseg.datasets.ap4ad
import mmseg.models.segmentors.encoder_decoder_action
import mmseg.models.decode_heads.action_head

# Load config
cfg = Config.fromfile('/home/negreami/project/mmsegmentation/configs/encoder_decoder_action/ap4ad_rgb.py')

# Set work directory
cfg.work_dir = 'home/negreami/project/mmsegmentation/work_dirs/ap4ad_rgb_test'

# Build dataset
datasets = [build_dataset(cfg.data.train)]

# Build model
model = build_segmentor(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))

# Set random seed for reproducibility
set_random_seed(0, deterministic=False)
model.init_weights()

# Train
train_segmentor(model, datasets, cfg, distributed=False, validate=True)