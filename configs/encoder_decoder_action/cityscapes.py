# Minimal Cityscapes config for action regression

model = dict(
    type='EncoderDecoderAction',
    backbone=dict(
        type='ResNet',
        depth=18,
        in_channels=3,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        style='pytorch'
    ),
    decode_head=dict(
        type='ActionHead',
        in_channels=512,  # match backbone output channels
    ),
    train_cfg=None,
    test_cfg=dict(mode='whole')
)

dataset_type = 'CityscapesDataset'
data_root = '/home/negreami/datasets/cityscapes'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='RandomFlip', prob=0.0),  # Explicitly do nothing, but set 'flip' key
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])  # For now, use gt_semantic_seg as dummy action
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='leftImg8bit/train',
        ann_dir='gtFine/train',
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='leftImg8bit/val',
        ann_dir='gtFine/val',
        pipeline=train_pipeline,
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='leftImg8bit/val',
        ann_dir='gtFine/val',
        pipeline=train_pipeline,
    )
)

optimizer = dict(type='Adam', lr=0.001)
optimizer_config = dict(grad_clip=None)

lr_config = dict(policy='step', step=[10, 20])
total_epochs = 1

log_config = dict(
    interval=10,
    hooks=[dict(type='TextLoggerHook')]
)

checkpoint_config = dict(interval=5)

evaluation = dict(interval=5, metric='mse')
