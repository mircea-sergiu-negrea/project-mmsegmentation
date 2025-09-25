# Minimal AP4AD config for action regression, only RGB inputs

norm_cfg = dict(type='BN', requires_grad=True)

# Original ResNet-50 v1c backbone config (disabled for faster debug):
# model = dict(
#     type='EncoderDecoderAction',
#     pretrained='open-mmlab://resnet50_v1c',
#     backbone=dict(
#         type='ResNetV1c',
#         depth=50,
#         in_channels=4,
#         num_stages=4,
#         out_indices=(0, 1, 2, 3),
#         dilations=(1, 1, 2, 4),
#         strides=(1, 2, 1, 1),
#         norm_cfg=norm_cfg,
#         norm_eval=False,
#         style='pytorch',
#         contract_dilation=True
#     ),
#     decode_head=dict(
#         type='ActionHead',
#         in_channels=2048,  # ResNet-50 final output channels
#     ),
#     train_cfg=None,
#     test_cfg=dict(mode='whole')
# )

# Faster ResNet-18 backbone config for debug:
model = dict(
    type='EncoderDecoderAction',
    # switched to a lightweight backbone for fast debug runs
    pretrained=None,
    backbone=dict(
        type='ResNet',
        depth=18,
        in_channels=4,  # RGBD temporary merge -> 4 channels
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch'
    ),
    decode_head=dict(
        type='ActionHead',
        in_channels=512,  # ResNet-18 final output channels
    ),
    train_cfg=None,
    test_cfg=dict(mode='whole')
)


# Normalization configs per modality
rgb_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
depth_norm_cfg = dict(mean=[0.334576027037], std=[0.370892853092], to_rgb=False)  # was already normalized

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadDepthFromFile'),
    dict(type='RandomFlip', prob=0.0),  # Explicitly do nothing, but set 'flip' key
    # Normalize each modality separately
    dict(type='NormalizeByKey', key='img', mean=rgb_norm_cfg['mean'], std=rgb_norm_cfg['std'], to_rgb=rgb_norm_cfg['to_rgb']),
    dict(type='NormalizeByKey', key='depth', mean=depth_norm_cfg['mean'], std=depth_norm_cfg['std'], to_rgb=depth_norm_cfg['to_rgb']),
    # Concatenate into a 4-channel tensor for the backbone
    dict(type='ConcatModalities', keys=['img', 'depth'], out_key='img'),
    # convert image and action to tensors (image: HWC->CHW)
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['action']),
    dict(type='Collect', keys=['img', 'action'])  # Dataset provides 'action' key
]

dataset_type = 'AP4ADDataset'
data_root = '/home/negreami/datasets/ap4ad_local'  # Change to ap4ad_local for actual runs, or ap4ad_local/test_dataset for debug
log_level = 'INFO'
gpu_ids = [3]  # Which GPU to use for train.py (single GPU training)
seed = 0
device = 'cuda'

data = dict(
    samples_per_gpu=2,  # batch size
    workers_per_gpu=2,
    persistent_workers=False,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='only_rgb',  # AP4AD image folder
        action_dir='actions',  # AP4AD actions folder
        depth_dir='depth',  # enable depth loader via dataset pre_pipeline
    pipeline=train_pipeline,
    classes=['action'],
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='only_rgb',
        action_dir='actions',
        depth_dir='depth',
    pipeline=train_pipeline,
    classes=['action'],
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='only_rgb',
        action_dir='actions',
        depth_dir='depth',
    pipeline=train_pipeline,
    classes=['action'],
    )
)

optimizer = dict(type='Adam', lr=0.0005)
optimizer_config = dict(grad_clip=None)

lr_config = dict(policy='step', step=[10, 20])
total_epochs = 4

log_config = dict(
    interval=10,
        hooks=[dict(type='TextLoggerHook'),
                     dict(type='WandbLoggerHook', by_epoch=False,
                         init_kwargs={'entity': "orangemsn",
                                                    'project': "ap4ad",
                                                    'name': "ap4ad-rgb-d_resnet18_debug"
                                                })],
)

checkpoint_config = dict(interval=1)

evaluation = dict(interval=1, metric='mse')

# Runner/workflow and other globals for epoch-based training
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
workflow = [('train', 1)]
# if we use the hardcoded work_dir in the dist_train.sh script, we don't need it here.
# work_dir = '/home/negreami/project/mmsegmentation/work_dirs/ap4ad_rgb_test'
classes = ['action']
resume_from = None
auto_resume = False
load_from = None

# Distributed training parameters (required for multi-GPU)
dist_params = dict(backend='nccl')
