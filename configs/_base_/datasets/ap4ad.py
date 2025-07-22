# Minimal config for AP4ADDataset

dataset_type = 'AP4ADDataset'
data_root = '/home/negreami/datasets/ap4ad_local'

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Collect', keys=['img', 'action']),
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='only_rgb',
        action_dir='actions',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='only_rgb',
        action_dir='actions',
        pipeline=train_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='only_rgb',
        action_dir='actions',
        pipeline=train_pipeline)
)
