dataset_type = 'SODADDataset'
data_root = '/home/cpl/dataset/SODA-D/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1200, 1200), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1200, 1200),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'train/Annotations/train.json',
        img_prefix=data_root + 'train/Images/train/',
        pipeline=train_pipeline,
        ori_ann_file=data_root + 'Annotations/train.json'
    ),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'val/Annotations/val.json',
        img_prefix=data_root + 'val/Images/val/',
        pipeline=test_pipeline,
        ori_ann_file=data_root + 'Annotations/val.json'
    ),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'test/Annotations/test.json',
        img_prefix=data_root + 'test/Images/test/',
        pipeline=test_pipeline,
        ori_ann_file=data_root + 'Annotations/test.json'
    ))
