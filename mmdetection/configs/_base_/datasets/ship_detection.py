# dataset settings
dataset_type = 'ShipDataset'
data_root = './data/'
classes = ("ship",)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# albu_train_transforms = [
#     dict(
#         type='OneOf',
#         transforms=[
#             dict(type='MultiplicativeNoise', multiplier=[0.5, 1.5], elementwise=True, p=1),
#             dict(type='MultiplicativeNoise', multiplier=0.5, p=1),
#             dict(type='MultiplicativeNoise', multiplier=[0.5, 1.5], elementwise=True, per_channel=True, p=1)
#         ],
#         p=0.1,),
#     # dict(
#     #     type='RandomBrightnessContrast',
#     #     brightness_limit=[0.1, 0.3],
#     #     contrast_limit=[0.1, 0.3],
#     #     p=0.2),
#     # dict(type='JpegCompression', quality_lower=85, quality_upper=95, p=0.2),
#     # # dict(type='ChannelShuffle', p=0.1),
#     # dict(type='Cutout', num_holes=10, max_h_size=20, max_w_size=20, fill_value=0, p=1),
#     dict(
#         type='OneOf',
#         transforms=[
#             dict(type='Blur', blur_limit=3, p=1.0),
#             dict(type='Blur', blur_limit=15, p=1.0),
#             dict(type='MedianBlur', blur_limit=3, p=1.0)
#         ],
#         p=0.1),
# ]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    # dict(type='Resize', img_scale=(896, 896), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    # dict(
    #     type='RandomFlip',
    #     flip_ratio=[0.25, 0.25, 0.25],
    #     direction=['horizontal', 'vertical', 'diagonal']),

    # dict(
    #     type='Albu',
    #     transforms=albu_train_transforms,
    #     bbox_params=dict(
    #         type='BboxParams',
    #         format='pascal_voc',
    #         label_fields=['gt_labels'],
    #         min_area=0,
    #         min_visibility=0,
    #         filter_lost_elements=True),
    #     # keymap={
    #     #     'img': 'image',
    #     #     'gt_masks': 'masks',
    #     #     'gt_bboxes': 'bboxes'
    #     # },
    #     update_pad_shape=False,
    #     skip_img_without_anno=True),

    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        # img_scale=(896, 896),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        classes=classes,
        # ann_file=data_root + 'annotations/train.json',
        ann_file=data_root + 'annotations/instances_all.json',
        img_prefix=data_root + '/images/train_data/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'annotations/val.json',
        img_prefix=data_root + '/images/train_data/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'annotations/test.json',
        img_prefix=data_root + '/images/test_data/',
        pipeline=test_pipeline))
