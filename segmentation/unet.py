dataset_type = 'PascalVOCDataset'
data_root = 'data/demo_voc'

model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='UNet',
        in_channels=3,
        base_channels=64,
        num_stages=5,
        enc_num_convs=(2, 2, 2, 2, 2),
        dec_num_convs=(2, 2, 2, 2),
        enc_dilations=(1, 1, 1, 1, 1),
        dec_dilations=(1, 1, 1, 1),
        downsamples=(True, True, True, True),
        strides=(1, 1, 1, 1, 1),
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        act_cfg=dict(type='ReLU'),
        upsample_cfg=dict(type='InterpConv'),
        with_cp=False),
    decode_head=dict(
        type='ASPPHead',
        in_channels=64,
        in_index=4,
        channels=16,
        dilations=(1, 12, 24, 36),
        num_classes=2,
        dropout_ratio=0.1,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(type='CrossEntropyLoss')),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=128,
        in_index=3,
        channels=64,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(type='CrossEntropyLoss')),
    train_cfg=dict(),
    test_cfg=dict(
        mode='slide',
        crop_size=(32, 64),
        stride=(42, 42)),
)

crop_size = (32, 64)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(64, 32)),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]

train_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='ImageSets/train.txt',
        data_prefix=dict(
            img_path='JPEGImages',
            seg_map_path='SegmentationClassPNG'),
        pipeline=train_pipeline),
    sampler=dict(type='InfiniteSampler', shuffle=True),
)

val_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='ImageSets/val.txt',
        data_prefix=dict(
            img_path='JPEGImages',
            seg_map_path='SegmentationClassPNG'),
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)

test_dataloader = val_dataloader
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator
visualizer = dict(
    type='SegLocalVisualizer',
    name='visualizer',
    save_dir='./result/deeplab_unet/output',
    vis_backends=[dict(type='LocalVisBackend')],
)

work_dir = 'result/deeplab_unet'
