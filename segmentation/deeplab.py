_base_ = [
    'mmseg::_base_/models/deeplabv3.py',
    'mmseg::_base_/datasets/pascal_voc12.py',
    'mmseg::_base_/default_runtime.py',
    'mmseg::_base_/schedules/schedule_20k.py'
]

model = dict(
    decode_head=dict(num_classes=2),
    auxiliary_head=dict(num_classes=2),
)

data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[16.816, 16.609, 19.589],
    std=[32.704, 32.418, 36.326],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
)

dataset_type = 'PascalVOCDataset'
data_root = 'data/voc/'

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
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
        pipeline=train_pipeline,
    ),
)

val_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='ImageSets/val.txt',
        data_prefix=dict(
            img_path='JPEGImages',
            seg_map_path='SegmentationClassPNG'),
        pipeline=test_pipeline,
    ),
)

test_dataloader = val_dataloader

val_evaluator = dict(
    type='IoUMetric',
    iou_metrics=['mIoU'],
)

test_evaluator = val_evaluator
