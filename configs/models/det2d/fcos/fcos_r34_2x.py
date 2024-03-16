# -*- coding: utf-8 -*-
# @Author  : ace
# Copyright by Ace, All Rights Reserved.

_base_ = [
    '../../../_base_/runtimes/det2d.py',
    '../../../_base_/datasets/det2d_bdd100k.py',
    '../../../_base_/pipelines/det2d_default.py'
]

batch_size = 32
score_thr = 0.35
repeat_times = 2


"""
model
"""
style = 'pytorch'

norm_cfg = dict(type='BN', requires_grad=True)
pretrained = dict(type='Pretrained', checkpoint='torchvision://resnet34')

if style == 'caffe':
    norm_cfg['requires_grad'] = False
    pretrained['checkpoint'] = 'open-mmlab://detectron2/resnet50_caffe'

model = dict(
    type='AlchemyDet2dDetector',
    data_preprocessor={{_base_.data_preprocessor}},
    backbone=dict(
        type='mmdet.ResNet',
        depth=34,
        num_stages=4,
        norm_eval=True,   # batch_size太小, 使用预训练的bn参数
        frozen_stages=0,  # conv=7层
        out_indices=(0, 1, 2, 3),
        norm_cfg=norm_cfg,
        style=style,
        init_cfg=pretrained
    ),
    neck=dict(
        type='AlchemyFPN',
        in_channels=[64, 128, 256, 512],
        out_channels=128,
        start_level=1,
        relu_before_extra_convs=True,
        extra_layers_source='on_output',
        out_indices=(1, 2, 3, 4, 5)
    ),
    head=dict(
        type='AlchemyFCOS',
        in_channels=128,
        stacked_convs=4,
        feat_channels=128,
        center_sampling=True,
        centerness_on_reg=True,
        strides=[8, 16, 32, 64, 128], 
        num_classes={{_base_.num_classes}},
        norm_cfg=norm_cfg,
        regress_ranges=((-1, 32), (32, 64), (64, 128), (128, 256), (256, 1e8)),
        loss_bbox=dict(type='EfficientIoULoss', loss_weight=1.0),
        loss_centerness=dict(type='mmdet.CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_cls=dict(type='mmdet.FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0),
        test_cfg=dict(nms_pre=1000, min_bbox_size=0, score_thr=score_thr, nms=dict(type='nms', iou_threshold=0.5), max_per_img=100),
        init_cfg=dict(type='Normal', layer='Conv2d', std=0.01, override=dict(type='Normal', name='cls_head', std=0.01, bias_prob=0.01)))
    )


"""
dataloader
"""
train_dataloader = dict(
    batch_size=batch_size,
    dataset=dict(
        type='RepeatDataset',
        times=repeat_times,
        dataset=dict(
            type={{_base_.dataset_type}},
            data_root={{_base_.data_root}},
            ann_file={{_base_.train_ann_file}},
            data_prefix=dict(img_path='images'),
            pipeline={{_base_.train_pipeline}},
            metainfo={{_base_.metainfo}}
        ))
    )

val_dataloader = dict(
    batch_size=batch_size,
    dataset=dict(
            test_mode=True,
            type={{_base_.dataset_type}},
            data_root={{_base_.data_root}},
            ann_file={{_base_.val_ann_file}},
            data_prefix=dict(img_path='images'),
            pipeline={{_base_.test_pipeline}},
            metainfo={{_base_.metainfo}})
    )

test_dataloader = val_dataloader


"""
scheduler
"""
param_scheduler = [
    dict(type='ConstantLR', factor=1.0 / 3, by_epoch=False, begin=0, end=500),
    dict(type='MultiStepLR', begin=0, end=12, by_epoch=True, milestones=[8, 11], gamma=0.1)
]


"""
optimizer
"""
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001),
    paramwise_cfg=dict(bias_lr_mult=2., bias_decay_mult=0.),
    clip_grad=dict(max_norm=35, norm_type=2)
    )


"""
config
"""
test_cfg = dict(type='TestLoop')
val_cfg = dict(type='AlchemyValLoop')
train_cfg = dict(type='EpochBasedTrainLoop', val_interval=1, max_epochs=12)


"""
auto scale lr
"""
auto_scale_lr = dict(enable=True, base_batch_size=16)