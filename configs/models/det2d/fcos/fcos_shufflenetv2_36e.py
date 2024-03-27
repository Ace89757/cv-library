# -*- coding: utf-8 -*-
# @Author  : ace
# Copyright by Ace, All Rights Reserved.

_base_ = [
    '../../../_base_/runtimes/det2d.py',
    '../../../_base_/pipelines/det2d_default.py',
    '../../../_base_/schedules/adamw_36epoch.py',
    '../../../_base_/datasets/det2d_13cls_bdd100k.py'
]

batch_size = 16

"""
model
"""
# model settings
model = dict(
    type='mmdet.FCOS',
    data_preprocessor={{_base_.data_preprocessor}},
    backbone=dict(type='AlchemyShuffleNetV2', depth='x1.0'),
    neck=dict(
        type='AlchemyFPN',
        in_channels=[24, 116, 232, 1024],
        out_channels=64,
        start_level=1,
        out_indices=(1, 2, 3, 4, 5)
    ),
    bbox_head=dict(
        type='AlchemyFCOSHead',
        num_classes={{_base_.num_classes}},
        in_channels=64,
        stacked_convs=4,
        feat_channels=64,
        strides=[8, 16, 32, 64, 128],
        norm_on_bbox=True,
        centerness_on_reg=True,
        dcn_on_last_conv=False,
        center_sampling=True,
        conv_bias=True,
        norm_cfg=dict(type='BN', requires_grad=True),
        regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 512), (512, 1e9)),
        loss_bbox=dict(type='EfficientIoULoss', loss_weight=1.0),
        loss_cls=dict(type='mmdet.FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0),
        loss_centerness=dict(type='mmdet.CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        bbox_coder=dict(type='mmdet.DistancePointBBoxCoder')),
    test_cfg=dict(nms_pre=1000, min_bbox_size=0, score_thr={{_base_.score_thr}}, nms=dict(type='nms', iou_threshold=0.6), max_per_img=100))

train_dataloader = dict(
    batch_size=batch_size,
    dataset=dict(
        type={{_base_.dataset_type}},
        data_root={{_base_.data_root}},
        ann_file={{_base_.train_ann_file}},
        data_prefix=dict(img_path='images'),
        pipeline={{_base_.train_pipeline}},
        metainfo={{_base_.metainfo}}
        )
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
