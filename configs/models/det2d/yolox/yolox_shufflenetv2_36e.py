# -*- coding: utf-8 -*-
# @Author  : ace
# Copyright by Ace, All Rights Reserved.

_base_ = [
    '../../../_base_/runtimes/det2d.py',
    '../../../_base_/pipelines/det2d_default.py',
    '../../../_base_/schedules/adamw_36epoch.py',
    '../../../_base_/datasets/det2d_13cls_bdd100k.py'
]

batch_size = 116

"""
model
"""
# model settings
model = dict(
    type='mmdet.YOLOX',
    data_preprocessor={{_base_.data_preprocessor}},
    backbone=dict(type='AlchemyShuffleNetV2', depth='x1.0'),
    neck=dict(
        type='AlchemyFPN',
        in_channels=[24, 116, 232, 1024],
        out_channels=64,
        start_level=1,
        out_indices=(1, 2, 3)
    ),
    bbox_head=dict(
        type='mmdet.YOLOXHead',
        num_classes={{_base_.num_classes}},
        in_channels=64,
        feat_channels=64,
        stacked_convs=2,
        strides=(8, 16, 32),
        use_depthwise=False,
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='ReLU'),
        loss_cls=dict(type='mmdet.CrossEntropyLoss', use_sigmoid=True, reduction='sum', loss_weight=1.0),
        loss_bbox=dict(type='EfficientIoULoss', eps=1e-16, reduction='sum', loss_weight=5.0),
        loss_obj=dict(type='mmdet.CrossEntropyLoss', use_sigmoid=True, reduction='sum', loss_weight=1.0),
        loss_l1=dict(type='mmdet.L1Loss', reduction='sum', loss_weight=1.0)),
    train_cfg=dict(assigner=dict(type='mmdet.SimOTAAssigner', center_radius=2.5, iou_calculator=dict(type='mmdet.BboxOverlaps2D'))),
    test_cfg=dict(score_thr={{_base_.score_thr}}, nms=dict(type='nms', iou_threshold=0.65)))


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
