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
    type='mmdet.RetinaNet',
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
        type='mmdet.RetinaHead',
        num_classes={{_base_.num_classes}},
        in_channels=64,
        stacked_convs=4,
        feat_channels=64,
        anchor_generator=dict(
            type='mmdet.AnchorGenerator',
            scales=[1.5],                   # 每个anchor的缩放系数
            ratios=[0.5, 1.0, 2.0],         # 每个anchor的宽高比
            strides=[8, 16, 32, 64, 128]    # 这个可以指定每层的anchor的base_size
            ),
        bbox_coder=dict(
            type='mmdet.DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='mmdet.L1Loss', loss_weight=1.0)),
    train_cfg=dict(
        assigner=dict(
            type='mmdet.MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1,
            iou_calculator=dict(type='mmdet.BboxOverlaps2D')),
        sampler=dict(type='mmdet.PseudoSampler'),  # Focal loss should use PseudoSampler
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr={{_base_.score_thr}},
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100))


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
