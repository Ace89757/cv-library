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
    type='mmdet.CenterNet',
    data_preprocessor={{_base_.data_preprocessor}},
    backbone=dict(type='AlchemyShuffleNetV2', depth='x1.0'),
    neck=dict(
        type='AlchemyFPN',
        in_channels=[24, 116, 232, 1024],
        out_channels=64,
        out_indices=(0, )
    ),
    bbox_head=dict(
        type='AlchemyCenterNetHead',
        in_channels=64,
        feat_channels=64,
        num_classes={{_base_.num_classes}},
        loss_center_heatmap=dict(type='mmdet.GaussianFocalLoss', loss_weight=1.0),
        loss_wh=dict(type='mmdet.L1Loss', loss_weight=0.1),
        loss_offset=dict(type='mmdet.L1Loss', loss_weight=1.0)),
    train_cfg=dict(down_ratio=4),
    test_cfg=dict(topk=100, local_maximum_kernel=3, max_per_img=100, score_thr={{_base_.score_thr}}))

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
