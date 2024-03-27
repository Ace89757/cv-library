# -*- coding: utf-8 -*-
# @Author  : ace
# Copyright by Ace, All Rights Reserved.

_base_ = [
    '../../../_base_/runtimes/det2d.py',
    '../../../_base_/pipelines/det2d_default.py',
    '../../../_base_/schedules/adamw_36epoch.py',
    '../../../_base_/datasets/det2d_2cls_bdd100k.py'
]

batch_size = 2


"""
model
"""
# model settings
model = dict(
    type='mmdet.FasterRCNN',
    data_preprocessor={{_base_.data_preprocessor}},
    backbone=dict(type='AlchemyShuffleNetV2', depth='x1.0'),
    neck=dict(
        type='AlchemyFPN',
        in_channels=[24, 116, 232, 1024],
        out_channels=64,
        out_indices=(0, 1, 2, 3, 4)
    ),
    rpn_head=dict(
        type='mmdet.RPNHead',
        in_channels=64,
        feat_channels=64,
        anchor_generator=dict(
            type='mmdet.AnchorGenerator',
            scales=[1.5],                   # 每个anchor的缩放系数
            ratios=[0.5, 1.0, 2.0],         # 每个anchor的宽高比
            strides=[4, 8, 16, 32, 64]),    # 这个可以指定每层的anchor的base_size
        bbox_coder=dict(
            type='mmdet.DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(type='mmdet.CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='mmdet.L1Loss', loss_weight=1.0)),
    roi_head=dict(
        type='mmdet.StandardRoIHead',
        bbox_roi_extractor=dict(
            type='mmdet.SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=64,
            finest_scale=8,   # 参考: docs/guide/mmdet-AnchorGenerator使用说明.md
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='mmdet.Shared2FCBBoxHead',
            in_channels=64,
            fc_out_channels=512,
            roi_feat_size=7,
            num_classes={{_base_.num_classes}},
            bbox_coder=dict(
                type='mmdet.DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_predictor_cfg=dict(type='mmdet.Linear'),
            cls_predictor_cfg=dict(type='mmdet.Linear'),
            reg_class_agnostic=False,
            loss_cls=dict(type='mmdet.CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='mmdet.L1Loss', loss_weight=1.0))),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='mmdet.MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1,
                iou_calculator=dict(type='mmdet.BboxOverlaps2D')),
            sampler=dict(
                type='mmdet.RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='mmdet.MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1,
                iou_calculator=dict(type='mmdet.BboxOverlaps2D')),
            sampler=dict(
                type='mmdet.RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100),
        score_thr={{_base_.score_thr}},
    ))


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
