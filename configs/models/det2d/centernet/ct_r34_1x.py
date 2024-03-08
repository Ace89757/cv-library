# -*- coding: utf-8 -*-
# @Author  : ace
# Copyright by Ace, All Rights Reserved.

_base_ = [
    '../../../_base_/runtimes/det2d.py',
    '../../../_base_/datasets/det2d_bdd100k.py',
    '../../../_base_/pipelines/det2d_default.py',
    '../../../_base_/lr_schedules/schedule_1x.py'
]

batch_size = 32
score_thr = 0.35

times = 1
epoch_1x = 28
milestones_1x = [18, 24]

max_epochs = int(epoch_1x * times)
milestones = [int(x * times) for x in milestones_1x]


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
        out_indices=(0, )
    ),
    head=dict(
        type='AlchemyCenterNet',
        in_channels=128,
        num_convs=1,
        feat_channels=128,
        class_agnostic=True, 
        num_classes={{_base_.num_classes}},
        test_cfg=dict(topk=100, local_maximum_kernel=3, score_thr=score_thr))
    )


"""
dataloader
"""
train_dataloader = dict(
    batch_size=batch_size,
    dataset=dict(
            type={{_base_.dataset_type}},
            data_root={{_base_.data_root}},
            ann_file={{_base_.train_ann_file}},
            data_prefix=dict(img_path='images'),
            pipeline={{_base_.train_pipeline}},
            metainfo={{_base_.metainfo}})
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
auto scale lr
"""
auto_scale_lr = dict(enable=True, base_batch_size=128)


"""
learning rate
"""
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=1000),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[18, 24],
        gamma=0.1)
]

"""
optimizer
"""
optim_wrapper = dict(clip_grad=dict(max_norm=35, norm_type=2))

train_cfg = dict(max_epochs=max_epochs)