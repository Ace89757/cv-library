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
epoch_1x = 12
milestones_1x = [8, 11]

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
        type='AlchemyTTFNet',
        in_channels=128,
        bbox_convs=2,
        bbox_channels=64,
        heatmap_convs=2,
        heatmap_channels=128,
        base_anchor=16,
        num_classes={{_base_.num_classes}},
        bbox_gaussian=True,
        bbox_area_process='log',
        loss_bbox=dict(type='EfficientIoULoss', loss_weight=5.0),
        loss_heatmap=dict(type='mmdet.GaussianFocalLoss', loss_weight=1.0),
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
    dict(type='LinearLR', start_factor=1.0 / 5, by_epoch=False, begin=0, end=500),
    dict(type='MultiStepLR', begin=0, end=max_epochs, by_epoch=True, milestones=milestones, gamma=0.1)
]

"""
optimizer
"""
optim_wrapper = dict(
    optimizer=dict(lr=0.016, momentum=0.9, weight_decay=0.0004),
    paramwise_cfg=dict(bias_lr_mult=2., bias_decay_mult=0.),
    clip_grad=dict(max_norm=35, norm_type=2))


train_cfg = dict(max_epochs=max_epochs)