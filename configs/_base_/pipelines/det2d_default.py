# -*- coding: utf-8 -*-
# @Author  : ace
# Copyright by Ace, All Rights Reserved.

backend_args = None


"""
data preprocessor
"""
data_preprocessor = dict(
    type='mmdet.DetDataPreprocessor', 
    mean=[123.675, 116.28, 103.53],  # rgb
    std=[58.395, 57.12, 57.375],     # rgb
    bgr_to_rgb=True,                 # 先转换通道顺序，再 -mean / std
    pad_size_divisor=32
    )


"""
pipeline
"""

input_h = 480
input_w = 800
train_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args=backend_args, to_float32=True),
    dict(type='mmdet.LoadAnnotations', with_bbox=True),
    dict(type='mmdet.PhotoMetricDistortion', brightness_delta=32, contrast_range=(0.5, 1.5), saturation_range=(0.5, 1.5), hue_delta=18),
    dict(type='AlchemyLetterBoxFixed', input_size=(input_h, input_w)),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(type='mmdet.PackDetInputs')
]


test_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args=backend_args, to_float32=True),
    dict(type='AlchemyLetterBoxFixed', input_size=(input_h, input_w)),
    dict(type='mmdet.PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]


"""
dataloader
"""
train_dataloader = dict(
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True))

val_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False)
    )

test_dataloader = val_dataloader