# -*- coding: utf-8 -*-
# @Author  : ace
# Copyright by Ace, All Rights Reserved.

max_epochs = 12
val_interval = 1


"""
optimizer
"""
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', 
        lr=1.25e-4, 
        weight_decay=0.001,
        betas=(0.95, 0.99)),
    clip_grad=dict(max_norm=35, norm_type=2)
    )


"""
learning rate
"""
param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=5000),
    dict(type='MultiStepLR', begin=0, end=max_epochs, by_epoch=True, milestones=[9, 11], gamma=0.1)
]


"""
config
"""
train_cfg = dict(type='EpochBasedTrainLoop', val_interval=val_interval, max_epochs=max_epochs)
val_cfg = dict(type='AlchemyValLoop')
test_cfg = dict(type='TestLoop')


"""
auto scale lr
"""
auto_scale_lr = dict(enable=False, base_batch_size=16)