# -*- coding: utf-8 -*-
# @Author  : ace
# Copyright by Ace, All Rights Reserved.

max_epochs = 24
val_interval = 1


"""
learning rate
"""

param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=12,
        by_epoch=True,
        milestones=[16, 22],
        gamma=0.1)
]


"""
optimizer
"""

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001))


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