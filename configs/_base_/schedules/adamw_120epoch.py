# -*- coding: utf-8 -*-
# @Author  : ace
# Copyright by Ace, All Rights Reserved.

base_lr = 0.001
max_epochs = 120
num_last_epochs = 15
num_warmup_epochs = 5


"""
scheduler
"""
param_scheduler = [
    dict(
        # use quadratic formula to warm up 3 epochs and lr is updated by iteration
        type='mmdet.QuadraticWarmupLR',
        by_epoch=True,
        begin=0,
        end=num_warmup_epochs,
        convert_to_iter_based=True),
    dict(
        # use cosine lr from 3 to 105 epoch
        type='CosineAnnealingLR',
        eta_min=1e-6,
        begin=num_warmup_epochs,
        T_max=max_epochs - num_last_epochs,
        end=max_epochs - num_last_epochs,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        # use fixed lr during last 15 epochs
        type='ConstantLR',
        by_epoch=True,
        factor=1,
        begin=max_epochs - num_last_epochs,
        end=max_epochs,
    )
]


"""
optimizer
"""
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.01),
    clip_grad=dict(max_norm=35, norm_type=2)
    )


"""
config
"""
test_cfg = dict(type='TestLoop')
val_cfg = dict(type='AlchemyValLoop')
train_cfg = dict(type='EpochBasedTrainLoop', val_interval=1, max_epochs=max_epochs)


"""
auto scale lr
"""
auto_scale_lr = dict(enable=True, base_batch_size=32)