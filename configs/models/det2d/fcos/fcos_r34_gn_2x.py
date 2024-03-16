# -*- coding: utf-8 -*-
# @Author  : ace
# Copyright by Ace, All Rights Reserved.

_base_ = [
    './fcos_r34_2x.py'
]


model = dict(
    head=dict(
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True)
    ))
