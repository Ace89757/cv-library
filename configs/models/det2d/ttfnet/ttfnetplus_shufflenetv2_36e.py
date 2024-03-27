# -*- coding: utf-8 -*-
# @Author  : ace
# Copyright by Ace, All Rights Reserved.

_base_ = [
    './ttfnet_shufflenetv2_36e.py'
]


model = dict(
    bbox_head=dict(
        type='AlchemyTTFNetPlusHead')
    )

