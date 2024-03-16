# -*- coding: utf-8 -*-
# @Author  : ace
# Copyright by Ace, All Rights Reserved.

_base_ = [
    './ttfnet_r34_2x.py'
]


model = dict(
    head=dict(
        type='AlchemyTTFNetPlus')
    )

