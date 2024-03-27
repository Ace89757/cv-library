# -*- coding: utf-8 -*-
# @Author  : ace
# Copyright by Ace, All Rights Reserved.

from mmdet.models.detectors.centernet import CenterNet

from ...registry import MODELS


@MODELS.register_module()
class AlchemyTTFNet(CenterNet):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
