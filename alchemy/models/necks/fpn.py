# -*- coding: utf-8 -*-
# @Author  : ace
# Copyright by Ace, All Rights Reserved.

import torch.nn as nn

import torch.nn.functional as F

from torch import Tensor
from typing import List, Tuple

from mmcv.cnn import ConvModule

from mmengine.model import BaseModule

from mmdet.utils import ConfigType, MultiConfig, OptConfigType

from ...registry import MODELS



@MODELS.register_module()
class AlchemyFPN(BaseModule):
    def __init__(self,
                 in_channels: List[int],
                 out_channels: int,
                 out_indices: Tuple[int] = (0, 1, 2, 3),  # p2, p3, p4, p5的顺序
                 start_level: int = 0,
                 end_level: int = -1,
                 extra_layers_source: str = 'on_input',
                 conv_cfg: OptConfigType = None,
                 norm_cfg: OptConfigType = None,
                 act_cfg: OptConfigType = None,
                 no_norm_on_lateral: bool = False,
                 relu_before_extra_convs: bool = False,
                 upsample_cfg: ConfigType = dict(mode='nearest'),
                 init_cfg: MultiConfig = dict(type='Xavier', layer='Conv2d', distribution='uniform')) -> None:
        super().__init__(init_cfg=init_cfg)

        """
        大致分3中情况:
        1. num_in = num_out:
            正常操作
        2. num_in > num_out:
            不需要额外层, 根据start_level和end_level构建lateral层, 根据out_indices构建fpn层
        3. num_in < num_out
            需要额外层
            1. 计算需要的额外层数
        """

        assert isinstance(in_channels, list)
        assert start_level <= out_indices[0]

        # args
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = len(out_indices)
        self.out_indices = out_indices
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.upsample_cfg = upsample_cfg.copy()

        # 计算实际使用的backbone的起始及终止层索引
        self.start_level = start_level

        if end_level == -1 or end_level == self.num_ins - 1:   # 最后一层
            self.backbone_end_level = self.num_ins
        else:
            self.backbone_end_level = end_level + 1
            assert end_level < self.num_ins
        
        # 构建lateral层
        self.lateral_convs = self._build_lateral_layers()

        # 构建fpn层
        self.fpn_convs = self._build_fpn_layers()

        # 判断是否需要额外的层
        self.add_extra_layers = False
        self.num_extra_levels = 0      # 需要额外层的数量
        for ind in out_indices:
            if ind + 1 > self.num_ins:
                self.add_extra_layers = True   # 如果输出的层索引大于输入的层数, 就需要额外层
                self.num_extra_levels += 1

        # 额外层的输入使用哪个阶段
        assert extra_layers_source in ('on_input', 'on_lateral', 'on_output')
        self.extra_layers_source = extra_layers_source

        # 构建额外层
        self._build_extra_layers()
        
        self.fp16_enabled = False

    def _build_lateral_layers(self) -> nn.ModuleList:
        lateral_convs = nn.ModuleList()
        for i in range(self.start_level, self.backbone_end_level):
            lateral_convs.append(
                ConvModule(
                    self.in_channels[i],
                    self.out_channels,
                    kernel_size=1,
                    stride=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg if not self.no_norm_on_lateral else None,
                    act_cfg=self.act_cfg,
                    inplace=False)
            )
        
        return lateral_convs
    
    def _build_fpn_layers(self) -> nn.ModuleList:
        fpn_convs = nn.ModuleList()

        for out_ind in self.out_indices:
            if out_ind < self.num_ins:
                fpn_convs.append(
                    ConvModule(
                        self.out_channels,
                        self.out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg,
                        inplace=False)
                    )
        
        return fpn_convs

    def _build_extra_layers(self) -> None:
        if self.add_extra_layers and self.num_extra_levels > 0:
            for i in range(self.num_extra_levels):
                if i == 0 and self.extra_layers_source == 'on_input':
                    in_channels = self.in_channels[self.backbone_end_level - 1]   # 使用backbone的end_level作为第一个额外层的输入
                else:
                    in_channels = self.out_channels
                
                extra_fpn_conv = ConvModule(
                    in_channels,
                    self.out_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                    inplace=False)
                
                self.fpn_convs.append(extra_fpn_conv)
    
    def forward(self, inputs: Tuple[Tensor]) -> tuple:
        """
        Forward function.

        Args:
            inputs (tuple[Tensor]): Features from the upstream network, each is a 4D-tensor.

        Returns:
            tuple: Feature maps, each is a 4D-tensor.
        """
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
            #  it cannot co-exist with `size` in `F.interpolate`.
            if 'scale_factor' in self.upsample_cfg:
                # fix runtime error of "+=" inplace operation in PyTorch 1.10
                laterals[i - 1] = laterals[i - 1] + F.interpolate(laterals[i], **self.upsample_cfg)
            else:
                prev_shape = laterals[i - 1].shape[2:]
                laterals[i - 1] = laterals[i - 1] + F.interpolate(laterals[i], size=prev_shape, **self.upsample_cfg)
        
        # build outputs
        # part 1: from original levels
        outs = []
        fpn_ind = 0
        for idx, out_ind in enumerate(self.out_indices):
            if out_ind < self.num_ins:
                fpn_ind = idx
                outs.append(self.fpn_convs[fpn_ind](laterals[out_ind - self.start_level]))
                
        # part 2: add extra levels
        if self.add_extra_layers:
            if self.extra_layers_source == 'on_input':
                extra_source = inputs[self.backbone_end_level - 1]
            elif self.extra_layers_source == 'on_lateral':
                extra_source = laterals[-1]
            elif self.extra_layers_source == 'on_output':
                extra_source = outs[-1]
            else:
                raise NotImplementedError

            fpn_ind = len(outs)
            outs.append(self.fpn_convs[fpn_ind](extra_source))
            
            for _ in self.out_indices[len(outs):]:
                fpn_ind += 1
                if self.relu_before_extra_convs:
                    outs.append(self.fpn_convs[fpn_ind](F.relu(outs[-1])))
                else:
                    outs.append(self.fpn_convs[fpn_ind](outs[-1]))

        return tuple(outs)