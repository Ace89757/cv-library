# -*- coding: utf-8 -*-
# @Author  : ace
# Copyright by Ace, All Rights Reserved.

import torch
import torch.nn as nn

from torch import Tensor
from torchvision.utils import _log_api_usage_once
from torchvision.models.shufflenetv2 import ShuffleNet_V2_X0_5_Weights, ShuffleNet_V2_X1_0_Weights, ShuffleNet_V2_X1_5_Weights, ShuffleNet_V2_X2_0_Weights

from ...registry import MODELS


def channel_shuffle(x: Tensor, groups: int) -> Tensor:
    """
    copy from 'torchvision/models/shufflenetv2.py'
    """
    batchsize, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


class InvertedResidual(nn.Module):
    def __init__(self, inp: int, oup: int, stride: int) -> None:
        super().__init__()

        if not (1 <= stride <= 3):
            raise ValueError("illegal stride value")
        self.stride = stride

        branch_features = oup // 2
        if (self.stride == 1) and (inp != branch_features << 1):
            raise ValueError(
                f"Invalid combination of stride {stride}, inp {inp} and oup {oup} values. If stride == 1 then inp should be equal to oup // 2 << 1."
            )

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                self.depthwise_conv(inp, inp, kernel_size=3, stride=self.stride, padding=1),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True),
            )
        else:
            self.branch1 = nn.Sequential()

        self.branch2 = nn.Sequential(
            nn.Conv2d(
                inp if (self.stride > 1) else branch_features,
                branch_features,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            self.depthwise_conv(branch_features, branch_features, kernel_size=3, stride=self.stride, padding=1),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def depthwise_conv(
        i: int, o: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False
    ) -> nn.Conv2d:
        return nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias, groups=i)

    def forward(self, x: Tensor) -> Tensor:
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = channel_shuffle(out, 2)

        return out


def weights_verify(depth, weights):
    if depth == 'x0.5':
        return ShuffleNet_V2_X0_5_Weights.verify(weights)
    elif depth == 'x1.0':
        return ShuffleNet_V2_X1_0_Weights.verify(weights)
    elif depth == 'x1.5':
        return ShuffleNet_V2_X1_5_Weights.verify(weights)
    elif depth == 'x2.0':
        return ShuffleNet_V2_X2_0_Weights.verify(weights)


@MODELS.register_module()
class AlchemyShuffleNetV2(nn.Module):
    arch_settings = {
        'x0.5': ([4, 8, 4], [24, 48, 96, 192, 1024], ShuffleNet_V2_X0_5_Weights.IMAGENET1K_V1),
        'x1.0': ([4, 8, 4], [24, 116, 232, 464, 1024], ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1),
        'x1.5': ([4, 8, 4], [24, 176, 352, 704, 1024], ShuffleNet_V2_X1_5_Weights.IMAGENET1K_V1),
        'x2.0': ([4, 8, 4], [24, 244, 488, 976, 2048], ShuffleNet_V2_X2_0_Weights.IMAGENET1K_V1)
    }

    def __init__(self, depth: str = 'x1.0') -> None:
        super().__init__()
        _log_api_usage_once(self)
        stages_repeats, stages_out_channels, weights = self.arch_settings[depth]

        weights = weights_verify(depth, weights)

        if len(stages_repeats) != 3:
            raise ValueError("expected stages_repeats as list of 3 positive ints")
        
        if len(stages_out_channels) != 5:
            raise ValueError("expected stages_out_channels as list of 5 positive ints")
        
        self._stage_out_channels = stages_out_channels

        input_channels = 3
        output_channels = self._stage_out_channels[0]
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )
        input_channels = output_channels

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Static annotations for mypy
        self.stage2: nn.Sequential
        self.stage3: nn.Sequential
        self.stage4: nn.Sequential
        stage_names = [f"stage{i}" for i in [2, 3, 4]]
        for name, repeats, output_channels in zip(stage_names, stages_repeats, self._stage_out_channels[1:]):
            seq = [InvertedResidual(input_channels, output_channels, 2)]
            for _ in range(repeats - 1):
                seq.append(InvertedResidual(output_channels, output_channels, 1))
            setattr(self, name, nn.Sequential(*seq))
            input_channels = output_channels

        output_channels = self._stage_out_channels[-1]
        self.conv5 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )

        # load state
        self.load_state_dict(weights.get_state_dict(progress=True), strict=False)

    def forward(self, x: Tensor) -> Tensor:
        outs = []

        x = self.conv1(x)     # 2x    channels=24
        x = self.maxpool(x)   # 4x    channels=24
        outs.append(x)

        x = self.stage2(x)    # 8x    channels=116    
        outs.append(x)

        x = self.stage3(x)    # 16x   channels=232
        outs.append(x)

        x = self.stage4(x)    # 32x   channels=464
        x = self.conv5(x)     # 32x   channels=1024
        outs.append(x)

        return tuple(outs)




if __name__ == '__main__':
    inputs = torch.ones([32, 3, 384, 640])

    model = ShuffleNetV2(depth='x1.0')
    model(inputs)