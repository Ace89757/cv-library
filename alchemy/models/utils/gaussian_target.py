# -*- coding: utf-8 -*-
# @Author  : ace
# Copyright by Ace, All Rights Reserved.  

import torch
import numpy as np


def gaussian2d(shape, sigma_x=1, sigma_y=1):
    """
    Generate 2D gaussian kernel.

    Args:
        radius (int): Radius of gaussian kernel.
        sigma_x (int): Sigma X of gaussian function. Default: 1.
        sigma_y (int): Sigma Y of gaussian function. Default: 1.

    Returns:
        h (Tensor): Gaussian kernel with a '(2 * radius + 1) * (2 * radius + 1)' shape.
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x / (2 * sigma_x * sigma_x) + y * y / (2 * sigma_y * sigma_y)))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def gen_truncate_gaussian_target(heatmap, center, h_radius, w_radius, k=1):
        """
        Generate 2D gaussian heatmap.

        Args:
            heatmap (Tensor): Input heatmap, the gaussian kernel will cover on it and maintain the max value.
            center (list[int]): Coord of gaussian kernel's center.
            h_radius (int): Radius h of gaussian kernel.
            w_radius (int): Radius w of gaussian kernel.
            k (int): Coefficient of gaussian kernel. Default: 1.

        Returns:
            out_heatmap (Tensor): Updated heatmap covered by gaussian kernel.
        """
        h, w = 2 * h_radius + 1, 2 * w_radius + 1
        sigma_x = w / 6
        sigma_y = h / 6
        gaussian = gaussian2d((h, w), sigma_x=sigma_x, sigma_y=sigma_y)
        gaussian = heatmap.new_tensor(gaussian)

        x, y = int(center[0]), int(center[1])

        height, width = heatmap.shape[0:2]

        left, right = min(x, w_radius), min(width - x, w_radius + 1)
        top, bottom = min(y, h_radius), min(height - y, h_radius + 1)

        masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
        masked_gaussian = gaussian[h_radius - top:h_radius + bottom, w_radius - left:w_radius + right]

        if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
            torch.max(masked_heatmap, masked_gaussian * k, out=masked_heatmap)

        return heatmap