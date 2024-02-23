# -*- coding: utf-8 -*-
# @Author  : ace
# Copyright by Ace, All Rights Reserved. 

import math
import random
import numpy as np

from numpy import random
from copy import deepcopy
from typing import Optional, Union, Tuple

from mmcv.transforms import BaseTransform
from mmcv.transforms.utils import cache_randomness

from mmdet.structures.bbox import autocast_box_type

from ..registry import TRANSFORMS


@TRANSFORMS.register_module()
class AlchemyRandomCrop(BaseTransform):
    def __init__(self, crop_ratios: Union[tuple, float] = (0.5, 0.8), center_crop: bool = True, keep_aspect: bool = False, prob: float = 0.5):
        super().__init__()
        self.crop_ratios = crop_ratios
        self.center_crop = center_crop
        self.keep_aspect = keep_aspect
        self.prob = prob
    
    def _center_crop(self, img_size: Union[list, tuple], crop_size: Union[list, tuple]) -> Tuple[int, int]:
        """
        计算crop区域左上角的坐标
        """
        # 原始图像中心点坐标
        ctx_img = math.ceil(img_size[1] / 2)
        cty_img = math.ceil(img_size[0] / 2)

        # crop区域的坐标
        offset_w = math.ceil(ctx_img - crop_size[1] / 2)
        offset_h = math.ceil(cty_img - crop_size[0] / 2)

        return (offset_h, offset_w)
    
    def _random_crop(self, img_size: Union[list, tuple], crop_size: Union[list, tuple]) -> Tuple[int, int]:
        """
        计算crop区域左上角的坐标
        """
        offset_w = random.randint(0, img_size[1] - crop_size[1] - 1)
        offset_h = random.randint(0, img_size[0] - crop_size[0] - 1)

        return (offset_h, offset_w)

    @cache_randomness
    def _random_prob(self) -> float:
        return random.uniform(0, 1)

    @cache_randomness
    def _get_crop_size(self, img_size: Union[list, tuple]) -> Tuple[tuple, tuple]:
        """
        计算crop区域的坐标

        Args:
            img_size (tuple | list): 原始图片的尺寸 [h, w]

        Returns:
            crop区域的坐标[x1, y1, x2, y2]
        """

        # 计算crop size
        ratio_range = np.arange(self.crop_ratios[0], self.crop_ratios[1], step=0.05)

        if self.keep_aspect:
            ratio = np.random.choice(ratio_range)
            ratios = (ratio, ratio)
        else:
            h_crop_ratio = np.random.choice(ratio_range)
            w_crop_ratio = np.random.choice(ratio_range)
            ratios = (h_crop_ratio, w_crop_ratio)

        crop_size = (math.ceil(img_size[0] * ratios[0]), math.ceil(img_size[1] * ratios[1]))

        # 获取crop区域的左上角点
        if self.center_crop:
            offsets = self._center_crop(img_size, crop_size)
        else:
            if self._random_prob() < 0.3:
                offsets = self._center_crop(img_size, crop_size)
            else:
                offsets = self._random_crop(img_size, crop_size)

        return crop_size, offsets

    @staticmethod
    def _crop_bboxes(results: dict, offsets: Tuple[int, int], img_shape: Tuple[int, int, int]):
        """
        crop bboxes

        Args:
            results (dict): Result dict from loading pipeline.
            offsets (tuple): crop区域相对于原图的偏移值 [h, w]
            img_shape (tuple): crop后图像的尺寸

        """
        if results.get('gt_bboxes', None) is not None:
            bboxes = results['gt_bboxes']
            bboxes.translate_([-offsets[1], -offsets[0]])
            bboxes.clip_(img_shape[:2])

            # 过滤无效目标
            valid_inds = bboxes.is_inside(img_shape[:2]).numpy()

            if not valid_inds.any():
                return None
            
            results['gt_bboxes'] = bboxes[valid_inds]
            if results.get('gt_ignore_flags', None) is not None:
                results['gt_ignore_flags'] = results['gt_ignore_flags'][valid_inds]

            if results.get('gt_bboxes_labels', None) is not None:
                results['gt_bboxes_labels'] = results['gt_bboxes_labels'][valid_inds]
        
        return results

    @autocast_box_type()
    def transform(self, results: dict) -> Optional[dict]:
        if self._random_prob() < self.prob:
            raw_results = deepcopy(results)

            img = results['img']
            img_h, img_w = img.shape[:2]

            crop_size, offsets = self._get_crop_size((img_h, img_w))

            # crop the image
            crop_y1, crop_y2 = offsets[0], offsets[0] + crop_size[0]
            crop_x1, crop_x2 = offsets[1], offsets[1] + crop_size[1]

            img = img[crop_y1: crop_y2, crop_x1: crop_x2, ...]
            img_shape = img.shape

            # crop bboxes 如果crop后没有有效目标, 直接返回原始results
            results_ = self._crop_bboxes(results, offsets, img_shape)
            if results_ is None:
                return raw_results

            results_['img'] = img
            results_['img_shape'] = img_shape[:2]

            # Record the homography matrix for the crop
            homography_matrix = np.array([
                [1, 0, -offsets[1]], 
                [0, 1, -offsets[0]], 
                [0, 0, 1]
                ], dtype=np.float32)

            if results_.get('homography_matrix', None) is None:
                results_['homography_matrix'] = homography_matrix
            else:
                results_['homography_matrix'] = homography_matrix @ results_['homography_matrix']

            return results_
        else:
            return results
    
    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(crop_ratios={self.crop_ratios}, '
        repr_str += f'center_crop={self.center_crop}, '
        repr_str += f'prob={self.prob})'

        return repr_str
    
    def _vis(self, results):
        import os
        import cv2
        crop_img = results['img'].copy().astype(np.uint8)
        bboxes = results['gt_bboxes']
        for bbox in bboxes:
            x1, y1, x2, y2 = [int(x) for x in bbox.tensor.reshape(-1)]
            cv2.rectangle(crop_img, (x1, y1), (x2, y2), (0, 255, 0), 1)
        
        cv2.imwrite(os.path.join('/raid/xinjin_ai/ace/models/Ace-AlchemyFurnace/alchemy/transforms', f'{results["img_id"]}_crop.jpg'), crop_img)
        exit()