# -*- coding: utf-8 -*-
# @Author  : ace
# Copyright by Ace, All Rights Reserved. 

import cv2
import math
import numpy as np

from typing import Optional, Union

from mmcv.transforms import BaseTransform

from mmdet.structures.bbox import autocast_box_type

from ..registry import TRANSFORMS


@TRANSFORMS.register_module()
class AlchemyLetterBoxFixed(BaseTransform):
    def __init__(self, input_size: Union[tuple, int] = (512, 512)):
        super().__init__()
        self.input_sizes = input_size
    
    def _resize_img(self, results: dict) -> None:
        if results.get('img', None) is not None:
            raw_img_h, raw_img_w = results['img'].shape[:2]

            sacle = min(self.input_sizes[0] / raw_img_h, self.input_sizes[1] / raw_img_w)

            img = results['img']

            if sacle != 1.0:
                img = cv2.resize(img, (math.ceil(raw_img_w * sacle), math.ceil(raw_img_h * sacle)), interpolation=cv2.INTER_LINEAR)
            
            results['img'] = img
            results['img_shape'] = img.shape[:2]
            results['scale_factor'] = (sacle, sacle)

            # 记录转换矩阵
            homography_matrix = np.array([
                [sacle, 0, 0], 
                [0, sacle, 0], 
                [0, 0, 1]], dtype=np.float32)

            if results.get('homography_matrix', None) is None:
                results['homography_matrix'] = homography_matrix
            else:
                results['homography_matrix'] = homography_matrix @ results['homography_matrix']

    def _resize_bbox(self, results: dict) -> None:
        if results.get('gt_bboxes', None) is not None:
            results['gt_bboxes'].rescale_(results['scale_factor'])
    
    @autocast_box_type()
    def transform(self, results: dict) -> Optional[dict]:
        # img
        self._resize_img(results)

        # bbox
        self._resize_bbox(results)
        
        return results
    
    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(input_sizes={self.input_sizes})'

        return repr_str
    
    def _vis(self, results):
        import os
        import cv2
        img_id = results['img_id']
        crop_img = results['img'].copy().astype(np.uint8)
        bboxes = results['gt_bboxes']

        for bbox in bboxes:
            x1, y1, x2, y2 = [int(x) for x in bbox.tensor.reshape(-1)]
            cv2.rectangle(crop_img, (x1, y1), (x2, y2), (0, 255, 0), 1)
        
        cv2.imwrite(os.path.join('/raid/xinjin_ai/ace/models/Ace-AlchemyFurnace/alchemy/transforms', f'{img_id}_resize.jpg'), crop_img)
        exit()

