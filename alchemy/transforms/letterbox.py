# -*- coding: utf-8 -*-
# @Author  : ace
# Copyright by Ace, All Rights Reserved. 

import cv2
import math
import numpy as np

from typing import Optional, Union, Tuple, Any

from mmcv.transforms import BaseTransform

from mmdet.structures.bbox import autocast_box_type

from ..registry import TRANSFORMS


CV2_INTERP_CODES = {
    'nearest': cv2.INTER_NEAREST,
    'bilinear': cv2.INTER_LINEAR,
    'bicubic': cv2.INTER_CUBIC,
    'area': cv2.INTER_AREA,
    'lanczos': cv2.INTER_LANCZOS4
}


@TRANSFORMS.register_module()
class AlchemyLetterBox(BaseTransform):
    def __init__(self,
                 keep_aspect: bool = True,
                 random_interp: bool = False,
                 scale_ratios: Tuple[float] = (1., ),
                 input_size: Union[tuple, int] = (512, 512)):
        super().__init__()

        self.keep_aspect = keep_aspect

        # 为了适配多尺度图像, 最终的输入尺寸为input_size * scale_ratio
        input_size = (input_size, input_size) if isinstance(input_size, int) else input_size
        self.input_sizes = [(self._scale_size(input_size[0], ratio), self._scale_size(input_size[1], ratio)) for ratio in scale_ratios]

        if random_interp:
            self.interp_methods = ['nearest', 'bilinear', 'bicubic', 'area', 'lanczos']
        else:
            self.interp_methods = ['bilinear']
    
    @staticmethod
    def _scale_size(input_size: int, scale: float) -> int:
        """
        保证尺寸缩放后能被32整除
        """
        out_size = int(input_size * scale)

        if out_size % 32 != 0:  # 保证被32整除
            out_size = int((out_size // 32 + 1) * 32)
        
        return out_size
    
    def _random_input_size(self) -> Tuple[int, int]:
        """
        Randomly select an scale from given candidates.

        Returns:
            (tuple, int): input_size
        """

        index = np.random.randint(len(self.input_sizes))
        
        return self.input_sizes[index]
    
    def _random_interp_method(self) -> Any:
        """Randomly select an scale from given candidates.

        Returns:
            (tuple, int): interp methods
        """

        index = np.random.randint(len(self.interp_methods))
        
        return CV2_INTERP_CODES[self.interp_methods[index]]
    
    def _resize_img(self, results: dict) -> None:
        if results.get('img', None) is not None:
            raw_img_h, raw_img_w = results['img'].shape[:2]

            input_h, input_w = self._random_input_size()

            if self.keep_aspect:
                scale_h = scale_w = min(input_h / raw_img_h, input_w / raw_img_w)
            else:
                scale_h = input_h / raw_img_h
                scale_w = input_w / raw_img_w

            img = results['img']

            if scale_h != 1.0 or scale_w != 1.0:
                interpolation = self._random_interp_method()
                img = cv2.resize(img, (math.ceil(raw_img_w * scale_w), math.ceil(raw_img_h * scale_h)), interpolation=interpolation)
            
            results['img'] = img
            results['img_shape'] = img.shape[:2]
            results['scale_factor'] = (scale_w, scale_h)

            # 记录转换矩阵
            homography_matrix = np.array([
                [scale_w, 0, 0], 
                [0, scale_h, 0], 
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
        repr_str += f'(input_sizes=[{self.input_sizes}], '
        repr_str += f'keep_aspect={self.keep_aspect}, '
        repr_str += f'interp_method={self.interp_methods})'

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

