# -*- coding: utf-8 -*-
# @Author  : ace
# Copyright by Ace, All Rights Reserved. 

import numpy as np

from mmdet.datasets.transforms import RandomShift

from ..registry import TRANSFORMS


@TRANSFORMS.register_module()
class AlchemyRandomShift(RandomShift):
    
    def transform(self, results: dict) -> dict:
        results = super().transform(results)

        return results

    def _vis(self, results):
        import os
        import cv2
        shift_img = results['img'].copy().astype(np.uint8)
        bboxes = results['gt_bboxes']
        for bbox in bboxes:
            x1, y1, x2, y2 = [int(x) for x in bbox.tensor.reshape(-1)]
            cv2.rectangle(shift_img, (x1, y1), (x2, y2), (0, 255, 0), 1)
        
        cv2.imwrite(os.path.join('/raid/xinjin_ai/ace/models/Ace-AlchemyFurnace/alchemy/transforms', f'{results["img_id"]}_shift.jpg'), shift_img)
        exit()