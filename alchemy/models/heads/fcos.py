# -*- coding: utf-8 -*-
# @Author  : ace
# Copyright by Ace, All Rights Reserved.

from typing import Optional

from mmcv.ops import batched_nms

from mmengine.config import ConfigDict
from mmengine.structures import InstanceData

from mmdet.models.dense_heads.fcos_head import FCOSHead
from mmdet.structures.bbox import (get_box_tensor, get_box_wh, scale_boxes)

from ...registry import MODELS


@MODELS.register_module()
class AlchemyFCOSHead(FCOSHead):
    def _bbox_post_process(self, results: InstanceData, cfg: ConfigDict, rescale: bool = False, with_nms: bool = True, img_meta: Optional[dict] = None) -> InstanceData:
        """
        bbox post-processing method.

        The boxes would be rescaled to the original image scale and do the nms operation. Usually `with_nms` is False is used for aug test.

        Args:
            results (:obj:`InstaceData`): Detection instance results, each item has shape (num_bboxes, ).
            cfg (ConfigDict): Test / postprocessing configuration, if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space. Default to False.
            with_nms (bool): If True, do nms before return boxes. Default to True.
            img_meta (dict, optional): Image meta info. Defaults to None.

        Returns:
            :obj:`InstanceData`: Detection results of each image after the post process.
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4), the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        if rescale:
            assert img_meta.get('scale_factor') is not None
            scale_factor = [1 / s for s in img_meta['scale_factor']]
            results.bboxes = scale_boxes(results.bboxes, scale_factor)

        if hasattr(results, 'score_factors'):
            # FIXME: Add sqrt operation in order to be consistent with the paper.
            score_factors = results.pop('score_factors')
            results.scores = (results.scores * score_factors).sqrt()  # sqrt提高得分

        # filter small size bboxes
        if cfg.get('min_bbox_size', -1) >= 0:
            w, h = get_box_wh(results.bboxes)
            valid_mask = (w > cfg.min_bbox_size) & (h > cfg.min_bbox_size)
            if not valid_mask.all():
                results = results[valid_mask]

        if with_nms and results.bboxes.numel() > 0:
            bboxes = get_box_tensor(results.bboxes)

            nms_cfg = cfg.get('nms', dict(type='nms', iou_threshold=0.5))
            det_bboxes, keep_idxs = batched_nms(bboxes, results.scores, results.labels, nms_cfg)

            results = results[keep_idxs]

            # some nms would reweight the score, such as softnms
            results.scores = det_bboxes[:, -1]
            results = results[:cfg.max_per_img]

        return results