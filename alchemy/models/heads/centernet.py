# -*- coding: utf-8 -*-
# @Author  : ace
# Copyright by Ace, All Rights Reserved.

import torch

from torch import Tensor
from typing import Tuple, List, Optional

from mmengine.structures import InstanceData

from mmdet.utils import InstanceList
from mmdet.models.dense_heads.centernet_head import CenterNetHead
from mmdet.models.utils import gaussian_radius, gen_gaussian_target, multi_apply, transpose_and_gather_feat, get_local_maximum, get_topk_from_heatmap

from ...registry import MODELS


@MODELS.register_module()
class AlchemyCenterNetHead(CenterNetHead):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # args
        self.down_ratio = 4 if self.train_cfg is None else self.train_cfg.get('down_ratio', 4)
        self.score_thr = 0.1 if self.test_cfg is None else self.test_cfg.get('score_thr', 0.1)
    
    def get_targets(self, gt_bboxes: List[Tensor], gt_labels: List[Tensor], feat_shape: tuple, img_shape: tuple) -> Tuple[dict, int]:
        """
        Compute regression and classification targets in multiple images.

        Args:
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box.
            feat_shape (tuple): feature map shape with value [B, _, H, W]
            img_shape (tuple): image shape.

        Returns:
            tuple[dict, float]: The float value is mean avg_factor, the dict
            has components below:
               - center_heatmap_target (Tensor): targets of center heatmap, shape (B, num_classes, H, W).
               - wh_target (Tensor): targets of wh predict, shape (B, 2, H, W).
               - offset_target (Tensor): targets of offset predict, shape (B, 2, H, W).
               - wh_offset_target_weight (Tensor): weights of wh and offset predict, shape (B, 2, H, W).
        """
        feat_h, feat_w = feat_shape[2:]

        center_heatmap_target, wh_target, offset_target, wh_offset_target_weight = multi_apply(
            self._build_targets_single,
            gt_bboxes,
            gt_labels,
            featmap_size=(feat_h, feat_w)
            )
        
        center_heatmap_target, wh_target, offset_target, wh_offset_target_weight = [
            torch.stack(t, dim=0).detach() for t in [
                center_heatmap_target,
                wh_target,
                offset_target,
                wh_offset_target_weight
            ]
        ]

        avg_factor = max(1, center_heatmap_target.eq(1).sum())

        target_result = dict(
            center_heatmap_target=center_heatmap_target,
            wh_target=wh_target,
            offset_target=offset_target,
            wh_offset_target_weight=wh_offset_target_weight
        )

        return target_result, avg_factor
    
    def _build_targets_single(self, gt_bboxes: Tensor, gt_labels: Tensor, featmap_size: Tuple[int, int]) -> Tuple[Tensor, ...]:
        """
        Generate regression and classification targets in single image.

        Args:
            gt_bboxes (Tensor): Ground truth bboxes for each image with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (Tensor): class indices corresponding to each box.
            featmap_size (tuple): feature map shape with value [H, W]

        Returns:
            has components below:
               - center_heatmap_target (Tensor): targets of center heatmap, shape (num_classes, H, W).
               - wh_target (Tensor): targets of wh predict, shape (2, H, W).
               - offset_target (Tensor): targets of offset predict, shape (2, H, W).
               - offset_target_weight (Tensor): weights of offset predict, shape (2, H, W).
        """
        feat_h, feat_w = featmap_size

        # init targets
        wh_target = gt_bboxes.new_zeros((2, feat_h, feat_w))
        offset_target = gt_bboxes.new_zeros((2, feat_h, feat_w))       
        wh_off_target_weight = gt_bboxes.new_zeros((2, feat_h, feat_w))

        heatmap_target = gt_bboxes.new_zeros((self.num_classes, feat_h, feat_w))
        
        # down sample
        feat_bboxes = gt_bboxes / self.down_ratio
        feat_bboxes[:, [0, 2]] = torch.clamp(feat_bboxes[:, [0, 2]], min=0, max=feat_w - 1)
        feat_bboxes[:, [1, 3]] = torch.clamp(feat_bboxes[:, [1, 3]], min=0, max=feat_h - 1)
        feat_bboxes_hs = feat_bboxes[:, 3] - feat_bboxes[:, 1]
        feat_bboxes_ws = feat_bboxes[:, 2] - feat_bboxes[:, 0]

        feat_bboxes_ctxs = (feat_bboxes[:, 0] + feat_bboxes[:, 2]) / 2
        feat_bboxes_ctxs_int = (feat_bboxes_ctxs).to(torch.int32)

        feat_bboxes_ctys = (feat_bboxes[:, 1] + feat_bboxes[:, 3]) / 2
        feat_bboxes_ctys_int = (feat_bboxes_ctys).to(torch.int32)

        for idx in range(feat_bboxes.shape[0]):
            obj_h, obj_w = feat_bboxes_hs[idx].item(), feat_bboxes_ws[idx].item()
            
            cat_id = gt_labels[idx]
            ctx, cty = feat_bboxes_ctxs[idx].item(), feat_bboxes_ctys[idx].item()
            ctx_int, cty_int = feat_bboxes_ctxs_int[idx].item(), feat_bboxes_ctys_int[idx].item()

            # heatmap
            radius = gaussian_radius([obj_h, obj_w],  min_overlap=0.3)
            radius = max(0, int(radius))
            gen_gaussian_target(heatmap_target[cat_id], [ctx_int, cty_int], radius)

            # wh
            wh_target[0, cty_int, ctx_int] = obj_w
            wh_target[1, cty_int, ctx_int] = obj_h
            
            # offset
            offset_target[0, cty_int, ctx_int] = ctx - ctx_int
            offset_target[1, cty_int, ctx_int] = cty - cty_int

            # weight
            wh_off_target_weight[:, cty_int, ctx_int] = 1

        return heatmap_target, wh_target, offset_target, wh_off_target_weight 
    
    def _decode_heatmap(self, center_heatmap_pred: Tensor, wh_pred: Tensor, offset_pred: Tensor, k: int = 100, kernel: int = 3) -> Tuple[Tensor, Tensor]:
        """Transform outputs into detections raw bbox prediction.

        Args:
            center_heatmap_pred (Tensor): center predict heatmap, shape (B, num_classes, H, W).
            wh_pred (Tensor): wh predict, shape (B, 2, H, W).
            offset_pred (Tensor): offset predict, shape (B, 2, H, W).
            k (int): Get top k center keypoints from heatmap. Defaults to 100.
            kernel (int): Max pooling kernel for extract local maximum pixels. Defaults to 3.

        Returns:
            tuple[Tensor]: Decoded output of CenterNetHead, containing the following Tensors:

              - batch_bboxes (Tensor): Coords of each box with shape (B, k, 5)
              - batch_topk_labels (Tensor): Categories of each box with  shape (B, k)
        """
        center_heatmap_pred = get_local_maximum(center_heatmap_pred, kernel=kernel)

        *batch_dets, topk_ys, topk_xs = get_topk_from_heatmap(center_heatmap_pred, k=k)
        batch_scores, batch_index, batch_topk_labels = batch_dets

        wh = transpose_and_gather_feat(wh_pred, batch_index)
        offset = transpose_and_gather_feat(offset_pred, batch_index)
        topk_xs = topk_xs + offset[..., 0]
        topk_ys = topk_ys + offset[..., 1]

        tl_x = (topk_xs - wh[..., 0] / 2)
        tl_y = (topk_ys - wh[..., 1] / 2)
        br_x = (topk_xs + wh[..., 0] / 2)
        br_y = (topk_ys + wh[..., 1] / 2)
        batch_bboxes = torch.stack([tl_x, tl_y, br_x, br_y], dim=2) * self.down_ratio
        batch_bboxes = torch.cat((batch_bboxes, batch_scores[..., None]), dim=-1)

        return batch_bboxes, batch_topk_labels
    
    def predict_by_feat(self,
                        center_heatmap_preds: List[Tensor],
                        wh_preds: List[Tensor],
                        offset_preds: List[Tensor],
                        batch_img_metas: Optional[List[dict]] = None,
                        rescale: bool = True,
                        with_nms: bool = False) -> InstanceList:
        """Transform network output for a batch into bbox predictions.

        Args:
            center_heatmap_preds (list[Tensor]): Center predict heatmaps for
                all levels with shape (B, num_classes, H, W).
            wh_preds (list[Tensor]): WH predicts for all levels with
                shape (B, 2, H, W).
            offset_preds (list[Tensor]): Offset predicts for all levels
                with shape (B, 2, H, W).
            batch_img_metas (list[dict], optional): Batch image meta info.
                Defaults to None.
            rescale (bool): If True, return boxes in original image space.
                Defaults to True.
            with_nms (bool): If True, do nms before return boxes.
                Defaults to False.

        Returns:
            list[:obj:`InstanceData`]: Instance segmentation
            results of each image after the post process.
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4), the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        assert len(center_heatmap_preds) == len(wh_preds) == len(offset_preds) == 1

        # decode
        batch_det_bboxes, batch_labels = self._decode_heatmap(center_heatmap_preds[0], wh_preds[0], offset_preds[0], k=self.test_cfg.topk, kernel=self.test_cfg.local_maximum_kernel)
        
        # post-process
        result_list = []

        for batch_id, img_meta in enumerate(batch_img_metas):
            det_bboxes = batch_det_bboxes[batch_id].view([-1, 5])
            det_labels = batch_labels[batch_id].view(-1)

            # filter by score thr
            keep = det_bboxes[:, -1] > self.score_thr
            if sum(keep):
                det_bboxes = det_bboxes[keep]
                det_labels = det_labels[keep]

                if rescale and 'scale_factor' in img_meta:
                    det_bboxes[..., :4] /= det_bboxes.new_tensor(img_meta['scale_factor']).repeat((1, 2))

                if with_nms:
                    det_bboxes, det_labels = self._bboxes_nms(det_bboxes, det_labels, self.test_cfg)
            else:
                det_bboxes = det_bboxes.new_zeros((0, 5))
                det_labels = det_bboxes.new_zeros((0, ))

            results = InstanceData()
            results.bboxes = det_bboxes[:, :-1]
            results.scores = det_bboxes[:, -1]
            results.labels = det_labels

            result_list.append(results)

        return result_list
