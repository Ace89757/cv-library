# -*- coding: utf-8 -*-
# @Author  : ace
# Copyright by Ace, All Rights Reserved.

import torch

from torch import Tensor
from typing import Tuple

from mmengine.structures import InstanceData

from ...registry import MODELS
from ..utils import cal_bboxes_area, gen_truncate_gaussian_target
from .ttfnet import AlchemyTTFNet


@MODELS.register_module()
class AlchemyTTFNetPlus(AlchemyTTFNet):
    def _build_targets_single(self, gt_instances: InstanceData, featmap_size: Tuple[int, int]) -> Tuple[Tensor, ...]:
        """
        Generate regression and classification targets in single image.

        Args:
            gt_instances (InstanceData):
            featmap_size (tuple): feature map shape with value [H, W]

        Returns:
            has components below:
               - heatmap_target (Tensor): targets of center heatmap, shape (num_classes, H, W).
               - bbox_target (Tensor): targets of bbox predict, shape (4, H, W).
               - bbox_target_weight (Tensor): weights of bbox predict, shape (4, H, W).
        """
        feat_h, feat_w = featmap_size
        gt_bboxes = gt_instances.bboxes
        gt_labels = gt_instances.labels

        # init targets
        fake_heatmap = gt_bboxes.new_zeros((feat_h, feat_w))
        heatmap_target = gt_bboxes.new_zeros((self.num_classes, feat_h, feat_w))

        bbox_target = gt_bboxes.new_ones((4, feat_h, feat_w)) * -1
        bbox_target_weight = gt_bboxes.new_zeros((4, feat_h, feat_w))

        # 计算bbox的面积
        bbox_areas = cal_bboxes_area(gt_bboxes)

        # 减小大、小目标的影响
        if self.bbox_area_process == 'log':
            bbox_areas = bbox_areas.log()
        elif self.bbox_area_process == 'sqrt':
            bbox_areas = bbox_areas.sqrt()

        # 按面积大小排序(从大到小)
        num_objs = bbox_areas.size(0)
        bbox_areas_sorted, boxes_ind = torch.topk(bbox_areas, num_objs)

        if self.bbox_area_process == 'norm':
            bbox_areas_sorted[:] = 1.

        gt_bboxes = gt_bboxes[boxes_ind]
        gt_labels = gt_labels[boxes_ind]

        # down sample
        feat_bboxes = gt_bboxes / self.down_ratio
        feat_bboxes[:, [0, 2]] = torch.clamp(feat_bboxes[:, [0, 2]], min=0, max=feat_w - 1)
        feat_bboxes[:, [1, 3]] = torch.clamp(feat_bboxes[:, [1, 3]], min=0, max=feat_h - 1)
        feat_bboxes_hs = feat_bboxes[:, 3] - feat_bboxes[:, 1]
        feat_bboxes_ws = feat_bboxes[:, 2] - feat_bboxes[:, 0]

        # 目标的中心点(在输出图上)
        feat_ctx = (gt_bboxes[:, 0] + gt_bboxes[:, 2]) / 2 / self.down_ratio
        feat_ctx_ints = feat_ctx.to(torch.int)
        feat_ctx_offset = feat_ctx - feat_ctx_ints

        feat_cty = (gt_bboxes[:, 1] + gt_bboxes[:, 3]) / 2 / self.down_ratio
        feat_cty_ints = feat_cty.to(torch.int)
        feat_cty_offset = feat_cty - feat_cty_ints

        # 根据中心点offset判断更偏向哪个位置(制作额外的正例样本)
        feat_ctx_offset[feat_ctx_offset < 0.5] = -1
        feat_ctx_offset[feat_ctx_offset >= 0.5] = 1
        feat_cty_offset[feat_cty_offset < 0.5] = -1
        feat_cty_offset[feat_cty_offset >= 0.5] = 1

        # 目标的高斯半径
        h_radiuses_alpha = (feat_bboxes_hs / 2. * self.alpha).int()
        w_radiuses_alpha = (feat_bboxes_ws / 2. * self.alpha).int()
        if self.bbox_gaussian and self.alpha != self.beta:
            h_radiuses_beta = (feat_bboxes_hs / 2. * self.beta).int()
            w_radiuses_beta = (feat_bboxes_ws / 2. * self.beta).int()

        # 如果bbox的标签不使用高斯范围的，计算每个目标的中心区域，此区域内的点作为正例点
        if not self.bbox_gaussian:
            r1 = (1 - self.beta) / 2
            ctr_x1s, ctr_y1s, ctr_x2s, ctr_y2s = self._calc_region(gt_bboxes.transpose(0, 1), r1)

            ctr_x1s = torch.round(ctr_x1s.float() / self.down_ratio).int()
            ctr_y1s = torch.round(ctr_y1s.float() / self.down_ratio).int()
            ctr_x2s = torch.round(ctr_x2s.float() / self.down_ratio).int()
            ctr_y2s = torch.round(ctr_y2s.float() / self.down_ratio).int()

            ctr_x1s, ctr_x2s = [torch.clamp(x, max=feat_w - 1) for x in [ctr_x1s, ctr_x2s]]
            ctr_y1s, ctr_y2s = [torch.clamp(y, max=feat_h - 1) for y in [ctr_y1s, ctr_y2s]]

        for idx in range(num_objs):
            obj_h, obj_w = feat_bboxes_hs[idx].item(), feat_bboxes_ws[idx].item()

            cat_id = gt_labels[idx]
            ctx, cty = feat_ctx_ints[idx], feat_cty_ints[idx]
            
            # heatmap
            fake_heatmap = fake_heatmap.zero_()
            gen_truncate_gaussian_target(fake_heatmap, (ctx, cty), h_radiuses_alpha[idx].item(), w_radiuses_alpha[idx].item())

            # 若两个目标的高斯范围有重叠区域，选取大的值作为标签(对小目标友好)
            heatmap_target[cat_id] = torch.max(heatmap_target[cat_id], fake_heatmap)

            # bbox权重
            if self.bbox_gaussian:
                if self.alpha != self.beta:
                    fake_heatmap = fake_heatmap.zero_()
                    gen_truncate_gaussian_target(fake_heatmap, (ctx, cty), h_radiuses_beta[idx].item(), w_radiuses_beta[idx].item())

                bbox_target_inds = fake_heatmap > 0

                local_heatmap = fake_heatmap[bbox_target_inds]
                ct_div = local_heatmap.sum()
                local_heatmap *= bbox_areas_sorted[idx]
                bbox_target_weight[:, bbox_target_inds] = torch.max(bbox_target_weight[:, bbox_target_inds], local_heatmap / ct_div)
            else:
                ctr_x1, ctr_y1, ctr_x2, ctr_y2 = ctr_x1s[idx], ctr_y1s[idx], ctr_x2s[idx], ctr_y2s[idx]
                bbox_target_inds = torch.zeros_like(fake_heatmap, dtype=torch.uint8)
                bbox_target_inds[ctr_y1:ctr_y2 + 1, ctr_x1:ctr_x2 + 1] = 1
                bbox_target_weight[:, bbox_target_inds] = bbox_areas_sorted[idx] / bbox_target_inds.sum().float()
                
            # 回归的是输入图上的尺寸
            bbox_target[:, bbox_target_inds] = gt_bboxes[idx][:, None]

            # 添加额外的正样本
            extra_ctx, extra_cty = None, None

            # lb
            if obj_w > 1:
                extra_ctx = int(min(max(ctx + feat_ctx_offset[idx], 0), feat_w - 1))
                heatmap_target[cat_id, int(cty), extra_ctx] = 1

            # tb
            if obj_h > 1:
                extra_cty = int(min(max(cty + feat_cty_offset[idx], 0), feat_h - 1))
                heatmap_target[cat_id, extra_cty, int(ctx)] = 1

            # 对角线
            if (extra_ctx is not None) and (extra_cty is not None):
                heatmap_target[cat_id, extra_cty, extra_ctx] = 1
                
        return heatmap_target, bbox_target, bbox_target_weight