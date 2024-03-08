# -*- coding: utf-8 -*-
# @Author  : ace
# Copyright by Ace, All Rights Reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import Tuple, List, Dict

from mmcv.cnn import ConvModule

from mmengine.structures import InstanceData
from mmengine.model import bias_init_with_prob, normal_init

from mmdet.utils import ConfigType, OptMultiConfig, InstanceList, OptInstanceList, OptConfigType
from mmdet.models.utils import gaussian_radius, gen_gaussian_target, multi_apply, transpose_and_gather_feat

from ...registry import MODELS
from .anchor_free_head import AnchorFreeDet2dHead


@MODELS.register_module()
class AlchemyCenterNet(AnchorFreeDet2dHead):
    def __init__(self, 
                 in_channels: int,
                 num_classes: int,
                 num_convs: int = 2,
                 feat_channels: int = 128,
                 class_agnostic: bool = True,
                 loss_wh: ConfigType = dict(type='mmdet.L1Loss', loss_weight=0.1),
                 loss_offset: ConfigType = dict(type='mmdet.L1Loss', loss_weight=1.0),
                 loss_heatmap: ConfigType = dict(type='mmdet.GaussianFocalLoss', loss_weight=1.0),
                 conv_cfg: OptConfigType = None,
                 norm_cfg: OptConfigType = None,
                 test_cfg: OptMultiConfig = None,
                 init_cfg: OptMultiConfig = None,
                 train_cfg: OptMultiConfig = None) -> None:
        # args
        self.num_convs = num_convs
        self.feat_channels = feat_channels
        self.class_agnostic = class_agnostic
        self.wh_dim = 2 if self.class_agnostic else 2 * num_classes

        super().__init__(init_cfg=init_cfg, test_cfg=test_cfg, train_cfg=train_cfg, in_channels=in_channels, num_classes=num_classes, conv_cfg=conv_cfg, norm_cfg=norm_cfg)

        # init loss
        self.loss_wh = MODELS.build(loss_wh)
        self.loss_offset = MODELS.build(loss_offset)
        self.loss_heatmap = MODELS.build(loss_heatmap)

        # args
        self.topk = 100 if self.test_cfg is None else self.test_cfg.get('topk', 100)
        self.down_ratio = 4 if self.train_cfg is None else self.train_cfg.get('down_ratio', 4)
        self.local_maximum_kernel = 3 if self.test_cfg is None else self.test_cfg.get('local_maximum_kernel', 3)

    def initial_head(self) -> None:
        self.offset_head = self._build_head(self.num_convs, self.feat_channels, 2)
        self.wh_head = self._build_head(self.num_convs, self.feat_channels, self.wh_dim)
        self.heatmap_head = self._build_head(self.num_convs, self.feat_channels, self.num_classes)
            
    def init_weights(self) -> None:
        super().init_weights()

        bias_init = bias_init_with_prob(0.1)
        self.heatmap_head[-1].bias.data.fill_(bias_init)
        
        for head in [self.wh_head, self.offset_head]:
            for m in head.modules():
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.001)

    def forward(self, img_feats: Tuple[Tensor, ...], **kwargs) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Forward features.

        Args:
            img_feats (tuple[Tensor]): Features from the upstream network, each is a 4D-tensor.

        Returns:
            wh_pred (Tensor): wh predicts, the channels number is wh_dim.
            offset_pred (Tensor): center offset predicts, the channels number is 2.
            heatmap_pred (Tensor): center predict heatmaps, the channels number is num_classes.
        """
        feat = img_feats[0]

        wh_pred = self.wh_head(feat)
        offset_pred = self.offset_head(feat)
        heatmap_pred = self.heatmap_head(feat).sigmoid()

        return (wh_pred, offset_pred, heatmap_pred)
    
    def loss_by_feats(self, 
                      wh_pred: Tensor, 
                      offset_pred: Tensor, 
                      heatmap_pred: Tensor, 
                      batch_gt_instances: InstanceList, 
                      batch_img_metas: List[dict], 
                      batch_gt_instances_ignore: OptInstanceList = None) -> Dict[str, Tensor]:
        """
        Calculate the loss based on the features extracted by the detection head.

        Args:
            hetmap_pred (Tensor): center predict heatmaps with shape (B, num_classes, H, W).
            wh_pred (list[Tensor]): wh predicts for all levels with shape (B, wh_dim, H, W).
            offset_pred (list[Tensor]): offset predicts with shape (B, 2, H, W).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of gt_instance. It usually includes 'bboxes' and 'labels' attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g., image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], optional): Batch of gt_instances_ignore. It includes 'bboxes' attribute data that is ignored during training and testing. Defaults to None.
        
        Returns:
            dict of losses, include:
                heatmap loss
                wh loss
                offset loss
        """
        # generate ground-truth
        heatmap_targets, wh_targets, wh_target_weights, offset_targets, offset_target_weights = self._build_targets(
            featmap_size=heatmap_pred.shape[2:], 
            batch_gt_instances=batch_gt_instances
            )

        # heatmap loss
        heatmap_avg_factor = max(heatmap_targets.eq(1).sum(), 1)
        heatmap_pred = torch.clamp(heatmap_pred, min=1e-4, max=1 - 1e-4)
        loss_center_heatmap = self.loss_heatmap(heatmap_pred, heatmap_targets, avg_factor=heatmap_avg_factor)

        # wh loss
        wh_avg_factors = max(wh_target_weights.eq(1).sum(), 1)
        loss_wh = self.loss_wh(wh_pred, wh_targets, wh_target_weights, avg_factor=wh_avg_factors)

        # offset loss
        offset_avg_factors = max(offset_target_weights.eq(1).sum(), 1)
        loss_offset = self.loss_offset(offset_pred, offset_targets, offset_target_weights, avg_factor=offset_avg_factors)

        return dict(
            loss_heatmap=loss_center_heatmap,
            loss_wh=loss_wh,
            loss_offset=loss_offset
            )
    
    def predict_by_feats(self, wh_pred: Tensor, offset_pred: Tensor, heatmap_pred: Tensor, batch_img_metas: List[dict], rescale: bool = False, **kwargs) -> List[Tensor]:
        # decode
        topk_bboxes, topk_scores, topk_labels = self._heatmap_decoding(wh_pred, offset_pred, heatmap_pred)

        # post process
        result_list = []

        for idx in range(topk_bboxes.shape[0]):
            scores = topk_scores[idx]
            keep = scores > self.score_thr

            if sum(keep):
                img_meta = batch_img_metas[idx]
                det_bboxes = topk_bboxes[idx][keep]

                # rescale -> raw
                if rescale and 'scale_factor' in img_meta:
                    det_bboxes[..., :4] /= det_bboxes.new_tensor(img_meta['scale_factor']).repeat((1, 2))

                det_scores = scores[keep]
                det_labels = topk_labels[idx][keep]
            else:
                det_bboxes = topk_bboxes.new_zeros((0, 4))
                det_scores = topk_bboxes.new_zeros((0, ))
                det_labels = topk_bboxes.new_zeros((0, ))

            result = InstanceData()
            result.bboxes = det_bboxes
            result.scores = det_scores
            result.labels = det_labels

            result_list.append(result)
        
        return result_list
    
    def _build_head(self, num_convs, feat_channels, out_channels) -> nn.Sequential:
        """
        Build CenterNet head.

        Args:
            num_convs (int): The number of conv layers.
            feat_channels (int): the channels of hidden layer.
            out_channels (int): The channels of output layer.
        
        Return:
            The Sequential of head layer.
        """
        head_convs = [ConvModule(self.in_channels, feat_channels, 3, padding=1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg)]
        
        for _ in range(num_convs - 1):
            head_convs.append(ConvModule(feat_channels, feat_channels, 3, padding=1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg))

        head_convs.append(nn.Conv2d(feat_channels, out_channels, 1))

        return nn.Sequential(*head_convs)
    
    def _build_targets(self, featmap_size: Tuple[int, int], batch_gt_instances: InstanceList) -> Tuple[Tensor, ...]:
        """
        Generate regression and classification targets in multiple images.

        Args:
            batch_gt_instances: InstanceList
            feat_shapes (tuple): feature map shape with value [H, W]

        Returns:
            the dict ponents below:
               - center_heatmap_target (Tensor): targets of center heatmap, shape (B, num_classes, H, W).
               - wh_target (Tensor): targets of wh predict, shape (B, wh_dim, H, W).
               - wh_target_weight (Tensor): weights of wh predict, shape (B, wh_dim, H, W).
               - offset_target (Tensor): targets of offset predict, shape (B, 2, H, W).
               - offset_target_weight (Tensor): weights of and offset predict, shape (B, 2, H, W).
        """
        heatmap_targets, wh_targets, wh_target_weights, offset_targets, offset_target_weights = multi_apply(
            self._build_targets_single,
            batch_gt_instances,
            featmap_size=featmap_size
            )
        
        heatmap_targets, wh_targets, wh_target_weights, offset_targets, offset_target_weights = [
            torch.stack(t, dim=0).detach() for t in [
                heatmap_targets,
                wh_targets,
                wh_target_weights,
                offset_targets,
                offset_target_weights
            ]
        ]

        return (heatmap_targets, wh_targets, wh_target_weights, offset_targets, offset_target_weights)
    
    def _build_targets_single(self, gt_instances: InstanceData, featmap_size: Tuple[int, int]) -> Tuple[Tensor, ...]:
        """
        Generate regression and classification targets in single image.

        Args:
            gt_instances (InstanceData):
            featmap_size (tuple): feature map shape with value [H, W]

        Returns:
            has components below:
               - center_heatmap_target (Tensor): targets of center heatmap, shape (num_classes, H, W).
               - wh_target (Tensor): targets of wh predict, shape (wh_dim, H, W).
               - wh_target_weight (Tensor): weights of wh predict, shape (wh_dim, H, W).
               - offset_target (Tensor): targets of offset predict, shape (2, H, W).
               - offset_target_weight (Tensor): weights of offset predict, shape (2, H, W).
        """
        feat_h, feat_w = featmap_size
        gt_bboxes = gt_instances.bboxes
        gt_labels = gt_instances.labels

        # init targets
        offset_target = gt_bboxes.new_zeros((2, feat_h, feat_w))
        offset_target_weight = gt_bboxes.new_zeros((2, feat_h, feat_w))

        wh_target = gt_bboxes.new_zeros((self.wh_dim, feat_h, feat_w))
        wh_target_weight = gt_bboxes.new_zeros((self.wh_dim, feat_h, feat_w))

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
            if self.class_agnostic:
                wh_target[0, cty_int, ctx_int] = obj_w
                wh_target[1, cty_int, ctx_int] = obj_h
                wh_target_weight[:, cty_int, ctx_int] = 1
            else:
                s = int(cat_id * 2)
                wh_target[s, cty_int, ctx_int] = obj_w
                wh_target[s + 1, cty_int, ctx_int] = obj_h
                wh_target_weight[s: s + 2, cty_int, ctx_int] = 1

            # offset
            offset_target[0, cty_int, ctx_int] = ctx - ctx_int
            offset_target[1, cty_int, ctx_int] = cty - cty_int
            offset_target_weight[:, cty_int, ctx_int] = 1

        return heatmap_target, wh_target, wh_target_weight, offset_target, offset_target_weight

    def _heatmap_decoding(self, wh_pred: Tensor, offset_pred: Tensor, heatmap_pred: Tensor) -> Tuple[Tensor, ...]:
        # simple nms
        pad = (self.local_maximum_kernel - 1) // 2
        hmax = F.max_pool2d(heatmap_pred, self.local_maximum_kernel, stride=1, padding=pad)
        keep = (hmax == heatmap_pred).float()
        heatmap_pred = heatmap_pred * keep

        # topk
        batch_size, num_classes, output_h, output_w = heatmap_pred.shape
        flatten_dim = int(output_w * output_h)

        topk_scores, topk_indexes = torch.topk(heatmap_pred.view(batch_size, -1), self.topk)
        topk_scores = topk_scores.view(batch_size, self.topk)

        topk_labels = torch.div(topk_indexes, flatten_dim, rounding_mode="trunc")
        topk_labels = topk_labels.view(batch_size, self.topk)                          # [n, topk]

        topk_indexes = topk_indexes % flatten_dim
        topk_ys = torch.div(topk_indexes, output_w, rounding_mode="trunc").to(torch.float32)
        topk_xs = (topk_indexes % output_w).to(torch.float32)

        # 中心点偏移
        topk_offsets = transpose_and_gather_feat(offset_pred, topk_indexes)   # [n, topk, 2]
        topk_ctxs = topk_xs + topk_offsets[..., 0]
        topk_ctys = topk_ys + topk_offsets[..., 1]

        # wh
        topk_whs = transpose_and_gather_feat(wh_pred, topk_indexes)           # [n, topk, 2 if class_agnostic else 2 * num_classes]
        if not self.class_agnostic:
            topk_whs = topk_whs.view(batch_size, self.topk, num_classes, 2)
            classes_ind = topk_labels.view(batch_size, self.topk, 1, 1).expand(batch_size, self.topk, 1, 2).long()
            topk_whs = topk_whs.gather(2, classes_ind).view(batch_size, self.topk, 2)

        # bbox
        x1s = (topk_ctxs - topk_whs[..., 0] / 2)
        y1s = (topk_ctys - topk_whs[..., 1] / 2)
        x2s = (topk_ctxs + topk_whs[..., 0] / 2)
        y2s = (topk_ctys + topk_whs[..., 1] / 2)

        topk_bboxes = torch.stack([x1s, y1s, x2s, y2s], dim=2) * self.down_ratio

        return topk_bboxes, topk_scores, topk_labels 
    