# -*- coding: utf-8 -*-
# @Author  : ace
# Copyright by Ace, All Rights Reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import Tuple, List, Dict

from mmcv.cnn import ConvModule
from mmcv.ops import batched_nms

from mmengine.config import ConfigDict
from mmengine.structures import InstanceData
from mmengine.model import bias_init_with_prob, normal_init

from mmdet.models.utils import multi_apply
from mmdet.models.dense_heads.base_dense_head import BaseDenseHead
from mmdet.utils import ConfigType, OptMultiConfig, InstanceList, OptInstanceList, OptConfigType

from ...registry import MODELS
from ..utils import cal_bboxes_area, gen_truncate_gaussian_target


@MODELS.register_module()
class AlchemyTTFNetHead(BaseDenseHead):
    def __init__(self,
                 in_channels: int,
                 num_classes: int,
                 beta: float = 0.54,
                 alpha: float = 0.54,
                 bbox_convs: int = 2,
                 base_anchor: int = 16,
                 bbox_channels: int = 64,
                 heatmap_convs: int = 2,
                 bbox_gaussian: bool = True,
                 heatmap_channels: int = 128,
                 bbox_area_process: str = 'log',
                 loss_bbox: ConfigType = dict(type='EfficientIoULoss', loss_weight=5.0),
                 loss_heatmap: ConfigType = dict(type='mmdet.GaussianFocalLoss', loss_weight=1.0),
                 conv_cfg: OptConfigType = None,
                 norm_cfg: OptConfigType = None,
                 test_cfg: OptMultiConfig = None,
                 init_cfg: OptMultiConfig = None,
                 train_cfg: OptMultiConfig = None) -> None:
        # args
        self.beta = beta
        self.alpha = alpha
        self.base_loc = None
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        self.bbox_convs = bbox_convs
        self.num_classes =num_classes
        self.base_anchor = base_anchor
        self.in_channels = in_channels
        self.bbox_channels = bbox_channels
        self.heatmap_convs = heatmap_convs
        self.bbox_gaussian = bbox_gaussian
        self.heatmap_channels = heatmap_channels
        self.bbox_area_process = bbox_area_process

        super().__init__(init_cfg=init_cfg)

        self.initial_head()

        # init loss
        self.loss_bbox = MODELS.build(loss_bbox)
        self.loss_heatmap = MODELS.build(loss_heatmap)

        # args
        self.topk = 100 if self.test_cfg is None else self.test_cfg.get('topk', 100)
        self.down_ratio = 4 if self.train_cfg is None else self.train_cfg.get('down_ratio', 4)
        self.score_thr = 0.1 if self.test_cfg is None else self.test_cfg.get('score_thr', 0.1)
        self.local_maximum_kernel = 3 if self.test_cfg is None else self.test_cfg.get('local_maximum_kernel', 3)

    def initial_head(self) -> None:
        self.bbox_head = self._build_head(self.bbox_convs, self.bbox_channels, 4)
        self.heatmap_head = self._build_head(self.heatmap_convs, self.heatmap_channels, self.num_classes)
            
    def init_weights(self) -> None:
        super().init_weights()
        
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.heatmap_head[-1], std=0.01, bias=bias_cls)

        for _, m in self.bbox_head.named_modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001)

    def forward(self, img_feats: Tuple[Tensor, ...], **kwargs) -> Tuple[Tensor, Tensor]:
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

        bbox_pred = F.relu(self.bbox_head(feat)) * self.base_anchor
        heatmap_pred = self.heatmap_head(feat).sigmoid()

        return (bbox_pred, heatmap_pred)
    
    def loss_by_feat(self, 
                     bbox_pred: Tensor,
                     heatmap_pred: Tensor, 
                     batch_gt_instances: InstanceList, 
                     batch_img_metas: List[dict], 
                     batch_gt_instances_ignore: OptInstanceList = None) -> Dict[str, Tensor]:
        """
        Calculate the loss based on the features extracted by the detection head.

        Args:
            hetmap_pred (Tensor): center predict heatmaps with shape (B, num_classes, H, W).
            bbox_pred (list[Tensor]): ltrb predicts for all levels with shape (B, 4, H, W).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of gt_instance. It usually includes 'bboxes' and 'labels' attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g., image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], optional): Batch of gt_instances_ignore. It includes 'bboxes' attribute data that is ignored during training and testing. Defaults to None.
        
        Returns:
            dict of losses, include:
                heatmap loss
                bbox loss
        """
        feat_h, feat_w = heatmap_pred.shape[2:]
        # generate ground-truth
        heatmap_targets, bbox_targets, bbox_target_weights= self._build_targets(
            featmap_size=(feat_h, feat_w), 
            batch_gt_instances=batch_gt_instances
            )

        # heatmap loss
        heatmap_avg_factor = max(heatmap_targets.eq(1).sum(), 1)
        heatmap_pred = torch.clamp(heatmap_pred, min=1e-4, max=1 - 1e-4)
        loss_center_heatmap = self.loss_heatmap(heatmap_pred, heatmap_targets, avg_factor=heatmap_avg_factor)

        # bbox loss
        if (self.base_loc is None) or (feat_h != self.base_loc.shape[1]) or (feat_w != self.base_loc.shape[2]):
            base_step = self.down_ratio
            shifts_x = torch.arange(0, (feat_w - 1) * base_step + 1, base_step, dtype=torch.float32, device=heatmap_pred.device)
            shifts_y = torch.arange(0, (feat_h - 1) * base_step + 1, base_step, dtype=torch.float32, device=heatmap_pred.device)
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing='ij')
            self.base_loc = torch.stack((shift_x, shift_y), dim=0)  # (2, h, w)

        pred_boxes = torch.cat((self.base_loc - bbox_pred[:, [0, 1]], self.base_loc + bbox_pred[:, [2, 3]]), dim=1).permute(0, 2, 3, 1).reshape(-1, 4)
        
        bbox_targets = bbox_targets.permute(0, 2, 3, 1).reshape(-1, 4)
        bbox_target_weights = bbox_target_weights.permute(0, 2, 3, 1).reshape(-1, 4)
        bbox_avg_factor = torch.sum(bbox_target_weights[:, 0]) + 1e-6
        loss_bbox = self.loss_bbox(pred_boxes, bbox_targets, bbox_target_weights, avg_factor=bbox_avg_factor)
        
        return dict(
            loss_heatmap=loss_center_heatmap,
            loss_bbox=loss_bbox
            )
    
    def predict_by_feat(self, bbox_pred: Tensor, heatmap_pred: Tensor, batch_img_metas: List[dict], rescale: bool = True, with_nms: bool = False) -> List[Tensor]:
        # decode
        batch_det_bboxes, batch_labels = self._heatmap_decoding(bbox_pred, heatmap_pred)

        # post process
        result_list = []

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
               - bbox_target (Tensor): targets of bbox predict, shape (B, 4, H, W).
               - bbox_target_weight (Tensor): weights of bbox predict, shape (B, 4, H, W).
        """
        heatmap_targets, bbox_targets, bbox_target_weights = multi_apply(
            self._build_targets_single,
            batch_gt_instances,
            featmap_size=featmap_size
            )
        
        heatmap_targets, bbox_targets, bbox_target_weights = [
            torch.stack(t, dim=0).detach() for t in [
                heatmap_targets,
                bbox_targets,
                bbox_target_weights
            ]
        ]

        return (heatmap_targets, bbox_targets, bbox_target_weights)
    
    @staticmethod
    def _calc_region(bbox, ratio):
        """
        Calculate a proportional bbox region.

        The bbox center are fixed and the new h' and w' is h * ratio and w * ratio.

        Args:
            bbox (Tensor): Bboxes to calculate regions, shape (n, 4)
            ratio (float): Ratio of the output region.

        Returns:
            tuple: x1, y1, x2, y2
        """
        x1 = torch.round((1 - ratio) * bbox[0] + ratio * bbox[2]).long()
        y1 = torch.round((1 - ratio) * bbox[1] + ratio * bbox[3]).long()
        x2 = torch.round(ratio * bbox[0] + (1 - ratio) * bbox[2]).long()
        y2 = torch.round(ratio * bbox[1] + (1 - ratio) * bbox[3]).long()

        return (x1, y1, x2, y2)
    
    def _build_targets_single(self, gt_instances: InstanceData, featmap_size: Tuple[int, int]) -> Tuple[Tensor, ...]:
        """
        Generate regression and classification targets in single image.

        Args:
            gt_instances (InstanceData):
            featmap_size (tuple): feature map shape with value [H, W]

        Returns:
            has components below:
               - center_heatmap_target (Tensor): targets of center heatmap, shape (num_classes, H, W).
               - bbox_target (Tensor): targets of wh predict, shape (4, H, W).
               - bbox_target_weight (Tensor): weights of wh predict, shape (4, H, W).
        """
        feat_h, feat_w = featmap_size
        gt_bboxes = gt_instances.bboxes
        gt_labels = gt_instances.labels

        # init targets
        bbox_target = gt_bboxes.new_ones((4, feat_h, feat_w)) * -1
        bbox_target_weight = gt_bboxes.new_zeros((4, feat_h, feat_w))

        fake_heatmap = gt_bboxes.new_zeros((feat_h, feat_w))
        heatmap_target = gt_bboxes.new_zeros((self.num_classes, feat_h, feat_w))

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
        feat_ctx_ints = ((gt_bboxes[:, 0] + gt_bboxes[:, 2]) / 2 / self.down_ratio).to(torch.int)
        feat_cty_ints = ((gt_bboxes[:, 1] + gt_bboxes[:, 3]) / 2 / self.down_ratio).to(torch.int)

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
            cat_id = gt_labels[idx]
            
            # heatmap
            fake_heatmap = fake_heatmap.zero_()
            gen_truncate_gaussian_target(
                fake_heatmap, 
                (feat_ctx_ints[idx], feat_cty_ints[idx]), 
                h_radiuses_alpha[idx].item(), 
                w_radiuses_alpha[idx].item()
                )
            
            # 若两个目标的高斯范围有重叠区域，选取大的值作为标签(对小目标友好)
            heatmap_target[cat_id] = torch.max(heatmap_target[cat_id], fake_heatmap)

            # bbox的权重
            # TODO: 好像是回归权重没有考虑优先小目标？
            if self.bbox_gaussian:
                if self.alpha != self.beta:
                    fake_heatmap = fake_heatmap.zero_()
                    gen_truncate_gaussian_target(
                        fake_heatmap, 
                        (feat_ctx_ints[idx], feat_cty_ints[idx]),
                        h_radiuses_beta[idx].item(),
                        w_radiuses_beta[idx].item()
                        )
                bbox_target_inds = fake_heatmap > 0

                local_heatmap = fake_heatmap[bbox_target_inds]
                ct_div = local_heatmap.sum()
                local_heatmap *= bbox_areas_sorted[idx]
                bbox_target_weight[:, bbox_target_inds] = local_heatmap / ct_div
            else:
                ctr_x1, ctr_y1, ctr_x2, ctr_y2 = ctr_x1s[idx], ctr_y1s[idx], ctr_x2s[idx], ctr_y2s[idx]
                bbox_target_inds = torch.zeros_like(fake_heatmap, dtype=torch.uint8)
                bbox_target_inds[ctr_y1:ctr_y2 + 1, ctr_x1:ctr_x2 + 1] = 1
                bbox_target_weight[:, bbox_target_inds] = bbox_areas_sorted[idx] / bbox_target_inds.sum().float()
            
            # 回归的是输入图上的尺寸
            bbox_target[:, bbox_target_inds] = gt_bboxes[idx][:, None]

        return heatmap_target, bbox_target, bbox_target_weight
    
    def _heatmap_decoding(self, bbox_pred: Tensor, heatmap_pred: Tensor) -> Tuple[Tensor, ...]:
        # simple nms
        pad = (self.local_maximum_kernel - 1) // 2
        hmax = F.max_pool2d(heatmap_pred, self.local_maximum_kernel, stride=1, padding=pad)
        keep = (hmax == heatmap_pred).float()
        heatmap_pred = heatmap_pred * keep

        # topk
        batch_size, _, output_h, output_w = heatmap_pred.shape
        flatten_dim = int(output_w * output_h)

        topk_scores, topk_indexes = torch.topk(heatmap_pred.view(batch_size, -1), self.topk)
        topk_scores = topk_scores.view(batch_size, self.topk)

        topk_labels = torch.div(topk_indexes, flatten_dim, rounding_mode="trunc")
        topk_labels = topk_labels.view(batch_size, self.topk)                          # [n, topk]

        topk_indexes = topk_indexes % flatten_dim
        topk_ys = torch.div(topk_indexes, output_w, rounding_mode="trunc").to(torch.float32)
        topk_xs = (topk_indexes % output_w).to(torch.float32)

        # 中心点
        topk_xs = topk_xs.view(-1, self.topk, 1) * self.down_ratio
        topk_ys = topk_ys.view(-1, self.topk, 1) * self.down_ratio

        # bbox
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).contiguous()   # [n, h, w, c]
        bbox_pred = bbox_pred.view(batch_size, -1, 4)  # [n, h*w, c]
        topk_indexes = topk_indexes.unsqueeze(2).expand(batch_size, topk_indexes.size(1), 4)
        bbox_pred = bbox_pred.gather(1, topk_indexes).view(-1, self.topk, 4)

        topk_bboxes = torch.cat([
            topk_xs - bbox_pred[..., [0]], 
            topk_ys - bbox_pred[..., [1]],
            topk_xs + bbox_pred[..., [2]],
            topk_ys + bbox_pred[..., [3]]
            ], dim=2)
        
        batch_bboxes = torch.cat((topk_bboxes, topk_scores[..., None]), dim=-1)

        return batch_bboxes, topk_labels 
    
    def _bboxes_nms(self, bboxes: Tensor, labels: Tensor, cfg: ConfigDict) -> Tuple[Tensor, Tensor]:
        """bboxes nms."""
        if labels.numel() > 0:
            max_num = cfg.max_per_img
            bboxes, keep = batched_nms(bboxes[:, :4], bboxes[:, -1].contiguous(), labels, cfg.nms)
            if max_num > 0:
                bboxes = bboxes[:max_num]
                labels = labels[keep][:max_num]

        return bboxes, labels