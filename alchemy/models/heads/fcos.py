# -*- coding: utf-8 -*-
# @Author  : ace
# Copyright by Ace, All Rights Reserved.

import torch
import torch.nn as nn

from torch import Tensor
from typing import Tuple, List, Dict, Union, Sequence, Optional

from mmcv.cnn import Scale
from mmcv.cnn import ConvModule
from mmcv.ops import batched_nms

from mmengine.model import constant_init
from mmengine.structures import InstanceData

from mmdet.models.task_modules.prior_generators import MlvlPointGenerator
from mmdet.structures.bbox import cat_boxes, scale_boxes, get_box_wh, get_box_tensor
from mmdet.models.utils import multi_apply, filter_scores_and_topk, select_single_mlvl
from mmdet.models.task_modules.coders.distance_point_bbox_coder import DistancePointBBoxCoder
from mmdet.utils import ConfigType, OptMultiConfig, InstanceList, OptInstanceList, RangeType, OptConfigType, reduce_mean

from ...registry import MODELS
from .anchor_free_head import AnchorFreeDet2dHead


INF = 1e8
StrideType = Union[Sequence[int], Sequence[Tuple[int, int]]]


@MODELS.register_module()
class AlchemyFCOS(AnchorFreeDet2dHead):
    def __init__(self, 
                 in_channels: int,
                 num_classes: int,
                 feat_channels: int,
                 stacked_convs: int,
                 center_sampling: bool = False,
                 centerness_on_reg: bool = False,
                 center_sample_radius: float = 1.5,
                 strides: StrideType = (8, 16, 32, 64, 128),
                 regress_ranges: RangeType = ((-1, 64), (64, 128), (128, 256), (256, 512), (512, INF)),
                 loss_cls: ConfigType = dict(type='mmdet.FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0),
                 loss_bbox: ConfigType = dict(type='EfficientIoULoss', loss_weight=1.0),
                 loss_centerness: ConfigType = dict(type='mmdet.CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
                 conv_cfg: OptConfigType = None,
                 norm_cfg: OptConfigType = None,
                 test_cfg: OptMultiConfig = None,
                 init_cfg: OptMultiConfig = None,
                 train_cfg: OptMultiConfig = None) -> None:
        
        # args
        self.strides = strides
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.regress_ranges = regress_ranges
        self.center_sampling = center_sampling
        self.centerness_on_reg = centerness_on_reg
        self.center_sample_radius = center_sample_radius
        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)

        if self.use_sigmoid_cls:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1

        assert len(self.strides) == len(self.regress_ranges)

        super().__init__(init_cfg=init_cfg, test_cfg=test_cfg, train_cfg=train_cfg, in_channels=in_channels, num_classes=num_classes, conv_cfg=conv_cfg, norm_cfg=norm_cfg)

        # init loss
        self.loss_cls = MODELS.build(loss_cls)
        self.loss_bbox = MODELS.build(loss_bbox)
        self.loss_centerness = MODELS.build(loss_centerness)

        # 每层特征图点坐标生成器
        self.mlvl_points_generator = MlvlPointGenerator(self.strides)

        # 特征层数
        self.num_levels = len(self.strides)

        self.distance_point_coder = DistancePointBBoxCoder()

        # test config
        self.nms_pre = -1 if self.test_cfg is None else self.test_cfg.get('nms_pre', -1)
        self.max_per_img = 100 if self.test_cfg is None else self.test_cfg.get('max_per_img', 100)

    def initial_head(self) -> None:
        self.cls_convs = self._build_cls_convs()
        self.reg_convs = self._build_reg_convs()

        # heads
        self.cls_head = nn.Conv2d(self.feat_channels, self.cls_out_channels, kernel_size=3, stride=1, padding=1)
        self.reg_head = nn.Conv2d(self.feat_channels, 4, kernel_size=3, stride=1, padding=1)
        self.centerness_head = nn.Conv2d(self.feat_channels, 1, kernel_size=3, stride=1, padding=1)

        # scales
        self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides])
    
    def init_weights(self) -> None:
        super().init_weights()

        for m in self.modules():
            # DeformConv2dPack, ModulatedDeformConv2dPack
            if hasattr(m, 'conv_offset'):
                constant_init(m.conv_offset, 0)
  
    def forward(self, img_feats: Tuple[Tensor], **kwargs) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]:
        """
        Forward features from the upstream network.

        Args:
            img_feats (tuple[Tensor]): Features from the upstream network, each is a 4D-tensor.

        Returns:
            tuple: A tuple of each level outputs.

            - cls_scores (list[Tensor]): Box scores for each scale level, each is a 4D-tensor, the channel number is num_points * num_classes.
            - bbox_preds (list[Tensor]): Box energies / deltas for each scale level, each is a 4D-tensor, the channel number is num_points * 4.
            - centernesses (list[Tensor]): centerness for each scale level, each is a 4D-tensor, the channel number is num_points * 1.
        """
        return multi_apply(self.forward_single, img_feats, self.scales, self.strides)
    
    def forward_single(self, img_feat: Tensor, scale: Scale, stride: int) -> Tuple[Tensor, Tensor, Tensor]:
        """Forward features of a single scale level.

        Args:
            img_feat (Tensor): FPN feature maps of the specified stride.
            scale (:obj:`mmcv.cnn.Scale`): Learnable scale module to resize the bbox prediction.
            stride (int): The corresponding stride for feature maps.

        Returns:
            tuple: scores for each class, bbox predictions and centerness
            predictions of input feature maps.
        """
        cls_feat = img_feat
        reg_feat = img_feat

        # cls
        cls_feat = self.cls_convs(cls_feat)
        cls_score = self.cls_head(cls_feat)

        # reg
        reg_feat = self.reg_convs(reg_feat)
        bbox_pred = self.reg_head(reg_feat)

        # scale the bbox_pred of different level float to avoid overflow when enabling FP16
        bbox_pred = scale(bbox_pred).float().clamp(min=0) * stride

        # centerness
        if self.centerness_on_reg:
            centerness = self.centerness_head(reg_feat)
        else:
            centerness = self.centerness_head(cls_feat)

        return cls_score, bbox_pred, centerness
    
    def loss_by_feats(self, 
                      cls_scores: List[Tensor],
                      bbox_preds: List[Tensor],
                      centernesses: List[Tensor],
                      batch_gt_instances: InstanceList, 
                      batch_img_metas: List[dict], 
                      batch_gt_instances_ignore: OptInstanceList = None) -> Dict[str, Tensor]:
        """
        Calculate the loss based on the features extracted by the detection head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level, each is a 4D-tensor, the channel number is num_points * num_classes.
            bbox_preds (list[Tensor]): Box energies / deltas for each scale level, each is a 4D-tensor, the channel number is num_points * 4.
            centernesses (list[Tensor]): centerness for each scale level, each is a 4D-tensor, the channel number is num_points * 1.
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of gt_instance. It usually includes 'bboxes' and 'labels' attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g., image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], optional): Batch of gt_instances_ignore. It includes 'bboxes' attribute data that is ignored during training and testing. Defaults to None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        
        assert len(cls_scores) == len(bbox_preds) == len(centernesses)

        # 获取每层特征图的尺寸
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]

        # 根据每层特征图的尺寸, 生成点坐标(这里的坐标是指在输入图上的坐标)
        mlvl_points = self.mlvl_points_generator.grid_priors(featmap_sizes, dtype=bbox_preds[0].dtype,  device=bbox_preds[0].device)

        # generate ground-truth
        batch_gt_labels, batch_gt_bboxes = self._build_targets(mlvl_points, batch_gt_instances)

        bs = cls_scores[0].size(0)

        # flatten cls_scores, bbox_preds and centerness
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
            for cls_score in cls_scores
        ]

        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
            for bbox_pred in bbox_preds
        ]

        flatten_centerness = [
            centerness.permute(0, 2, 3, 1).reshape(-1)
            for centerness in centernesses
        ]

        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)
        flatten_centerness = torch.cat(flatten_centerness)

        flatten_labels = torch.cat(batch_gt_labels)
        flatten_bbox_targets = torch.cat(batch_gt_bboxes)

        # repeat points to align with bbox_preds
        flatten_points = torch.cat([points.repeat(bs, 1) for points in mlvl_points])

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = ((flatten_labels >= 0) & (flatten_labels < bg_class_ind)).nonzero().reshape(-1)
        num_pos = torch.tensor(len(pos_inds), dtype=torch.float, device=bbox_preds[0].device)
        num_pos = max(reduce_mean(num_pos), 1.0)
        loss_cls = self.loss_cls(flatten_cls_scores, flatten_labels, avg_factor=num_pos)

        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        pos_centerness = flatten_centerness[pos_inds]
        pos_bbox_targets = flatten_bbox_targets[pos_inds]
        pos_centerness_targets = self._centerness_target(pos_bbox_targets)

        # centerness weighted iou loss
        centerness_denorm = max(reduce_mean(pos_centerness_targets.sum().detach()), 1e-6)

        if len(pos_inds) > 0:
            pos_points = flatten_points[pos_inds]
            pos_decoded_bbox_preds = self.distance_point_coder.decode(pos_points, pos_bbox_preds)
            pos_decoded_target_preds = self.distance_point_coder.decode(pos_points, pos_bbox_targets)
            loss_bbox = self.loss_bbox(pos_decoded_bbox_preds, pos_decoded_target_preds, weight=pos_centerness_targets, avg_factor=centerness_denorm)
            loss_centerness = self.loss_centerness(pos_centerness, pos_centerness_targets, avg_factor=num_pos)
        else:
            loss_bbox = pos_bbox_preds.sum()
            loss_centerness = pos_centerness.sum()

        return dict(
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            loss_centerness=loss_centerness
            )
    
    def predict_by_feats(self, cls_scores: List[Tensor], bbox_preds: List[Tensor], centerness: Optional[List[Tensor]], batch_img_metas: List[dict], rescale: bool = False, **kwargs) -> List[Tensor]:
        assert len(cls_scores) == len(bbox_preds) == len(centerness)

        num_levels = len(cls_scores)

        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_points = self.mlvl_points_generator.grid_priors(featmap_sizes, dtype=cls_scores[0].dtype, device=cls_scores[0].device)
        
        result_list = []

        for img_id, img_meta in enumerate(batch_img_metas):
            cls_score_list = select_single_mlvl(cls_scores, img_id, detach=True)
            bbox_pred_list = select_single_mlvl(bbox_preds, img_id, detach=True)
            centerness_list = select_single_mlvl(centerness, img_id, detach=True)

            results = self._predict_by_feat_single(
                cls_score_list=cls_score_list,
                bbox_pred_list=bbox_pred_list,
                centerness_list=centerness_list,
                mlvl_points=mlvl_points,
                img_meta=img_meta,
                rescale=rescale)
            
            result_list.append(results)

        
        return result_list
    
    def _build_cls_convs(self) -> nn.Sequential:
        """
        Initialize classification conv layers of the head.
        """
        cls_convs = []
        for i in range(self.stacked_convs):
            cls_convs.append(
                ConvModule(
                    self.in_channels if i == 0 else self.feat_channels,
                    self.feat_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg
                    )
                )
        
        return nn.Sequential(*cls_convs)

    def _build_reg_convs(self) -> nn.Sequential:
        """
        Initialize bbox regression conv layers of the head.
        """
        reg_convs = []
        for i in range(self.stacked_convs):
            reg_convs.append(
                ConvModule(
                    self.in_channels if i == 0 else self.feat_channels,
                    self.feat_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg
                    )
                )
        
        return nn.Sequential(*reg_convs)
            
    def _build_targets(self, mlvl_points: List[Tensor], batch_gt_instances: InstanceList) -> Tuple[List[Tensor], List[Tensor]]:
        """
        Compute regression, classification and centerness targets for points in multiple images.

        Args:
            points (list[Tensor]): Points of each fpn level, each has shape (num_points, 2).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of gt_instance.  It usually includes 'bboxes' and 'labels' attributes.

        Returns:
            tuple: Targets of each level.

            - concat_lvl_labels (list[Tensor]): Labels of each level.
            - concat_lvl_bbox_targets (list[Tensor]): BBox targets of each level.
        """
        assert len(mlvl_points) == len(self.regress_ranges)

        # 计算每层的的特征点数量
        num_lvl_points = [center.size(0) for center in mlvl_points]

        # 将每层的回归范围, 扩展成与mlvl_points相同的维度[(num_lvl_points, (range_start, range_end)), (num_lvl_points, (range_start, range_end))]
        expanded_regress_ranges = [
            mlvl_points[lvl_id].new_tensor(self.regress_ranges[lvl_id])[None].expand_as(mlvl_points[lvl_id]) for lvl_id in range(self.num_levels)
        ]

        # 将所有层的特征点和回归范围拼起来
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)  # [num_mlvl_points, 2]
        concat_points = torch.cat(mlvl_points, dim=0)  # [num_mlvl_points, 2]

        # 制作每张图片的gt
        batch_gt_labels, batch_gt_bboxes = multi_apply(
            self._build_targets_single,
            batch_gt_instances,
            mlvl_points=concat_points,
            num_lvl_points=num_lvl_points,
            regress_ranges=concat_regress_ranges
            )
        
        # 划分成每张图每层的gt
        batch_gt_labels = [gt_labels.split(num_lvl_points, 0) for gt_labels in batch_gt_labels]
        batch_gt_bboxes = [gt_bboxes.split(num_lvl_points, 0) for gt_bboxes in batch_gt_bboxes]
        
        # 转换成, 每层的标签(按batch拼起来)
        concat_lvl_labels = []
        concat_lvl_bbox_targets = []

        for i in range(self.num_levels):
            concat_lvl_labels.append(torch.cat([gt_labels[i] for gt_labels in batch_gt_labels]))
            concat_lvl_bbox_targets.append(torch.cat([gt_bboxes[i] for gt_bboxes in batch_gt_bboxes]))

        return concat_lvl_labels, concat_lvl_bbox_targets
    
    def _build_targets_single(self, gt_instances: InstanceData, mlvl_points: Tensor, regress_ranges: Tensor, num_lvl_points: List[int]) -> Tuple[Tensor, Tensor]:
        """
        Compute regression and classification targets for a single image.
        """
        num_gts = len(gt_instances)
        num_points = mlvl_points.size(0)   # 每张图一共有多少个点

        gt_bboxes = gt_instances.bboxes    # [num_gts, 4]
        gt_labels = gt_instances.labels    # [num_gts]

        if num_gts == 0:
            return gt_labels.new_full((num_points,), self.num_classes), gt_bboxes.new_zeros((num_points, 4))

        # 计算目标的面积
        areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * (gt_bboxes[:, 3] - gt_bboxes[:, 1])  # [num_gts]

        # 维度转换
        areas = areas[None].repeat(num_points, 1)   # [num_points, num_gts]
        regress_ranges = regress_ranges[:, None, :].expand(num_points, num_gts, 2)   # [num_points, num_gts, 2]
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 4)   # [num_points, num_gts, 4]

        # 获取点的坐标(输入图上的坐标)
        xs, ys = mlvl_points[:, 0], mlvl_points[:, 1]   # [num_points]
        xs = xs[:, None].expand(num_points, num_gts)    # [num_points, num_gts]
        ys = ys[:, None].expand(num_points, num_gts)

        # 计算每个点距离4条边的距离
        left = xs - gt_bboxes[..., 0]
        top = ys - gt_bboxes[..., 1]
        right = gt_bboxes[..., 2] - xs
        bottom = gt_bboxes[..., 3] - ys
        bbox_targets = torch.stack((left, top, right, bottom), -1)   # 表示每个点距离所有gt_bbox4条边的距离值, 形状:[num_points, num_gts, 4]

        # 是否中心采样
        if self.center_sampling:
            # condition1: 确定gt_box自身所对应的内部有效区域
            radius = self.center_sample_radius

            # 计算所有gt_bbox各自的中心坐标
            center_xs = (gt_bboxes[..., 0] + gt_bboxes[..., 2]) / 2    # [num_points, num_gts]
            center_ys = (gt_bboxes[..., 1] + gt_bboxes[..., 3]) / 2
            center_gts = torch.zeros_like(gt_bboxes)
            stride = center_xs.new_zeros(center_xs.shape)   # [num_points, num_gts]

            # 中心采样参数是全局参数，需要和各个输出层stride联合起来作为控制参数
            # 例如: radius=1.5，
            #    对于 stride=4 的层，采样半径是1.5x4
            #    对于 stride=8 的层，采样半径是1.5x8
            lvl_begin = 0
            for lvl_idx, num_points_lvl in enumerate(num_lvl_points):
                lvl_end = lvl_begin + num_points_lvl
                stride[lvl_begin:lvl_end] = self.strides[lvl_idx] * radius
                lvl_begin = lvl_end

            # 计算中心扩展范围后新的gt_bbox，实际上相当于向内缩小了
            x_mins = center_xs - stride
            y_mins = center_ys - stride
            x_maxs = center_xs + stride
            y_maxs = center_ys + stride

            # 如果设置的半径范围太大，导致中心扩展范围超过了g_bbox 本身，则截断center_gts相当于4条边向内缩放后新的gt_bbox值
            center_gts[..., 0] = torch.where(x_mins > gt_bboxes[..., 0], x_mins, gt_bboxes[..., 0])
            center_gts[..., 1] = torch.where(y_mins > gt_bboxes[..., 1], y_mins, gt_bboxes[..., 1])
            center_gts[..., 2] = torch.where(x_maxs > gt_bboxes[..., 2], gt_bboxes[..., 2], x_maxs)
            center_gts[..., 3] = torch.where(y_maxs > gt_bboxes[..., 3], gt_bboxes[..., 3], y_maxs)

            # 计算每个点距center_gts, 4条边的距离
            cb_dist_left = xs - center_gts[..., 0]
            cb_dist_top = ys - center_gts[..., 1]
            cb_dist_right = center_gts[..., 2] - xs
            cb_dist_bottom = center_gts[..., 3] - ys
            center_bbox = torch.stack((cb_dist_left, cb_dist_top, cb_dist_right, cb_dist_bottom), -1)

            # 此时就可以得到进行中心采样后每个点是否是正样本(暂时), 在ltrb维度上去最小值, 若最小值>0则一定在范围内
            inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0   # [num_points, num_gts]
        else:
            # condition1: 在gt_box范围内
            # 在不开启中心采样情况下，只要点距4条边的距离都大于0，那么说明该点可能在某个或者某几个gt_bbox内部, 暂时属于正样本
            inside_gt_bbox_mask = bbox_targets.min(-1)[0] > 0  # [num_points, num_gts]

        # condition2: 对于每个点, 计算最长边值, bbox_targets形状为[num_points, num_gts, 4]
        max_regress_distance = bbox_targets.max(-1)[0]

        # 如果ltrb中最长边处于预设范围, 则属于正样本, 否则为负样本
        inside_regress_range = ((max_regress_distance >= regress_ranges[..., 0]) & (max_regress_distance <= regress_ranges[..., 1]))

        # 不满足condition1和condition2的点作为负样本
        areas[inside_gt_bbox_mask == 0] = INF
        areas[inside_regress_range == 0] = INF

        min_area, min_area_inds = areas.min(dim=1)

        labels = gt_labels[min_area_inds]
        labels[min_area == INF] = self.num_classes  # set as BG

        # 如果一个位置仍对应很多目标, 选择面积最小的那个
        bbox_targets = bbox_targets[range(num_points), min_area_inds]

        return labels, bbox_targets

    def _centerness_target(self, pos_bbox_targets: Tensor) -> Tensor:
        """
        Compute centerness targets.

        Args:
            pos_bbox_targets (Tensor): BBox targets of positive bboxes in shape (num_pos, 4)

        Returns:
            Tensor: Centerness target.
        """
        # only calculate pos centerness targets, otherwise there may be nan
        left_right = pos_bbox_targets[:, [0, 2]]
        top_bottom = pos_bbox_targets[:, [1, 3]]

        if len(left_right) == 0:
            centerness_targets = left_right[..., 0]
        else:
            centerness_targets = (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])

        return torch.sqrt(centerness_targets)
    
    def _predict_by_feat_single(self, cls_score_list: List[Tensor], bbox_pred_list: List[Tensor], centerness_list: List[Tensor], mlvl_points: List[Tensor], img_meta: dict, rescale: bool = False) -> InstanceData:
        """
        Transform a single image's features extracted from the head into bbox results.

        Args:
            cls_score_list (list[Tensor]): Box scores from all scale levels of a single image, each item has shape (num_priors * num_classes, H, W).
            bbox_pred_list (list[Tensor]): Box energies / deltas from all scale levels of a single image, each item has shape (num_priors * 4, H, W).
            centerness_list (list[Tensor]): Score factor from all scale levels of a single image, each item has shape (num_priors * 1, H, W).
            mlvl_points (list[Tensor]): Each element in the list is the priors of a single level in feature pyramid. 
                                        In all anchor-free methods, it has shape (num_priors, 2) when `with_stride=True`, otherwise it still has shape (num_priors, 4).
            img_meta (dict): Image meta info.
            rescale (bool): If True, return boxes in original image space. Defaults to False.

        Returns:
            :obj:`InstanceData`: Detection results of each image after the post process.
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4), the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        img_shape = img_meta['img_shape']

        mlvl_scores = []
        mlvl_labels = []
        mlvl_bbox_preds = []
        mlvl_valid_priors = []
        mlvl_score_factors = []

        for cls_score, bbox_pred, score_factor, lvl_points in zip(cls_score_list, bbox_pred_list, centerness_list, mlvl_points):

            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]

            cls_score = cls_score.permute(1, 2, 0).reshape(-1, self.cls_out_channels)
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, self.distance_point_coder.encode_size)
            score_factor = score_factor.permute(1, 2, 0).reshape(-1).sigmoid()
            
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
                # BG cat_id: num_class
                scores = cls_score.softmax(-1)[:, :-1]

            # filter and topk
            scores, labels, keep_idxs, filtered_results = filter_scores_and_topk(scores, self.score_thr, self.nms_pre, dict(bbox_pred=bbox_pred, priors=lvl_points))

            priors = filtered_results['priors']
            bbox_pred = filtered_results['bbox_pred']
            
            score_factor = score_factor[keep_idxs]

            mlvl_scores.append(scores)
            mlvl_labels.append(labels)
            mlvl_valid_priors.append(priors)
            mlvl_bbox_preds.append(bbox_pred)
            mlvl_score_factors.append(score_factor)

        bbox_pred = torch.cat(mlvl_bbox_preds)
        priors = cat_boxes(mlvl_valid_priors)
        bboxes = self.distance_point_coder.decode(priors, bbox_pred, max_shape=img_shape)

        results = InstanceData()
        results.bboxes = bboxes
        results.scores = torch.cat(mlvl_scores)
        results.labels = torch.cat(mlvl_labels)
        results.score_factors = torch.cat(mlvl_score_factors)

        return self._rescale_and_nms(results=results, rescale=rescale, img_meta=img_meta)
    
    def _rescale_and_nms(self, results: InstanceData, rescale: bool = False, img_meta: Optional[dict] = None) -> InstanceData:
        """
        bbox post-processing method.

        The boxes would be rescaled to the original image scale and do the nms operation.

        Args:
            results (:obj:`InstaceData`): Detection instance results, each item has shape (num_bboxes, ).
            rescale (bool): If True, return boxes in original image space. Default to False.
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
            # TODO: Add sqrt operation in order to be consistent with
            #  the paper.
            score_factors = results.pop('score_factors')
            results.scores = results.scores * score_factors

        # filter small size bboxes
        if self.test_cfg.get('min_bbox_size', -1) >= 0:
            w, h = get_box_wh(results.bboxes)
            valid_mask = (w > self.test_cfg.min_bbox_size) & (h > self.test_cfg.min_bbox_size)
            if not valid_mask.all():
                results = results[valid_mask]

        # nms
        if results.bboxes.numel() > 0:
            nms_cfg = self.test_cfg.get('nms', dict(type='nms', iou_threshold=0.5))
            bboxes = get_box_tensor(results.bboxes)
            det_bboxes, keep_idxs = batched_nms(bboxes, results.scores, results.labels, nms_cfg)
            results = results[keep_idxs]

            # some nms would reweight the score, such as softnms
            results.scores = det_bboxes[:, -1]
            results = results[:self.test_cfg.max_per_img]

        return results
