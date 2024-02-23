# -*- coding: utf-8 -*-
# @Author  : ace
# Copyright by Ace, All Rights Reserved.

import torch
import torch.nn as nn

from torch import Tensor
from copy import deepcopy
from typing import List, Tuple, Union, Dict

from mmengine.model.base_model import BaseModel

from mmdet.models.utils import samplelist_boxtype2tensor
from mmdet.utils import ConfigType, OptConfigType, InstanceList
from mmdet.structures import OptSampleList, SampleList, DetDataSample

from ...registry import MODELS


ForwardResults = Union[Dict[str, Tensor], List[DetDataSample], Tuple[Tensor], Tensor]


@MODELS.register_module()
class AlchemyDet2dDetector(BaseModel):
    def __init__(self,
                 backbone: ConfigType,
                 rpn: OptConfigType = None,
                 head: OptConfigType = None,
                 neck: Union[List[OptConfigType], OptConfigType] = None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # backbone & necks
        self._build_stem(backbone, neck)

        # rpn
        if rpn is not None:
            self._build_rpn(rpn)
        
        # head
        self._build_head(head)
    
    def _build_stem(self, backbone: OptConfigType, neck: Union[List[OptConfigType], OptConfigType] = None) -> None:
        """
        build backbone and neck module.
        """
        self.backbone = MODELS.build(backbone)

        if neck is not None:
            if isinstance(neck, dict):
                self.neck = MODELS.build(neck)
            
            # 多个neck组合
            elif isinstance(neck, list):
                necks = []
                for neck_cfg in neck:
                    necks.append(MODELS.build(neck_cfg))
                
                self.neck = nn.Sequential(*necks)
    
    def _build_rpn(self, rpn: OptConfigType = None) -> None:
        """
        build rpn module
        """
        rpn.update(num_classes=1)

        self.rpn = MODELS.build(rpn)

    def _build_head(self, head: OptConfigType) -> None:
        """
        build detector module.
        """ 
        self.head = MODELS.build(head)
    
    def forward(self, inputs: Tensor, data_samples: OptSampleList = None, mode: str = 'tensor') -> ForwardResults:
        """
        经过backbone & neck (rpn)之后, 进入head模块操作

        Args:
            inputs (Tensor): The input tensor with shape (N, C, ...) in general.
            data_samples (List[DetDataSample], optional): A batch of data samples that contain annotations and predictions. Defaults to None.
            mode (str): Return what kind of value. Defaults to 'tensor'.

        Returns:
            The return type depends on mode.

            - If mode="tensor", return a tensor or a tuple of tensor.
            - If mode="predict", return a list of :obj:`DetDataSample`.
            - If mode="loss", return a dict of tensor.
        """
        if mode == 'loss':
            # 返回loss-dict
            return self.loss(inputs, data_samples)
        
        elif mode == 'predict':
            # 返回检测结果(经过后处理)
            return self.predict(inputs, data_samples)
        
        elif mode == 'tensor':
            # 返回检测结果(未经过后处理)
            return self.predict(inputs, data_samples, return_tensor=True)
        
        else:
            raise RuntimeError(f'Invalid mode "{mode}". Only supports loss, predict and tensor mode')
    
    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
        """
        Extract features.

        Args:
            batch_inputs (Tensor): Image tensor with shape (N, C, H, W).

        Returns:
            tuple[Tensor]: Multi-level features that may have different resolutions.
        """
        x = self.backbone(batch_inputs)
        
        if self.has_neck:
            x = self.neck(x)

        return x
    
    def loss(self, batch_inputs: Tensor, batch_data_samples: SampleList) -> Union[dict, list]:
        """
        计算batch数据的loss.

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W). These should usually be mean centered and std scaled.
            batch_data_samples (list[DetDataSample]): The batch data samples. It usually includes information such as 'gt_instance' or 'gt_panoptic_seg' or 'gt_sem_seg'.

        Returns:
            dict: A dictionary of loss components.
        """
        img_feats = self.extract_feat(batch_inputs)

        losses = dict()

        # rpn
        if self.has_rpn:
            rpn_data_samples = deepcopy(batch_data_samples)

            # 构建rpn的标签类别, rpn是2分类, 将所有有效目标的类别id都设置为0
            for data_sample in rpn_data_samples:
                data_sample.gt_instances.labels = torch.zeros_like(data_sample.gt_instances.labels)

            rpn_losses, rpn_results_list = self.rpn.loss_and_predict(img_feats, rpn_data_samples)

            losses.update(rpn_losses)
            head_inputs = (img_feats, batch_data_samples, rpn_results_list)
        else:
            head_inputs = (img_feats, batch_data_samples)
        
        # head
        head_losses = self.head.loss(*head_inputs)
        losses.update(head_losses)

        return losses

    def predict(self, batch_inputs: Tensor, batch_data_samples: SampleList, rescale: bool = True, return_tensor: bool = False) -> SampleList:
        """
        Predict results from a batch of inputs and data samples with post-processing.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (List[:obj:'DetDataSample']): The Data Samples. It usually includes information such as 'gt_instance', 'gt_panoptic_seg' and 'gt_sem_seg'.
            rescale (bool): Whether to rescale the results. Defaults to True.

        Returns:
            list[:obj:'DetDataSample']: Detection results of the
            input images. Each DetDataSample usually contain 'pred_instances'. And the 'pred_instances' usually
            contains following keys.

                - scores (Tensor): Classification scores, has a shape (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4), the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        img_feats = self.extract_feat(batch_inputs)

        # rpn
        if self.has_rpn:
            # 如果没有预定义的proposals则使用rpn生成proposals
            if batch_data_samples[0].get('proposals', None) is None:
                rpn_results_list = self.rpn.predict(img_feats, batch_data_samples, rescale=False)
            else:
                rpn_results_list = [data_sample.proposals for data_sample in batch_data_samples]

            head_inputs = (img_feats, batch_data_samples, rpn_results_list)
        else:
            head_inputs = (img_feats, batch_data_samples)

        # head
        results_list = self.head.predict(*head_inputs, rescale=rescale)

        if return_tensor:
            return results_list
        
        batch_data_samples = self.add_pred_to_datasample(batch_data_samples, results_list)

        return batch_data_samples

    def add_pred_to_datasample(self, data_samples: SampleList, results_list: InstanceList) -> SampleList:
        """
        Add predictions to `DetDataSample`.

        Args:
            data_samples (list[:obj:`DetDataSample`], optional): A batch of data samples that contain annotations and predictions.
            results_list (list[:obj:`InstanceData`]): Detection results of each image.

        Returns:
            list[:obj:`DetDataSample`]: Detection results of the input images. 
                                        Each DetDataSample usually contain
            'pred_instances'. And the pred_instances usually contains following keys.

                - scores (Tensor): Classification scores, has a shape (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4), the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        for data_sample, pred_instances in zip(data_samples, results_list):
            data_sample.pred_instances = pred_instances

        samplelist_boxtype2tensor(data_samples)

        return data_samples
    
    @property
    def has_rpn(self) -> bool:
        return hasattr(self, 'rpn') and self.rpn is not None

    @property
    def has_neck(self) -> bool:
        return hasattr(self, 'neck') and self.neck is not None