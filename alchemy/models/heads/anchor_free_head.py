# -*- coding: utf-8 -*-
# @Author  : ace
# Copyright by Ace, All Rights Reserved.

from torch import Tensor
from abc import abstractmethod
from typing import Tuple, List, Optional

from mmengine.model import BaseModule, constant_init

from mmdet.utils import InstanceList
from mmdet.structures import SampleList
from mmdet.models.utils import unpack_gt_instances
from mmdet.utils import InstanceList, OptMultiConfig


class AnchorFreeDet2dHead(BaseModule):
    """
    Base class for anchor free of det2d Head.

    1. The 'init_weights' method is used to initialize head's model parameters. 
       After detector initialization, 'init_weights' is triggered when 'detector.init_weights()' is called externally.

    2. The 'loss' method is used to calculate the loss of densehead, which includes two steps: 
        (1) the densehead model performs forward propagation to obtain the feature maps 
        (2) The 'loss_by_feat' method is called based on the feature maps to calculate the loss.

    loss(): forward() -> loss_by_feat()

    3. The 'predict' method is used to predict detection results, which includes two steps: 
        (1) the densehead model performs forward propagation to obtain the feature maps 
        (2) The 'predict_by_feat' method is called based on the feature maps to predict detection results including post-processing.


    predict(): forward() -> predict_by_feat()

    4. The 'loss_and_predict' method is used to return loss and detection results at the same time. 
       It will call densehead's 'forward', 'loss_by_feat' and 'predict_by_feat' methods in order.  
       If one-stage is used as RPN, the densehead needs to return both losses and predictions.
       This predictions is used as the proposal of roihead.

    loss_and_predict(): forward() -> loss_by_feat() -> predict_by_feat()
    """

    def __init__(self,
                 in_channels: int,
                 num_classes: int,
                 test_cfg: OptMultiConfig = None,
                 init_cfg: OptMultiConfig = None,
                 train_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg=init_cfg)

        # args
        self.in_channels = in_channels
        self.num_classes = num_classes

        # cfg
        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        self.score_thr = self.test_cfg.get('score_thr', 0.5) if self.test_cfg is not None else 0.5

        self.fp16_enabled = False
        
        # init heads
        self.initial_head()

    @abstractmethod
    def initial_head(self) -> None:
        """
        Initialize the head modules.
        """
        raise NotImplementedError

    def init_weights(self) -> None:
        """
        Initialize the weights.
        """
        super().init_weights()

        # avoid init_cfg overwrite the initialization of 'conv_offset'
        for m in self.modules():
            if hasattr(m, 'conv_offset'):  # DeformConv2dPack, ModulatedDeformConv2dPack
                constant_init(m.conv_offset, 0)
    
    @abstractmethod
    def loss_by_feats(self, **kwargs) -> dict:
        """
        Calculate the loss based on the features extracted by the detection head.
        """
        raise NotImplementedError
    
    @abstractmethod
    def predict_by_feats(self, batch_img_metas: Optional[List[dict]] = None, rescale: bool = False, with_nms: bool = True, **kwargs) -> List[Tensor]:
        """
        Transform a batch of output features extracted from the head into bbox results.
        """
        raise NotImplementedError

    def loss(self, img_feats: Tuple[Tensor], batch_data_samples: SampleList, **kwargs) -> dict:
        """
        Perform forward propagation and loss calculation of the detection head on the features of the upstream network.

        Args:
            img_feats (tuple[Tensor]): Multi-level features from the upstream network, each is a 4D-tensor.
            batch_data_samples (List[DetDataSample]): The Data Samples. It usually includes information such as 'gt_instance'.

        Returns:
            dict: A dictionary of loss components.
        """
        # unpack
        batch_gt_instances, batch_gt_instances_ignore, batch_img_metas = unpack_gt_instances(batch_data_samples)

        # forward
        outs = self(img_feats=img_feats, batch_img_metas=batch_img_metas, **kwargs)
        outs = outs + (batch_gt_instances, batch_img_metas, batch_gt_instances_ignore)

        # loss
        losses = self.loss_by_feats(*outs)

        return losses

    def predict(self, img_feats: Tuple[Tensor], batch_data_samples: SampleList, rescale: bool = True, **kwargs) -> InstanceList:
        """
        Perform forward propagation of the detection head and predict detection results on the features of the upstream network.

        Args:
            img_feats (tuple[Tensor]): Multi-level features from the upstream network, each is a 4D-tensor.
            batch_data_samples (List[DetDataSample]): The Data Samples. It usually includes information such as 'gt_instance'.
            rescale (bool, optional): Whether to rescale the results. Defaults to False.

        Returns:
            list[InstanceData]: Object detection results of each image after the post process. 
                                Each item usually contains following keys.
                                    - scores (Tensor): Classification scores, has a shape (num_instance, )
                                    - labels (Tensor): Labels of bboxes, has a shape (num_instances, ).
                                    - bboxes (Tensor): Has a shape (num_instances, 4), the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        batch_img_metas = [data_samples.metainfo for data_samples in batch_data_samples]

        # forward
        outs = self(img_feats=img_feats, batch_img_metas=batch_img_metas, **kwargs)

        # decode & post-process
        result_list = self.predict_by_feats(*outs, batch_img_metas=batch_img_metas, rescale=rescale, proposal_cfg=self.test_cfg)

        return result_list
    
    def loss_and_predict(self, img_feats: Tuple[Tensor], batch_data_samples: SampleList, **kwargs) -> Tuple[dict, InstanceList]:
        """
        Perform forward propagation of the head, then calculate loss and predictions from the features and data samples.

        Args:
            img_feats (tuple[Tensor]): Multi-level features from the upstream network, each is a 4D-tensor.
            batch_data_samples (list[DetDataSample]): Each item contains the meta information of each image and corresponding annotations.
            proposal_cfg (ConfigDict, optional): Test / postprocessing configuration, if None, test_cfg would be used. Defaults to None.

        Returns:
            tuple: the return value is a tuple contains:
                - losses: (dict[str, Tensor]): A dictionary of loss components.
                - predictions (list['InstanceData']): Detection results of each image after the post process.
        """
        batch_gt_instances, batch_gt_instances_ignore, batch_img_metas = unpack_gt_instances(batch_data_samples)

        # forward
        outs = self(img_feats=img_feats, batch_img_metas=batch_img_metas, **kwargs)

        # loss
        loss_inputs = outs + (batch_gt_instances, batch_img_metas, batch_gt_instances_ignore)
        losses = self.loss_by_feats(*loss_inputs)

        # decode & post-process
        result_list = self.predict_by_feats(*outs, batch_img_metas=batch_img_metas, rescale=False, proposal_cfg=self.test_cfg)

        return losses, result_list

