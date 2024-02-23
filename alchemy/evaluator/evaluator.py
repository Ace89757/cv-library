# -*- coding: utf-8 -*-
# @Author  : ace
# Copyright by Ace, All Rights Reserved.

from typing import Sequence, Union

from mmengine.evaluator.evaluator import Evaluator
from mmengine.evaluator.metric import BaseMetric, BaseDataElement

from ..registry import EVALUATOR


@EVALUATOR.register_module()
class AlchemyEvaluator(Evaluator):
    """
    相较于mmengine中的evaluator增加了
        "current_epoch" 
        "work_dir"
    属性,在计算kpi时,可以直接调用这些属性,进行操作
    """
    def __init__(self, metrics: Union[dict, BaseMetric, Sequence]):
        super().__init__(metrics)

        self._work_dir: str = None
        self._current_epoch: int = None
        
    @property
    def current_epoch(self) -> int:
        return self._current_epoch

    @current_epoch.setter
    def current_epoch(self, current_epoch: int) -> None:
        """Set the dataset meta info to the evaluator and it's metrics."""
        self._current_epoch = current_epoch
        for metric in self.metrics:
            metric.current_epoch = current_epoch

    @property
    def work_dir(self) -> int:
        return self._work_dir

    @work_dir.setter
    def work_dir(self, work_dir: str) -> None:
        """
        Set the dataset meta info to the evaluator and it's metrics.
        """
        self._work_dir = work_dir
        for metric in self.metrics:
            metric.work_dir = work_dir
    
    def process(self, data_samples: Sequence[BaseDataElement]):
        """
        Convert BaseDataSample to dict and invoke process method of each metric.

        Args:
            data_samples (Sequence[BaseDataElement]): predictions of the model, and the ground truth of the validation set.
        """
        _data_samples = [data_sample.to_dict() if isinstance(data_sample, BaseDataElement) else data_sample for data_sample in data_samples]

        for metric in self.metrics:
            metric.process(_data_samples)