# -*- coding: utf-8 -*-
# @Author  : ace
# Copyright by Ace, All Rights Reserved.

import torch

from typing import Sequence

from mmengine.runner.amp import autocast
from mmengine.runner.loops import ValLoop

from ..registry import LOOPS


@LOOPS.register_module()
class AlchemyValLoop(ValLoop):
    def run(self) -> dict:
        """
        Launch validation.
        """
        self.runner.call_hook('before_val')
        self.runner.call_hook('before_val_epoch')
        self.runner.model.eval()
        for idx, data_batch in enumerate(self.dataloader):
            self.run_iter(idx, data_batch)

        # compute metrics
        self.evaluator.current_epoch = self.runner.epoch   # 设置当前的epoch
        self.evaluator.work_dir = self.runner.log_dir      # 设置日志路径
        metrics = self.evaluator.evaluate(len(self.dataloader.dataset))
        self.runner.call_hook('after_val_epoch', metrics=metrics)
        self.runner.call_hook('after_val')
        return metrics
    
    @torch.no_grad()
    def run_iter(self, idx, data_batch: Sequence[dict]):
        """Iterate one mini-batch.

        Args:
            data_batch (Sequence[dict]): Batch of data
                from dataloader.
        """
        self.runner.call_hook('before_val_iter', batch_idx=idx, data_batch=data_batch)

        # outputs should be sequence of BaseDataElement
        with autocast(enabled=self.fp16):
            outputs = self.runner.model.val_step(data_batch)

        self.evaluator.process(data_samples=outputs)
        self.runner.call_hook('after_val_iter', batch_idx=idx, data_batch=data_batch, outputs=outputs)