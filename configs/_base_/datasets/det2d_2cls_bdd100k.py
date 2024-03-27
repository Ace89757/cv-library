# -*- coding: utf-8 -*-
# @Author  : ace
# Copyright by Ace, All Rights Reserved.

"""
dataset settings
"""
dataset_type = 'AlchemyDet2dDataset'
data_root = 'E:/dl/datasets/bdd100k/'

train_ann_file = 'labels/det_20/det_train_2cls_61824_coco.json'
val_ann_file = 'labels/det_20/det_val_2cls_8853_coco.json'
test_ann_file = 'labels/det_20/det_val_2cls_8853_coco.json'

metainfo = {
        # 保证与annotation['categories']顺序一致
        'classes': ('traffic sign', 'traffic light'),
        'palette': [(255, 165, 79), (178, 58, 238)]
    }

num_classes = len(metainfo['classes'])


"""
evaluator
"""
val_evaluator = dict(
    type='AlchemyEvaluator',
    metrics=[
        dict(
            type='AlchemyCocoDet2dMetric',
            ann_file=data_root + val_ann_file,
            max_dets=(10, 30, 100),       # 每张图片, 按置信度排序, 排名前10、前30、前100个预测框的指标
            object_size=(8, 96, 1e5),     # 表示小、中、大目标的边长
            format_only=False,
            plt_pr_curve=True
        )
    ]
    )

test_evaluator = val_evaluator


score_thr = 0.35