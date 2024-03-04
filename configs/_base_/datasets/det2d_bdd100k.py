# -*- coding: utf-8 -*-
# @Author  : ace
# Copyright by Ace, All Rights Reserved.

"""
dataset settings
"""
dataset_type = 'AlchemyDet2dDataset'
data_root = 'E:/dl/datasets/bdd100k/'

train_ann_file = 'labels/det_20/det_train_13cls_69853_coco.json'
val_ann_file = 'labels/det_20/det_val_13cls_10000_coco.json'
test_ann_file = 'labels/det_20/det_val_13cls_10000_coco.json'

metainfo = {
        # 保证与annotation['categories']顺序一致
        'classes': ('car', 'bus', 'truck', 'rider', 'bicycle', 'motorcycle', 'pedestrian', 'other person', 'traffic sign', 'traffic light', 'train', 'other vehicle', 'trailer'),
        'palette': [
            (238, 180, 34), 
            (205, 205, 0), 
            (102, 205, 0), 
            (141, 238, 238),  
            (67, 205, 128), 
            (0, 206, 209), 
            (188, 208, 104),
            (178, 34, 34), 
            (138, 43, 226), 
            (46, 139, 87),
            (196, 172, 0), 
            (95, 54, 80), 
            (128, 76, 255),
            ]
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
            max_dets=(10, 30, 100),   # 每张图片, 按置信度排序, 排名前10、前30、前100个预测框的指标
            object_size=(32, 64, 1e5),     # 表示小、中、大目标的边长
            format_only=False,
            plt_pr_curve=True
        )
    ]
    )

test_evaluator = val_evaluator
