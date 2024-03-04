# -*- coding: utf-8 -*-
# @Author  : ace
# Copyright by Ace, All Rights Reserved.

import tempfile
import itertools
import numpy as np
import os.path as osp
import matplotlib.pyplot as plt

from typing import Dict, Sequence
from collections import OrderedDict
from terminaltables import AsciiTable

from mmengine.fileio import load
from mmengine import mkdir_or_exist
from mmengine.logging import MMLogger

from mmdet.datasets.api_wrappers import COCO, COCOeval
from mmdet.evaluation.metrics.coco_metric import CocoMetric

from ..registry import METRICS
from ..utils import heading_line


def softmax(x):
    max_num = np.max(x)

    x = x - max_num

    if len(x.shape) > 1:
        x=np.exp(x)/ np.sum(np.exp(x), axis=1).reshape(-1, 1)
    else:
        x=np.exp(x) / np.sum(np.exp(x))
        
    return x


@METRICS.register_module()
class AlchemyCocoDet2dMetric(CocoMetric):
    def __init__(self,
                 plt_pr_curve: bool = False,
                 max_dets: Sequence[int] = (10, 30, 100),
                 object_size: Sequence[float] = (32, 64, 1e5),
                 *args, **kwargs) -> None:
        
        self.plt_pr_curve = plt_pr_curve

        # max dets used to compute recall or precision.
        self.max_dets = list(max_dets)

        self.area_ranges = (
            [0 ** 2, object_size[2] ** 2], 
            [0 ** 2, object_size[0] ** 2], 
            [object_size[0] ** 2, object_size[1] ** 2], 
            [object_size[1] ** 2, object_size[2] ** 2]
            )
        
        # 获取每个类别的图片和bbox的数量
        self.cat_infos = None
        self.cat_weights = None
        
        self.work_dir = None
        self.current_epoch = -1  # 当前的epoch
        self.best_coco = dict()
        self.best_alchemy = dict()

        # 101 steps, from 0% to 100% recall.
        self.rec_interp = np.linspace(0, 1, 101)

        super().__init__(classwise=True, *args, **kwargs)
    
    def _obtain_cat_infos(self) -> dict:
        cat_infos = dict()

        for cat in self.dataset_meta['classes']:  # dataset_meta就是metainfo, 其classes的顺序要与annotation中一致
            cat_id = self._coco_api.get_cat_ids(cat_names=[cat])
            img_ids = self._coco_api.get_img_ids(cat_ids=cat_id)
            ann_ids = self._coco_api.get_ann_ids(img_ids=img_ids, cat_ids=cat_id, iscrowd=None)

            cat_infos[cat] = dict(anns=len(ann_ids), imgs=len(img_ids))

        class_counts = np.array([cat_infos[cat]['anns'] for cat in self.dataset_meta['classes']])  
        class_weights = softmax(np.log(class_counts + 20000))  # +20000是防止类别数量太少，导致权重为0

        for idx, cat in enumerate(cat_infos):
            cat_infos[cat]['weight'] = class_weights[idx]

        return cat_infos, class_weights.reshape(-1)

    def process(self, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of data samples that
                contain annotations and predictions.
        """
        for data_sample in data_samples:
            result = dict()
            pred = data_sample['pred_instances']
            result['img_id'] = data_sample['img_id']
            result['bboxes'] = pred['bboxes'].cpu().numpy()
            result['scores'] = pred['scores'].cpu().numpy()
            result['labels'] = pred['labels'].cpu().numpy()

            # parse gt
            gt = dict()
            gt['width'] = data_sample['ori_shape'][1]
            gt['height'] = data_sample['ori_shape'][0]
            gt['img_id'] = data_sample['img_id']
            gt['scale_factor'] = data_sample['scale_factor']

            if self._coco_api is None:
                assert 'instances' in data_sample, 'ground truth is required for evaluation when ann_file is not provided'
                gt_instances = data_sample['instances']
                gt_instances['gt_bboxes'][:, [0, 2]] /= gt['scale_factor'][0]
                gt_instances['gt_bboxes'][:, [1, 3]] /= gt['scale_factor'][1]

                gt['anns'] = gt_instances

            # add converted result to the results list
            self.results.append((gt, result))

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """
        Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()

        # split gt and prediction list
        gts, preds = zip(*results)

        tmp_dir = None
        if self.outfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            outfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            outfile_prefix = self.outfile_prefix

        if self._coco_api is None:
            # use converted gt json file to initialize coco api
            logger.info('Converting ground truth to coco format...')
            coco_json_path = self.gt_to_coco_json(gt_dicts=gts, outfile_prefix=outfile_prefix)
            self._coco_api = COCO(coco_json_path)

        # handle lazy init
        if self.cat_ids is None:
            self.cat_ids = self._coco_api.get_cat_ids(cat_names=self.dataset_meta['classes'])  # dataset_meta['classes']的顺序要与annotations['categories']的顺序一致

        if self.img_ids is None:
            self.img_ids = self._coco_api.get_img_ids()

        if self.cat_infos is None:
            self.cat_infos, self.cat_weights = self._obtain_cat_infos()

        # convert predictions to coco format and dump to json file
        result_files = self.results2json(preds, outfile_prefix)

        eval_results = OrderedDict()

        if self.format_only:
            logger.info(f'results are saved in {osp.dirname(outfile_prefix)}')
            return eval_results

        logger.info(f'Evaluating 2D BBox...')

        # evaluate proposal, bbox and segm
        if 'bbox' not in result_files:
            raise KeyError('bbox is not in results')

        try:
            predictions = load(result_files['bbox'])
            coco_dt = self._coco_api.loadRes(predictions)

        except IndexError:
            logger.error('The testing results of the whole dataset is empty.')
            return eval_results

        coco_eval = COCOeval(self._coco_api, coco_dt, 'bbox')

        coco_eval.params.catIds = self.cat_ids
        coco_eval.params.imgIds = self.img_ids
        coco_eval.params.iouThrs = self.iou_thrs

        # 设置大、中、小目标的面积
        coco_eval.params.areaRng = self.area_ranges

        # 设置最多检测框的数量
        coco_eval.params.maxDets = list(self.max_dets)

        # mapping of cocoEval.stats
        coco_metric_names = {
            'mAP': 0,
            'mAP@.5': 1,
            'mAP@.75': 2,
            'mAP@small': 3,
            'mAP@medium': 4,
            'mAP@large': 5,
            f'AR@{self.max_dets[0]}': 6,
            f'AR@{self.max_dets[1]}': 7,
            f'AR@{self.max_dets[2]}': 8,
            'AR@small': 9,
            'AR@medium': 10,
            'AR@large': 11
        }

        metric_items = self.metric_items
        if metric_items is not None:
            for metric_item in metric_items:
                if metric_item not in coco_metric_names:
                    raise KeyError(f'metric item "{metric_item}" is not supported')

        coco_eval.evaluate()
        coco_eval.accumulate()

        heading_line(log='COCO Metric', logger=logger)
        coco_eval.summarize()

        classwise_metrics = []
        headers = [
                'category', 'images', 'bboxes', 'weight', 'AP', 'AR', 'F1', 'AP@.5', 'R@.5', 'AP@.75', 'R@.75', 'AP@small', 'AP@medium', 'AP@large'
            ]

        # 获取每个类别的KPI
        if self.classwise:
            # Compute per-category AP from https://github.com/facebookresearch/detectron2/

            # precision: (iou, recall, cls, area range, max dets)
            precisions = coco_eval.eval['precision'] 

            # recall: (iou, cls, area range, max dets)
            recalls = coco_eval.eval['recall']
            
            assert len(self.cat_ids) == precisions.shape[2]

            results_per_category = []

            for idx, cat_id in enumerate(self.cat_ids):
                cls_metric = []

                # area range index 0: all area ranges
                # max dets index -1: typically 100 per image
                nm = self._coco_api.loadCats(cat_id)[0]

                cat_name = nm["name"]
                t = [f'{cat_name}', f'{self.cat_infos[cat_name]["imgs"]}', f'{self.cat_infos[cat_name]["anns"]}', f'{round(self.cat_infos[cat_name]["weight"], 2)}']

                # ap
                precision = precisions[:, :, idx, 0, -1]   # [10, 101]
                precision = precision[precision > -1]

                if precision.size:
                    ap = np.mean(precision)
                else:
                    ap = float('nan')

                t.append(f'{round(ap, 3)}')
                eval_results[f'{cat_name}_ap'] = round(ap, 3)
                cls_metric.append(round(ap, 3))

                # ar
                recall = recalls[:, idx, 0, -1]  # [10]
                recall = recall[recall > -1]

                if recall.size:
                    ar = np.mean(recall)
                else:
                    ar = float('nan')

                t.append(f'{round(ar, 3)}')
                eval_results[f'{cat_name}_ar'] = round(ar, 3)
                cls_metric.append(round(ar, 3))

                # F1
                if ap != np.nan and ar != np.nan:
                    f1 = 2 * ap * ar / (ap + ar + 1e-4)
                else:
                    f1 = float('nan')
                
                t.append(f'{round(f1, 3)}')
                eval_results[f'{cat_name}_f1'] = round(f1, 3)
                cls_metric.append(round(f1, 3))

                # AP@.5 & AP@.75
                for iou in [0, 5]:
                    precision = precisions[iou, :, idx, 0, -1]
                    precision = precision[precision > -1]

                    r = recalls[iou, idx, 0, -1]
                    r = max(r, 0)

                    if precision.size:
                        ap = np.mean(precision)

                        if self.plt_pr_curve:
                            iou_thr = 'IoU@.5' if iou == 0 else 'IoU@.75'
                            work_dir = osp.join(self.work_dir, 'pr_curvs', cat_name, iou_thr)
                            mkdir_or_exist(work_dir)

                            plt.clf()
                            plt.plot(self.rec_interp.copy(), precision)
                            plt.xlabel('recall')
                            plt.ylabel('precision')
                            plt.title('%s-%s:' % (cat_name, iou_thr))
                            plt.savefig(osp.join(work_dir, f'{cat_name}_epoch{self.current_epoch}_{iou_thr}_pr.jpg'))

                    else:
                        ap = float('nan')

                    t.append(f'{round(ap, 3)}')
                    t.append(f'{round(r, 3)}')
                    cls_metric.append(round(ap, 3))
                    cls_metric.append(round(r, 3))

                # indexes of area of small, median and large
                for area in [1, 2, 3]:
                    precision = precisions[:, :, idx, area, -1]
                    precision = precision[precision > -1]
                    if precision.size:
                        ap = np.mean(precision)
                    else:
                        ap = float('nan')
                    t.append(f'{round(ap, 3)}')
                    cls_metric.append(round(ap, 3))

                results_per_category.append(tuple(t))
                classwise_metrics.append(cls_metric)

            num_columns = len(results_per_category[0])
            results_flatten = list(itertools.chain(*results_per_category))

            results_2d = itertools.zip_longest(*[results_flatten[i::num_columns] for i in range(num_columns)])

            table_data = [headers]
            table_data += [result for result in results_2d]
            table = AsciiTable(table_data)

            heading_line(log='Class Metric', logger=logger)
            for line in table.table.split('\n'):
                logger.info(line)

        if metric_items is None:
            # metric_items = ['mAP', 'mAP@.5', 'mAP@.75', 'mAP@small', 'mAP@medium', 'mAP@large']
            metric_items = list(coco_metric_names.keys())

        heading_line(log='Average Metric (coco)', logger=logger)
        for metric_item in metric_items:
            val = float(f'{round(coco_eval.stats[coco_metric_names[metric_item]], 3)}')
            eval_results[metric_item] = val

            # best
            if metric_item not in self.best_coco:
                self.best_coco[metric_item] = (self.current_epoch, val)
            else:
                if val > self.best_coco[metric_item][1]:
                    self.best_coco[metric_item] = (self.current_epoch, val)

            # print
            current_str = '{: <20}'.format(f'{metric_item}: {val}')
            best_str = f'(best: {self.best_coco[metric_item][1]} [{self.best_coco[metric_item][0]}])'
            logger.info(f'{current_str}{best_str}')
        
        # mF1
        f1 = round(2 * coco_eval.stats[0] * coco_eval.stats[8] / (coco_eval.stats[0] + coco_eval.stats[8] + 1e-4), 3)
        if 'mF1' not in self.best_coco:
            self.best_coco['mF1'] = (self.current_epoch, f1)
        else:
            if f1 > self.best_coco['mF1'][1]:
                self.best_coco['mF1'] = (self.current_epoch, f1)

        current_str = '{: <20}'.format(f'mF1: {f1}')
        best_str = f'(best: {self.best_coco["mF1"][1]} [{self.best_coco["mF1"][0]}])'
        logger.info(f'{current_str}{best_str}')

        eval_results['mF1'] = f1

        if len(classwise_metrics):
            heading_line(log='Average Metric (alchemy)', logger=logger)
            classwise_metrics = np.nan_to_num(np.array(classwise_metrics).reshape(-1, len(headers[4:])))
            for col_id, metric_item in enumerate(headers[4:]):
                val = round(float(np.sum(classwise_metrics[:, col_id] * self.cat_weights)), 3)

                # best
                if metric_item not in self.best_alchemy:
                    self.best_alchemy[metric_item] = (self.current_epoch, val)
                else:
                    if val > self.best_alchemy[metric_item][1]:
                        self.best_alchemy[metric_item] = (self.current_epoch, val)

                # print
                current_str = '{: <20}'.format(f'm{metric_item}: {val}')
                best_str = f'(best: {self.best_alchemy[metric_item][1]} [{self.best_alchemy[metric_item][0]}])'
                logger.info(f'{current_str}{best_str}')

        if tmp_dir is not None:
            tmp_dir.cleanup()

        return eval_results