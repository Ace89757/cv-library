# -*- coding: utf-8 -*-
# @Author  : ace
# Copyright by Ace, All Rights Reserved. 

import os
import logging

from copy import deepcopy
from typing import List, Union

from mmengine.logging import print_log
from mmengine.fileio import get_local_path
from mmdet.datasets.api_wrappers import COCO

from ..registry import DATASETS
from .base_dataset import AlchemyBaseDataset


@DATASETS.register_module()
class AlchemyDet2dDataset(AlchemyBaseDataset):
    DATASET_TYPE = 'Alchemy 2D Dataset (Follow COCO)'

    def parse_data_info(self, img_id: int) -> Union[dict, List[dict]]:
        """
        Parse raw annotation to target format.

        Args:
            raw_data_info (dict): Raw data information load from ``ann_file``

        Returns:
            Union[dict, List[dict]]: Parsed annotation.
        """
        img_info = self.coco.load_imgs([img_id])[0]

        ann_ids = self.coco.get_ann_ids(img_ids=[img_id])

        data_info = {}

        img_path = os.path.join(self.data_prefix['img_path'], img_info['file_name'])
        if not os.path.exists(img_path):
            return None

        data_info['img_id'] = img_id
        data_info['img_path'] = img_path
        
        data_info['width'] = img_info['width']
        data_info['height'] = img_info['height']
        
        if self.return_classes:
            data_info['text'] = self.metainfo['classes']
            data_info['custom_entities'] = True

        instances = []
        for ann in self.coco.load_anns(ann_ids):
            instance = {}

            if ann.get('ignore', False):
                continue

            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))

            if inter_w * inter_h == 0:
                continue

            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            
            category_id = ann['category_id']
            if category_id not in self.cat_ids:
                continue

            bbox = [x1, y1, x1 + w, y1 + h]

            if ann.get('iscrowd', False):
                instance['ignore_flag'] = 1
            else:
                instance['ignore_flag'] = 0

            instance['bbox'] = bbox
            instance['bbox_label'] = self.cat2label[category_id]

            instances.append(instance)

        data_info['instances'] = instances

        return data_info

    def load_data_list(self) -> List[dict]:
        """
        
        """
        assert os.path.exists(self.ann_file), FileExistsError(f'{self.ann_file} is not exists!!')

        # load & parse data
        print_log(f'loading and parsing dataset, ann_file: "{self.ann_file}"', logger='current', level=logging.INFO)

        with get_local_path(self.ann_file, backend_args=self.backend_args) as local_path:
            self.coco = COCO(local_path)
        
        # {class_name: class_id}
        self._dataset_category = {cat['name']: cat['id'] for cat in self.coco.dataset['categories']}
        
        # cat_ids的顺序不会随着metainfo['classes']中的顺序改变
        self.cat_ids = [idx for name, idx in self._dataset_category.items() if name in self.metainfo['classes']]
        self.cat_names ={cat_id: name for name, cat_id in self._dataset_category.items()}
        
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}            # cat_id -> lable
        self.label2cat = {label: cat_id for cat_id, label in self.cat2label.items()}     # label -> cat_id

        # img_ids for every category
        self.cat_img_map = deepcopy(self.coco.cat_img_map)

        data_list = []

        for img_id in self.coco.get_img_ids():
            data_info = self.parse_data_info(img_id)
            if data_info is None:
                continue

            data_list.append(data_info)

        del self.coco

        return data_list

    def filter_data(self) -> List[dict]:
        """
        Filter annotations according to filter_cfg.

        Returns:
            List[dict]: Filtered results.
        """
        if self.test_mode or self.filter_cfg is None:
            valid_data_infos = self.data_list
        else:
            valid_data_infos = []

            min_size = self.filter_cfg.get('min_size', 0)   # 有效图片尺寸
            filter_empty_gt = self.filter_cfg.get('filter_empty_gt', False)   # 过滤没有gt的图片
            
            # 获取所有加载出来的图片id
            img_ids_with_ann = set(data_info['img_id'] for data_info in self.data_list)

            # 获取所有类别包含的图片id
            img_ids_in_cat = set()

            for class_id in self.cat_ids:
                img_ids_in_cat |= set(self.cat_img_map[class_id])

            # merge the image id sets of the two conditions and use the merged set to filter out images if self.filter_empty_gt=True
            img_ids_in_cat &= img_ids_with_ann

            for data_info in self.data_list:

                width = data_info['width']
                height = data_info['height']
                img_id = data_info['img_id']

                if filter_empty_gt and img_id not in img_ids_in_cat:
                    continue

                if min(width, height) >= min_size:
                    valid_data_infos.append(data_info)

        # 记录每个类别的数量
        for data_info in valid_data_infos:
            for instance in data_info['instances']:
                label = instance['bbox_label']
                category_id = self.label2cat[label]
                cat_name = self.cat_names[category_id]

                if cat_name not in self.category_info:
                    self.category_info[cat_name] = dict(counts=1, category_id=label)
                else:
                    self.category_info[cat_name]['counts'] += 1

        return valid_data_infos
    
    @property
    def dataset_category(self) -> None:
        return deepcopy(self._dataset_category)
