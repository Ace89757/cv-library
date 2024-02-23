# -*- coding: utf-8 -*-
# @Author  : ace
# Copyright by Ace, All Rights Reserved.

import os
import logging

from collections.abc import Mapping
from typing import Any, List, Union
from terminaltables import AsciiTable

from mmengine.config import Config
from mmengine.logging import print_log
from mmengine.dataset import BaseDataset
from mmengine.fileio import list_from_file, load

from ..utils import heading_line


class AlchemyBaseDataset(BaseDataset):
    DATASET_TYPE = 'Alchemy Base Dataset'

    def __init__(self, *args, **kwargs):
        self.category_info = dict()
        super().__init__(*args, **kwargs)

        # load func
        self.load_func = self.load_test_data if self.test_mode else self.load_train_data

        # 显示数据集信息
        self.print_dataset
    
    @classmethod
    def _load_metainfo(cls, metainfo: Union[Mapping, Config, None] = None) -> dict:
        """
        从'metainfo'字典中收集信息.

        Args:
            metainfo (Mapping or Config, optional): Meta information dict.

        Returns:
            dict: Parsed meta information.
        """
        assert metainfo is not None, ValueError(f'metainfo must be definend. but got None.')

        _metainfo = dict()

        if not isinstance(metainfo, (Mapping, Config)):
            raise TypeError(f'metainfo should be a Mapping or Config, but got {type(metainfo)}')
        
        for meta_key, meta_val in metainfo.items():
            if isinstance(meta_val, str):
                # If type of value is string, and can be loaded from corresponding backend. it means the file name of meta file.
                try:
                    _metainfo[meta_key] = list_from_file(meta_val)

                except (TypeError, FileNotFoundError):
                    print_log(f'{meta_val} is not a meta file, simply parsed as meta information', logger='current', level=logging.WARNING)
                    _metainfo[meta_key] = meta_val
            else:
                _metainfo[meta_key] = meta_val
        
        return _metainfo

    def load_data_list(self) -> List[dict]:
        assert os.path.exists(self.ann_file), FileExistsError(f'{self.ann_file} is not exists!!')

        # load
        annotations = load(self.ann_file)

        if not isinstance(annotations, dict):
            raise TypeError(f'The annotations loaded from annotation file should be a dict, but got {type(annotations)}!')
        
        assert 'annotations' in annotations, KeyError('Annotation must have "annotations" key')
        
        if 'metainfo' in annotations:
            metainfo = annotations['metainfo']

            # anntations中的'metainfo'不会影响已经存在在self._metainfo中的key
            for k, v in metainfo.items():
                self._metainfo.setdefault(k, v)
        
        # load & parse data
        print_log(f'loading and parsing dataset, ann_file: "{self.ann_file}"', logger='current', level=logging.INFO)

        return self.parse_dataset(annotations['annotations'])
    
    def parse_dataset(self, raw_data_list: List[dict]) -> List[dict]:
        data_list = []

        for raw_data_info in raw_data_list:
            # parse raw data information to target format
            data_info = self.parse_data_info(raw_data_info)

            if isinstance(data_info, dict):
                # For image tasks, 'data_info' should information if single
                # image, such as dict(img_path='xxx', width=360, ...)
                data_list.append(data_info)

            elif isinstance(data_info, list):
                # For video tasks, 'data_info' could contain image information of multiple frames, such as
                # [
                #   dict(video_path='xxx', timestamps=...),
                #   dict(video_path='xxx', timestamps=...)
                # ]
                for item in data_info:
                    if not isinstance(item, dict):
                        raise TypeError(f'data_info must be list of dict, but got {type(item)}')

                data_list.extend(data_info)
            else:
                raise TypeError(f'data_info should be a dict or list of dict, but got {type(data_info)}')
        
        return data_list

    def load_train_data(self, idx: int) -> Any:
        for _ in range(self.max_refetch + 1):
            data = self.prepare_data(idx)

            # Broken images or random augmentations may cause the returned data to be None
            if data is None:
                idx = self._rand_another()
                continue

            return data

        raise Exception(f'Cannot find valid image after {self.max_refetch}! Please check your image path and pipeline')
    
    def load_test_data(self, idx: int) -> Any:
        data = self.prepare_data(idx)

        if data is None:
            raise Exception('Test time pipline should not get "None" data_sample')
        
        return data

    def __getitem__(self, idx: int) -> Any:
        if not self._fully_initialized:
            print_log('Please call "full_init()" method manually to accelerate the speed.', logger='current', level=logging.WARNING)
            self.full_init()

        return self.load_func(idx)

    @property
    def print_dataset(self) -> None:
        heading_line(self.DATASET_TYPE)

        print_log(f'data root: {self.data_root}', logger='current', level=logging.INFO)
        print_log(f'test mode: {self.test_mode}', logger='current', level=logging.INFO)
        # print_log(f'annotation file: {self.ann_file}', logger='current', level=logging.INFO)

        print_log(f'The number of instances per category in the dataset:', logger='current', level=logging.INFO)

        content_show = [['id', 'category', 'counts']]
        for cat_name, cat_info in self.category_info.items():
            content_show.append([cat_info['category_id'], cat_name, cat_info['counts']])

        table = AsciiTable(content_show)

        for line in table.table.split('\n'):
            print_log(line, logger='current', level=logging.INFO)

        print_log(f'found {len(self)} samples, from "{self.ann_file}"', logger='current', level=logging.INFO)