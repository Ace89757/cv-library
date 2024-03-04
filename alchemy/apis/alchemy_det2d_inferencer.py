# -*- coding: utf-8 -*-
# @Author  : ace
# Copyright by Ace, All Rights Reserved.

import time
import copy
import mmcv
import torch
import logging
import mmengine
import warnings
import numpy as np
import torch.nn as nn
import os.path as osp

from tqdm import tqdm
from datetime import datetime
from typing import (Dict, Iterable, List, Optional, Sequence, Tuple, Union)

from mmdet.structures import DetDataSample
from mmdet.evaluation import get_classes

from mmcv.transforms import LoadImageFromFile

from mmengine.config import Config, ConfigDict


from mmengine.dataset import Compose
from mmengine.logging import print_log
from mmengine.device import get_device

from mmengine.registry import DefaultScope
from mmengine.dataset import pseudo_collate
from mmengine.structures import InstanceData
from mmengine.visualization import Visualizer
from mmengine.registry import init_default_scope
from mmengine.model.utils import revert_sync_batchnorm
from mmengine.fileio import (get_file_backend, isdir, join_path, list_dir_or_file)
from mmengine.runner.checkpoint import (_load_checkpoint, _load_checkpoint_to_model)

from alchemy.utils import heading_line
from alchemy.registry import DATASETS, VISUALIZERS, MODELS


InstanceList = List[InstanceData]
InputType = Union[str, np.ndarray, torch.Tensor]
InputsType = Union[InputType, Sequence[InputType]]
ImgType = Union[np.ndarray, Sequence[np.ndarray]]
ResType = Union[Dict, List[Dict]]
ConfigType = Union[Config, ConfigDict]
ModelType = Union[dict, ConfigType, str]
PredType = List[DetDataSample]

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')
COLLATE_FUNC = pseudo_collate


class AlchemyDet2dInferencer:
    def __init__(self,
                 model: Optional[Union[ModelType, str]] = None,
                 weights: Optional[str] = None,
                 device: Optional[str] = None,
                 scope: Optional[str] = 'alchemy',
                 out_dir: str = None,
                 palette: str = 'none') -> None:
        # scope
        self.scope = self._set_scope(scope)

        # load config to cfg
        self.cfg = self._load_config(model)

        # device
        if device is None:
            device = get_device()

        # palette
        self.palette = palette
        
        # model
        self.model = self._init_model(weights, device)
        self.model = revert_sync_batchnorm(self.model)

        # pipeline
        self.pipeline = self._init_pipeline()

        # 如果没设置out_dir, 将检测结果保存到模型权重文件夹
        if out_dir is None:
            if weights is not None:
                out_dir = osp.join(osp.dirname(weights), 'det2d_preds')
            else:
                out_dir = 'outputs'

        self.out_dir = out_dir
        self.img_out_dir = osp.join(out_dir, 'imgs')
        self.pred_out_dir = osp.join(out_dir, 'preds')
        mmengine.mkdir_or_exist(self.img_out_dir)
        mmengine.mkdir_or_exist(self.pred_out_dir)

        # visualizer
        self.visualizer = self._init_visualizer()

        # args
        self.num_predicted_imgs = 0
        self.num_visualized_imgs = 0

    def _set_scope(self, scope: Optional[str]) ->  Optional[str]:
        init_default_scope(scope)

        if scope is None:
            default_scope = DefaultScope.get_current_instance()
            if default_scope is not None:
                scope = default_scope.scope_name
        
        return scope
    
    def _load_config(self, model: Optional[Union[ModelType, str]]) -> ConfigType:
        cfg: ConfigType
        if isinstance(model, str):
            if osp.isfile(model):
                cfg = Config.fromfile(model)
            else:
                # Load config and weights from metafile. If `weights` is
                # assigned, the weights defined in metafile will be ignored.
                cfg, _weights = self._load_model_from_metafile(model)
                if weights is None:
                    weights = _weights
        elif isinstance(model, (Config, ConfigDict)):
            cfg = copy.deepcopy(model)
        elif isinstance(model, dict):
            cfg = copy.deepcopy(ConfigDict(model))
        elif model is None:
            if weights is None:
                raise ValueError('If model is None, the weights must be specified since the config needs to be loaded from the weights')
            cfg = ConfigDict()
        else:
            raise TypeError(f'model must be a filepath or any ConfigType object, but got {type(model)}')
        
        return cfg

    def _init_model(self, weights: Optional[str], device: str = 'cpu') -> nn.Module:
        """
        Initialize the model with the given config and checkpoint on the specific device.

        Args:
            weights (str, optional): Path to the checkpoint.
            device (str, optional): Device to run inference. Defaults to 'cpu'.

        Returns:
            nn.Module: Model loaded with checkpoint.
        """
        heading_line('Initial Model')
        checkpoint: Optional[dict] = None
        if weights is not None:
            checkpoint = _load_checkpoint(weights, map_location='cpu')

        if not self.cfg:
            assert checkpoint is not None
            try:
                # Prefer to get config from `message_hub` since `message_hub`
                # is a more stable module to store all runtime information.
                # However, the early version of MMEngine will not save config
                # in `message_hub`, so we will try to load config from `meta`.
                cfg_string = checkpoint['message_hub']['runtime_info']['cfg']
            except KeyError:
                assert 'meta' in checkpoint, (
                    'If model(config) is not provided, the checkpoint must'
                    'contain the config string in `meta` or `message_hub`, '
                    'but both `meta` and `message_hub` are not found in the '
                    'checkpoint.')
                meta = checkpoint['meta']
                if 'cfg' in meta:
                    cfg_string = meta['cfg']
                else:
                    raise ValueError(
                        'Cannot find the config in the checkpoint.')
            self.cfg.update(
                Config.fromstring(cfg_string, file_format='.py')._cfg_dict)

        # Delete the `pretrained` field to prevent model from loading the
        # the pretrained weights unnecessarily.
        if self.cfg.model.get('pretrained') is not None:
            del self.cfg.model.pretrained

        model = MODELS.build(self.cfg.model)
        model.cfg = self.cfg
        self._load_weights_to_model(model, checkpoint, self.cfg)
        model.to(device)
        model.eval()
        return model
    
    def _load_weights_to_model(self, model: nn.Module, checkpoint: Optional[dict], cfg: Optional[ConfigType]) -> None:
        """
        Loading model weights and meta information from cfg and checkpoint.

        Args:
            model (nn.Module): Model to load weights and meta information.
            checkpoint (dict, optional): The loaded checkpoint.
            cfg (Config or ConfigDict, optional): The loaded config.
        """

        if checkpoint is not None:
            _load_checkpoint_to_model(model, checkpoint)
            checkpoint_meta = checkpoint.get('meta', {})

            if 'epoch' in checkpoint_meta:
                print_log(msg=f'checkpoint epoch: {checkpoint_meta["epoch"]}', logger='current', level=logging.INFO)

            # save the dataset_meta in the model for convenience
            if 'dataset_meta' in checkpoint_meta:
                # mmdet 3.x, all keys should be lowercase
                model.dataset_meta = {
                    k.lower(): v
                    for k, v in checkpoint_meta['dataset_meta'].items()
                }
            elif 'CLASSES' in checkpoint_meta:
                # < mmdet 3.x
                classes = checkpoint_meta['CLASSES']
                model.dataset_meta = {'classes': classes}
            else:
                warnings.warn(
                    'dataset_meta or class names are not saved in the '
                    'checkpoint\'s meta data, use COCO classes by default.')
                model.dataset_meta = {'classes': get_classes('coco')}
            
        else:
            warnings.warn('Checkpoint is not loaded, and the inference '
                          'result is calculated by the randomly initialized '
                          'model!')
            warnings.warn('weights is None, use COCO classes by default.')
            model.dataset_meta = {'classes': get_classes('coco')}
        print_log(msg=f'model classes: {model.dataset_meta["classes"]}', logger='current', level=logging.INFO)

        # Priority:  args.palette -> config -> checkpoint
        if self.palette != 'none':
            model.dataset_meta['palette'] = self.palette
        else:
            test_dataset_cfg = copy.deepcopy(cfg.test_dataloader.dataset)

            # lazy init. We only need the metainfo.
            test_dataset_cfg['lazy_init'] = True
            metainfo = DATASETS.build(test_dataset_cfg).metainfo
            cfg_palette = metainfo.get('palette', None)
            if cfg_palette is not None:
                model.dataset_meta['palette'] = cfg_palette
            else:
                if 'palette' not in model.dataset_meta:
                    warnings.warn(
                        'palette does not exist, random is used by default. '
                        'You can also set the palette to customize.')
                    model.dataset_meta['palette'] = 'random'

    def _init_pipeline(self) -> Compose:
        """Initialize the test pipeline."""
        pipeline_cfg = self.cfg.test_dataloader.dataset.pipeline

        # For inference, the key of ``img_id`` is not used.
        if 'meta_keys' in pipeline_cfg[-1]:
            pipeline_cfg[-1]['meta_keys'] = tuple(
                meta_key for meta_key in pipeline_cfg[-1]['meta_keys']
                if meta_key != 'img_id')

        load_img_idx = self._get_transform_idx(pipeline_cfg, ('mmdet.LoadImageFromFile', LoadImageFromFile))

        if load_img_idx == -1:
            raise ValueError('LoadImageFromFile is not found in the test pipeline')
        pipeline_cfg[load_img_idx]['type'] = 'mmdet.InferencerLoader'
        return Compose(pipeline_cfg)

    def _init_visualizer(self) -> Optional[Visualizer]:
        """Initialize visualizers.

        Args:
            cfg (ConfigType): Config containing the visualizer information.

        Returns:
            Visualizer or None: Visualizer initialized with config.
        """
        if 'visualizer' not in self.cfg:
            return None
        timestamp = str(datetime.timestamp(datetime.now()))
        name = self.cfg.visualizer.get('name', timestamp)
        if Visualizer.check_instance_created(name):
            name = f'{name}-{timestamp}'
        self.cfg.visualizer.name = name
        self.cfg.visualizer.save_dir = self.out_dir
        visualizer = VISUALIZERS.build(self.cfg.visualizer)
        visualizer.dataset_meta = self.model.dataset_meta
        return visualizer
    
    def _get_transform_idx(self, pipeline_cfg: ConfigType, name: Union[str, Tuple[str, type]]) -> int:
        """Returns the index of the transform in a pipeline.

        If the transform is not found, returns -1.
        """
        for i, transform in enumerate(pipeline_cfg):
            if transform['type'] in name:
                return i
        return -1
    
    def _inputs_to_list(self, inputs: InputsType) -> list:
        """
        Preprocess the inputs to a list.

        Preprocess inputs to a list according to its type:

        - list or tuple: return inputs
        - str:
            - Directory path: return all files in the directory
            - other cases: return a list containing the string. The string
              could be a path to file, a url or other types of string according
              to the task.

        Args:
            inputs (InputsType): Inputs for the inferencer.

        Returns:
            list: List of input for the :meth:`preprocess`.
        """
        if isinstance(inputs, str):
            backend = get_file_backend(inputs)
            if hasattr(backend, 'isdir') and isdir(inputs):
                # Backends like HttpsBackend do not implement `isdir`, so only
                # those backends that implement `isdir` could accept the inputs
                # as a directory
                filename_list = list_dir_or_file(
                    inputs, list_dir=False, suffix=IMG_EXTENSIONS)
                inputs = [
                    join_path(inputs, filename) for filename in filename_list
                ]

        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]

        return list(inputs)
    
    def _preprocess(self, inputs: InputsType, batch_size: int = 1):
        """
        Process the inputs into a model-feedable format.

        Customize your preprocess by overriding this method. Preprocess should
        return an iterable object, of which each item will be used as the
        input of ``model.test_step``.

        ``BaseInferencer.preprocess`` will return an iterable chunked data,
        which will be used in __call__ like this:

        .. code-block:: python

            def __call__(self, inputs, batch_size=1, **kwargs):
                chunked_data = self.preprocess(inputs, batch_size, **kwargs)
                for batch in chunked_data:
                    preds = self.forward(batch, **kwargs)

        Args:
            inputs (InputsType): Inputs given by user.
            batch_size (int): batch size. Defaults to 1.

        Yields:
            Any: Data processed by the ``pipeline`` and ``collate_fn``.
        """
        chunked_data = self._get_chunk_data(inputs, batch_size)
        yield from map(COLLATE_FUNC, chunked_data)

    def _get_chunk_data(self, inputs: Iterable, chunk_size: int):
        """Get batch data from inputs.

        Args:
            inputs (Iterable): An iterable dataset.
            chunk_size (int): Equivalent to batch size.

        Yields:
            list: batch data.
        """
        inputs_iter = iter(inputs)
        while True:
            try:
                chunk_data = []
                for _ in range(chunk_size):
                    inputs_ = next(inputs_iter)
                    if isinstance(inputs_, dict):
                        if 'img' in inputs_:
                            ori_inputs_ = inputs_['img']
                        else:
                            ori_inputs_ = inputs_['img_path']
                        chunk_data.append((ori_inputs_, self.pipeline(copy.deepcopy(inputs_))))
                    else:
                        chunk_data.append((inputs_, self.pipeline(inputs_)))
                yield chunk_data

            except StopIteration:
                if chunk_data:
                    yield chunk_data
                break

    def _visualize(self, inputs: InputsType, preds: PredType, show: bool = False, wait_time: int = 0, pred_score_thr: float = 0.3) -> Union[List[np.ndarray], None]:
        """
        Visualize predictions.

        Args:
            inputs (List[Union[str, np.ndarray]]): Inputs for the inferencer.
            preds (List[:obj:`DetDataSample`]): Predictions of the model.
            show (bool): Whether to display the image in a popup window. Defaults to False.
            wait_time (float): The interval of show (s). Defaults to 0.
            pred_score_thr (float): Minimum score of bboxes to draw. Defaults to 0.3.

        Returns:
            List[np.ndarray] or None: Returns visualization results only if applicable.
        """
        if self.visualizer is None:
            raise ValueError('Visualization needs the "visualizer" term defined in the config, but got None.')

        results = []

        for single_input, pred in zip(inputs, preds):
            if isinstance(single_input, str):
                img_bytes = mmengine.fileio.get(single_input)
                img = mmcv.imfrombytes(img_bytes)
                img = img[:, :, ::-1]
                img_name = osp.basename(single_input)
            elif isinstance(single_input, np.ndarray):
                img = single_input.copy()
                img_num = str(self.num_visualized_imgs).zfill(8)
                img_name = f'{img_num}.jpg'
            else:
                raise ValueError(f'Unsupported input type: {type(single_input)}')

            out_file = osp.join(self.img_out_dir, img_name)

            self.visualizer.add_datasample(
                img_name,
                img,
                pred,
                show=show,
                wait_time=wait_time,
                draw_gt=False,
                draw_pred=True,
                pred_score_thr=pred_score_thr,
                out_file=out_file,
            )

            results.append(self.visualizer.get_image())
            self.num_visualized_imgs += 1

        return results
    
    def _postprocess(self, preds: PredType, result_imgs: Optional[List[np.ndarray]] = None) -> Dict:
        """
        Process the predictions and visualization results from ``forward``
        and ``visualize``.

        This method should be responsible for the following tasks:

        1. Convert datasamples into a json-serializable dict if needed.
        2. Pack the predictions and visualization results and return them.
        3. Dump or log the predictions.

        Args:
            preds (List[:obj:`DetDataSample`]): Predictions of the model.
            visualization (Optional[np.ndarray]): Visualized predictions.

        Returns:
            dict: Inference and visualization results with key ``predictions``
            and ``visualization``.

            - ``visualization`` (Any): Returned by :meth:`visualize`.
            - ``predictions`` (dict or DataSample): Returned by
                :meth:`forward` and processed in :meth:`postprocess`.
                If ``return_datasamples=False``, it usually should be a
                json-serializable dict containing only basic data elements such
                as strings and numbers.
        """
        result_dict = {}

        results = []
        for pred in preds:
            result = self._pred2dict(pred)
            results.append(result)
            
        # Add img to the results after printing and dumping
        result_dict['predictions'] = results
        result_dict['visualization'] = result_imgs
        return result_dict
    
    def _pred2dict(self, data_sample: DetDataSample) -> Dict:
        """
        Extract elements necessary to represent a prediction into a dictionary.

        It's better to contain only basic data elements such as strings and
        numbers in order to guarantee it's json-serializable.

        Args:
            data_sample (:obj:`DetDataSample`): Predictions of the model.

        Returns:
            dict: Prediction results.
        """
        if 'img_path' in data_sample:
            img_name = osp.basename(data_sample.img_path)
            img_name = osp.splitext(img_name)[0]
            out_json_path = osp.join(self.pred_out_dir, img_name + '.json')
        else:
            out_json_path = osp.join(self.pred_out_dir, f'{self.num_predicted_imgs}.json')
            self.num_predicted_imgs += 1

        result = {}
        if 'pred_instances' in data_sample:
            pred_instances = data_sample.pred_instances.numpy()

            result = {
                'labels': pred_instances.labels.tolist(),
                'scores': pred_instances.scores.tolist()
            }

            if 'bboxes' in pred_instances:
                result['bboxes'] = pred_instances.bboxes.tolist()

        mmengine.dump(result, out_json_path)

        return result
    
    def __call__(self, inputs: InputsType, batch_size: int = 1, show: bool = False, wait_time: int = 0, pred_score_thr: float = 0.3) -> dict:
        
        # 待检图片列表
        imgs_list = self._inputs_to_list(inputs)

        # currently only supports bs=1
        assert batch_size == 1, ValueError('currently only supports bs=1')
        inputs = self._preprocess(imgs_list, batch_size=batch_size)

        time_spent = []

        results_dict = {'predictions': [], 'visualization': []}

        with tqdm(total=len(imgs_list), ncols=150, desc=f'Det2d Inference (bs={batch_size})', unit='batch') as progress_bar:
            for ori_imgs, batch_data in inputs:   # ori_imgs是每个batch的图片路径列表
                # 推理 & 后处理
                start = time.perf_counter()
                
                with torch.no_grad():
                    batch_preds = self.model.test_step(batch_data)
                
                time_spent.append((time.perf_counter() - start) * 1000)

                # 渲染 & 保存渲染后的图片
                result_imgs = self._visualize(ori_imgs, batch_preds, show=show, wait_time=wait_time, pred_score_thr=pred_score_thr)

                # 保存检测结果  results是个字典，包含可视化结果图像和预测结果
                results = self._postprocess(batch_preds, result_imgs)

                results_dict['predictions'].extend(results['predictions'])
                if results['visualization'] is not None:
                    results_dict['visualization'].extend(results['visualization'])
                
                progress_bar.set_postfix(time=f'{round(np.mean(time_spent), 2)}ms')
                progress_bar.update(1)
            
        progress_bar.close()

        return results_dict
