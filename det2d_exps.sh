#!/usr/bin/env bash

WORK_DIR='E:/dl/weights'

source activate cv

# fcos
python tools/train.py configs/models/det2d/fcos/fcos_shufflenetv2_36e.py --prefix bdd100k/fcos/fcos_shufflenetv2_36e --work-dir ${WORK_DIR}
sleep 10m

# yolox
python tools/train.py configs/models/det2d/yolox/yolox_shufflenetv2_36e.py --prefix bdd100k/yolox/yolox_shufflenetv2_36e --work-dir ${WORK_DIR}
sleep 10m

# # retinanet
# python tools/train.py configs/models/det2d/retinanet/retinanet_shufflenetv2_36e.py --prefix bdd100k/tslr/retinanet/retinanet_shufflenetv2_36e --work-dir ${WORK_DIR}
# sleep 10m

# centernet
python tools/train.py configs/models/det2d/centernet/centernet_shufflenetv2_36e.py --prefix bdd100k/centernet/centernet_shufflenetv2_36e --work-dir ${WORK_DIR}
sleep 10m

# ttfnet
python tools/train.py configs/models/det2d/ttfnet/ttfnet_shufflenetv2_36e.py --prefix bdd100k/ttfnet/ttfnet_shufflenetv2_36e --work-dir ${WORK_DIR}
sleep 10m


