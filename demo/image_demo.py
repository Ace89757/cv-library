# -*- coding: utf-8 -*-
# @Author  : ace
# Copyright by Ace, All Rights Reserved.

from argparse import ArgumentParser

from mmengine.logging import print_log

from alchemy.apis.alchemy_det2d_inferencer import AlchemyDet2dInferencer


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('inputs', type=str, help='Input image file or folder path.')
    parser.add_argument('model',
                        type=str,
                        help='Config or checkpoint .pth file or the model name and alias defined in metafile. '
                             'The model configuration file will try to read from .pth if the parameter is a .pth weights file.')
    parser.add_argument('--weights', default=None, help='Checkpoint file')
    parser.add_argument('--out-dir', type=str, default=None, help='Output directory of images or prediction results.')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    parser.add_argument('--pred-score-thr', type=float, default=0.1, help='bbox score threshold')
    parser.add_argument('--batch-size', type=int, default=1, help='Inference batch size.')
    parser.add_argument('--show', action='store_true', help='Display the image in a popup window.')
    parser.add_argument('--show-cls-name', action='store_true', help='show class name')
    parser.add_argument('--bbox-line-width', type=int, default=1)
    parser.add_argument('--palette', default='none', choices=['coco', 'voc', 'citys', 'random', 'none'], help='Color palette used for visualization')

    call_args = vars(parser.parse_args())

    if call_args['model'].endswith('.pth'):
        print_log('The model is a weight file, automatically assign the model to --weights')
        call_args['weights'] = call_args['model']
        call_args['model'] = None

    init_kws = ['model', 'weights', 'device', 'palette', 'out_dir', "show_cls_name", "bbox_line_width"]
    init_args = {}
    for init_kw in init_kws:
        init_args[init_kw] = call_args.pop(init_kw)

    return init_args, call_args


def main():
    init_args, call_args = parse_args()
    inferencer = AlchemyDet2dInferencer(**init_args)

    inferencer(**call_args)

    print_log(f'results have been saved at {inferencer.out_dir}')


if __name__ == '__main__':
    main()
