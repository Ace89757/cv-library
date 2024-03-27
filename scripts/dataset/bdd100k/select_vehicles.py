# -*- coding: utf-8 -*-
# @Author  : ace
# Copyright by Ace, All Rights Reserved.

import os
import cv2
import json
import math
import argparse
import numpy as np

from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--label-dir', help='bdd100k labels dir')
    parser.add_argument('--imgs-dir', help='bdd100k images dir')

    return parser.parse_args()


CLASSES = {
    'car': 1,
    'bus': 2,
    'truck': 3,
    'train': 4, 
    'other vehicle': 5,
    'trailer': 6
}


if __name__ == '__main__':
    args = parse_args()

    error_imgs = [
        '162995f1-0e43ba5e.jpg', '1af55d81-20ae3997.jpg', '1d33c83b-71e1ea1c.jpg', '26e77b38-98ab4cf4.jpg', '273765fb-53173218.jpg', 
        '282f55fa-5fd561d1.jpg', '2e5a3ced-c7603d0d.jpg', '3af9b86e-bde99ec4.jpg', '48814754-fd974cd2.jpg', '4c888797-aa2ee802.jpg', 
        '527e239f-c921493c.jpg', '5ccd45ac-5b127b2d.jpg', '67c7bea2-283d3be6.jpg', '88089880-374b8b28.jpg', '8edb33ee-ec1bc2e0.jpg', 
        '92a75740-fd28035b.jpg', 'a57604f0-18f88cd6.jpg', 'abb74b5e-4e25a4b6.jpg', '0c96e121-8398710b.jpg', '0d9d0baa-e58b75c3.jpg',
        '1e3c65ba-64d02df4.jpg', '2173474d-208fea1a.jpg', '25ca53f2-e85bc019.jpg', '35400756-af0a0563.jpg', '36b03591-64915045.jpg',
        '41cc0108-aec78b26.jpg', '4ec6eb8d-1c5529b8.jpg', '5872965e-1e88b19f.jpg', '68068bd2-8dc9f4bd.jpg']
    
    num_smalls = {'traffic sign': 0, 'traffic light': 0}
    num_mediums = {'traffic sign': 0, 'traffic light': 0}
    num_larges = {'traffic sign': 0, 'traffic light': 0}

    scales = []
    
    finest = 8

    # 获取标注文件
    for label_file in os.listdir(args.label_dir):
        label_name = os.path.splitext(label_file)[0]
        if 'coco' in label_name:
            continue

        img_id = 1
        ann_id = 1

        coco = {
            'categories': [], 
            'annotations': [], 
            'images': []
            }

        # 加载标注结果
        with open(os.path.join(args.label_dir, label_file), 'r') as f:
            anns = json.load(f)

        levels = {0: 0, 1: 0, 2: 0, 3: 0}
        
        for ann in tqdm(anns, ncols=150, total=len(anns), unit='object', desc=label_name):
            if 'labels' not in ann:
                continue

            img_name = ann['name']
            if img_name in error_imgs:
                continue

            labels = ann['labels']
            timestamp = ann['timestamp']
            attributes = ann['attributes']

            img_file = os.path.join(args.imgs_dir, img_name)
            if not os.path.exists(img_file):
                continue

            used = False  # 判断这张图片是否包含有效目标
            for label in labels:
                category = label['category']
                x1, y1, x2, y2 = [float(label['box2d'][s]) for s in ['x1', 'y1', 'x2', 'y2']]
                w = x2 - x1
                h = y2 - y1

                tmp = math.sqrt(w * h)
                if tmp < finest * 2:
                    levels[0] += 1
                elif finest * 2 <= tmp < finest * 4:
                    levels[1] += 1
                elif finest * 4 < tmp < finest * 8:
                    levels[2] += 1
                else:
                    levels[3] += 1

                label_attrs = label['attributes']
                occluded = label_attrs.pop('occluded')
                truncated = label_attrs.pop('truncated')

                if category not in CLASSES:
                    continue
                
                if w * h <= 64:
                    num_smalls[category] += 1
                elif w * h > 64 and w * h < 96 * 96:
                    num_mediums[category] += 1
                else:
                    num_larges[category] += 1
                
                coco_ann = {
                    'bbox': [x1, y1, w, h],
                    'ignore': 0,
                    'area': w * h,
                    'iscrowd': 0,
                    'category_id': CLASSES[category],
                    'id': ann_id,
                    'image_id': img_id,
                    'occluded': occluded,
                    'truncated': truncated,
                    'attributes': label_attrs
                    }
                
                coco['annotations'].append(coco_ann)

                ann_id += 1

                used = True
            
            if used:
                coco['images'].append({
                    'file_name': img_name,
                    'width': 1280,
                    'height': 720,
                    'id': img_id,
                    'timestamp': timestamp,
                    'attributes': attributes
                })

                img_id += 1

        # for cls, cls_id in CLASSES.items():
        #     print(f'{cls}: {cls_id}')
        #     coco['categories'].append({'id': cls_id, 'name': cls})
        
        # print(f'found {img_id - 1} images')
        # with open(os.path.join(args.label_dir, f'{label_name}_{len(CLASSES)}cls_{img_id - 1}_coco.json'), 'w') as w:
        #     json.dump(coco, w)
                
        # scales.sort()
        # print(scales[int(len(scales) / 2)])

        # print(np.array_split(scales[:20], 3))
        # print(scales[:10])
        # print(min(scales))
        # print(max(scales))
        # exit()
        print(levels)






