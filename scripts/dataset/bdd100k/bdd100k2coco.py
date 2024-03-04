# -*- coding: utf-8 -*-
# @Author  : ace
# Copyright by Ace, All Rights Reserved.

import os
import json
import argparse

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
    'rider': 4,
    'bicycle': 5,
    'motorcycle': 6,
    'pedestrian': 7,
    'other person': 8, 
    'traffic sign': 9,
    'traffic light': 10, 
    'train': 11, 
    'other vehicle': 12,
    'trailer': 13
}


if __name__ == '__main__':
    args = parse_args()

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
        
        for ann in tqdm(anns, ncols=150, total=len(anns), unit='object', desc=label_name):
            if 'labels' not in ann:
                continue

            img_name = ann['name']
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

                label_attrs = label['attributes']
                occluded = label_attrs.pop('occluded')
                truncated = label_attrs.pop('truncated')

                if category not in CLASSES:
                    print(img_file)
                    print(category)
                    print(x1, y1, x2, y2)
                    exit()
                
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

        for cls, cls_id in CLASSES.items():
            print(f'{cls}: {cls_id}')
            coco['categories'].append({'id': cls_id, 'name': cls})
        
        print(f'found {img_id - 1} images')
        with open(os.path.join(args.label_dir, f'{label_name}_{len(CLASSES)}cls_{img_id - 1}_coco.json'), 'w') as w:
            json.dump(coco, w)






