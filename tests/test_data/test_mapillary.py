import cv2
import os
import numpy as np
from mmseg.datasets import MapillaryDataset

if __name__ == '__main__':
    data_root = '/nfs/volume-807-2/darrenwang/mapillary'
    img_dir = 'training/images'
    ann_dir = 'training/labels'
    pipeline = [dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations'),
                dict(type='Resize', min_size=1024, max_size=2048, ratio_range=[0.5, 2.0])]
    dataset = MapillaryDataset(pipeline=pipeline, data_root=data_root, img_dir=img_dir, ann_dir=ann_dir)
    palette = np.array(dataset.PALETTE)
    opacity = 0.5
    for step, data in enumerate(dataset):
        print(step, data['img'].shape, data['gt_semantic_seg'].shape)
        img = data['img']
        seg = data['gt_semantic_seg']
        color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
        for label, color in enumerate(palette):
            color_seg[seg == label, :] = color
        # convert to BGR
        color_seg = color_seg[..., ::-1]

        img = img * (1 - opacity) + color_seg * opacity
        img = img.astype(np.uint8)

        out_file = os.path.join('vis', str(step) + '.jpg')
        cv2.imwrite(out_file, img)
