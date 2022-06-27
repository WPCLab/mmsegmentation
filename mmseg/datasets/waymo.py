# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class WaymoDataset(CustomDataset):
    """Waymo dataset.

    Args:
        split (str): Split txt file for Waymo
    """

    CLASSES = ['EGO_VEHICLE', 'CAR', 'TRUCK', 'BUS', 'OTHER_LARGE_VEHICLE', 'BICYCLE', 'MOTORCYCLE',
               'TRAILER', 'PEDESTRIAN', 'CYCLIST', 'MOTORCYCLIST', 'BIRD', 'GROUND_ANIMAL',
               'CONSTRUCTION_CONE_POLE', 'POLE', 'PEDESTRIAN_OBJECT', 'SIGN', 'TRAFFIC_LIGHT',
               'BUILDING', 'ROAD', 'LANE_MARKER', 'ROAD_MARKER', 'SIDEWALK', 'VEGETATION',
               'SKY', 'GROUND', 'DYNAMIC', 'STATIC']

    PALETTE = [[102, 102, 102], [0, 0, 142], [0, 0, 70], [0, 60, 100], [61, 133, 198], [119, 11, 32], [0, 0, 230],
               [111, 168, 220], [220, 20, 60], [255, 0, 0], [180, 0, 0], [127, 96, 0], [91, 15, 0],
               [230, 145, 56], [153, 153, 153], [234, 153, 153], [246, 178, 107], [250, 170, 30],
               [70, 70, 70], [128, 64, 128], [234, 209, 220], [217, 210, 233], [244, 35, 232], [107, 142, 35],
               [70, 130, 180], [102, 102, 102], [102, 102, 102], [102, 102, 102]]

    def __init__(self, **kwargs):
        super(WaymoDataset, self).__init__(
            img_suffix='.png', seg_map_suffix='.png', **kwargs)
        assert osp.exists(self.img_dir)
