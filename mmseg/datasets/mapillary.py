# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class MapillaryDataset(CustomDataset):
    """Pascal Mapillary dataset.

    Args:
        split (str): Split txt file for Mapillary
    """

    CLASSES = ['Background', 'Car', 'Truck', 'Bus', 'Other Vehicle', 'MotorCyclist', 'Bicyclist', 'Pedestrian',
               'Sign', 'Traffic Light', 'Pole', 'Construction Cone', 'Bicycle', 'MotorCycle', 'Building',
               'Vegetation', 'Tree Trunk', 'Curb', 'Road', 'Lane Marker', 'Other Ground', 'Walkable', 'SideWalk']

    PALETTE = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
               [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0],
               [192, 0, 0], [64, 128, 0], [192, 128, 0], [64, 0, 128],
               [192, 0, 128], [64, 128, 128], [0, 192, 0], [0, 64, 0],
               [128, 64, 0], [128, 64, 128], [128, 192, 0], [0, 64, 128],
               [32, 178, 170], [244, 35, 232]]

    def __init__(self, **kwargs):
        super(MapillaryDataset, self).__init__(
            img_suffix='.jpg', seg_map_suffix='.png', **kwargs)
        assert osp.exists(self.img_dir)