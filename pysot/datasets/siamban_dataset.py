# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import sys

import cv2
import numpy as np

from pysot.datasets.base_dataset import TrkDataset
from pysot.utils.point import Point
from pysot.encoders.point_target import point_target

logger = logging.getLogger("global")

# setting opencv
pyv = sys.version[0]
if pyv[0] == '3':
    cv2.ocl.setUseOpenCL(False)


class SiamBANDataset(TrkDataset):
    def __init__(self, cfg):
        super(SiamBANDataset, self).__init__(cfg)
        desired_size = (cfg.TRAIN.SEARCH_SIZE - cfg.TRAIN.EXEMPLAR_SIZE) / cfg.POINT.STRIDE + 1 + cfg.TRAIN.BASE_SIZE
        if desired_size != cfg.TRAIN.OUTPUT_SIZE:
            raise Exception('size not match!')
        # # create point target
        self.points = Point(cfg.POINT.STRIDE, cfg.TRAIN.OUTPUT_SIZE, cfg.TRAIN.SEARCH_SIZE // 2).points

    def __getitem__(self, index):
        # read images with ground truth
        template, search, bbox, neg = self.read(index)
        # generate labels
        cls, delta = point_target(self.cfg, bbox, self.points, self.cfg.TRAIN.OUTPUT_SIZE, neg)
        return {
                'template': template,
                'search': search,
                'label_cls': cls,
                'label_loc': delta,
                'bbox': np.array(bbox)
                }
