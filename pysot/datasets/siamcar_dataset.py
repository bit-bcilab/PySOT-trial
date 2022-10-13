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

logger = logging.getLogger("global")

# setting opencv
pyv = sys.version[0]
if pyv[0] == '3':
    cv2.ocl.setUseOpenCL(False)


class SiamCARDataset(TrkDataset):
    def __init__(self, cfg):
        super(SiamCARDataset, self).__init__(cfg)

    def __getitem__(self, index):
        # read images with ground truth
        template, search, bbox, neg = self.read(index)
        cls = np.zeros((self.cfg.TRAIN.OUTPUT_SIZE, self.cfg.TRAIN.OUTPUT_SIZE), dtype=np.int64)
        return {
                'template': template,
                'search': search,
                'label_cls': cls,
                'bbox': np.array(bbox)
                }
