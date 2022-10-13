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
from pysot.utils.anchor import Anchors
from pysot.encoders.anchor_target import anchor_target

logger = logging.getLogger("global")

# setting opencv
pyv = sys.version[0]
if pyv[0] == '3':
    cv2.ocl.setUseOpenCL(False)


class SiamAttDataset(TrkDataset):
    def __init__(self, cfg):
        super(SiamAttDataset, self).__init__(cfg)
        desired_size = (cfg.TRAIN.SEARCH_SIZE - cfg.TRAIN.EXEMPLAR_SIZE) / cfg.ANCHOR.STRIDE + 1 + cfg.TRAIN.BASE_SIZE
        if desired_size != cfg.TRAIN.OUTPUT_SIZE:
            raise Exception('size not match!')
        self.anchors = Anchors(cfg.ANCHOR.STRIDE, cfg.ANCHOR.RATIOS, cfg.ANCHOR.SCALES)
        self.anchors.generate_all_anchors(im_c=cfg.TRAIN.SEARCH_SIZE//2, size=cfg.TRAIN.OUTPUT_SIZE)

    def __getitem__(self, index):
        # read images with ground truth
        template, search, bbox, neg = self.read(index)
        # generate labels
        cls, delta, delta_weight, overlap = anchor_target(self.cfg, bbox, self.anchors, self.cfg.TRAIN.OUTPUT_SIZE, neg)
        return {
                'template': template,
                'search': search,
                'label_cls': cls,
                'label_loc': delta,
                'label_loc_weight': delta_weight,
                'bbox': np.array(bbox)
                }
