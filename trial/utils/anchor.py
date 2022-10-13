

from __future__ import division

from trial.utils.bbox import corner2center

import numpy as np
import math


def generate_base_anchors(stride, ratios, scales):
    anchor_num = len(scales) * len(ratios)
    anchors = np.zeros((anchor_num, 4), dtype=np.float32)
    size = stride * 4.
    count = 0
    for r in ratios:
        ws = size * math.sqrt(1. / r)
        hs = ws * r

        for s in scales:
            w = int(ws * s)
            h = int(hs * s)
            anchors[count][:] = [-w * 0.5, -h * 0.5, w * 0.5, h * 0.5][:]
            count += 1
    return anchors


def generate_all_anchors(base_anchors, search_size, score_size, central=False, reshape=False):
    anchor_num = base_anchors.shape[0]
    stride = search_size[0] / score_size[0]
    shift_x = (np.arange(0, score_size[1], dtype=np.float32) + 0.5) * stride
    shift_y = (np.arange(0, score_size[0], dtype=np.float32) + 0.5) * stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    grid = np.concatenate((shift_x[..., None, None], shift_y[..., None, None]), axis=-1)
    all_anchors = np.tile(grid, (anchor_num, 2))
    all_anchors = base_anchors[None, None, ...] + all_anchors
    if central:
        center = np.array(search_size, dtype=np.float32).reshape((1, -1)) / 2.
        all_anchors = all_anchors - center

    anchors_xywh = corner2center(all_anchors)
    if reshape:
        all_anchors = all_anchors.reshape((-1, 4))
        anchors_xywh = anchors_xywh.reshape((-1, 4))
    return all_anchors, anchors_xywh


def init_anchor(anchor):
    anchor_num = len(anchor)
    anchors = np.zeros((anchor_num, 4), dtype=np.float32)
    for i in range(anchor_num):
        w, h = anchor[i]
        anchors[i][:] = [-w * 0.5, -h * 0.5, w * 0.5, h * 0.5][:]
    return anchors
