

import numpy as np

from trial.utils.rand import select
from trial.utils.bbox import box2roi, corner2center

import random

"""
Assign positive and negative samples based on rectangle
Regression target is the distance between the point on points and groundtruth bounding box (LTRB format) 

Refer to anchor-free detector FCOS
"""


def find_closest(a, b):
    if a < b[0]:
        return 0
    if a > b[-1]:
        return b.size - 1
    c = np.abs(a - b)
    pos = np.argmin(c)
    return pos


def boundary(bbox_, xs, ys, rate):
    bbox = box2roi(bbox_, rate, rate, boundary=None)
    x1 = find_closest(bbox[0], xs)
    y1 = find_closest(bbox[1], ys)
    x2 = find_closest(bbox[2], xs)
    y2 = find_closest(bbox[3], ys)
    if x2 < x1:
        x2 = x1 + 1
    if y2 < y1:
        y2 = y1 + 1
    return x1, y1, x2, y2


def pos_assign(mask, bbox, xs, ys, rate=0.5, max_num=-1):
    x1, y1, x2, y2 = boundary(bbox, xs, ys, rate)
    mask[y1: y2 + 1, x1: x2 + 1] = 1.

    pos = np.where(mask == 1.)
    pos_num = pos[0].shape[0]

    if max_num > 0:
        # 数量超过上限时，随机挑选
        if pos_num > max_num:
            pos, _, _ = select(pos, max_num)
            pos_num = max_num
        elif pos_num == 0:
            pos = ((y1 + y2) // 2, (x1 + x2) // 2)
            pos_num = 1
    return pos, pos_num


def neg_assign(mask, bbox, xs, ys, rate=1.0, max_num=-1):
    x1, y1, x2, y2 = boundary(bbox, xs, ys, rate)
    mask[y1: y2 + 1, x1: x2 + 1] = 1.

    neg = np.where(mask == 0.)

    neg_num = neg[0].shape[0]

    # 数量超过上限时，随机挑选
    if max_num > 0 and neg_num > max_num:
        neg, _, _ = select(neg, max_num)
        neg_num = max_num
    return neg, neg_num


def rectangle_mask(bbox, points, rate=1., positive=True):
    mask = np.zeros(points.shape[1:], dtype=np.float32)
    if positive:
        xs = points[0, 0, :]
        ys = points[1, :, 0]
        x1, y1, x2, y2 = boundary(bbox, xs, ys, rate)
        mask[y1: y2 + 1, x1: x2 + 1] = 1.
    return mask


def rectangle_encoder(true_box,
                      positive,
                      search_size,
                      score_size,
                      points,
                      pos_rate=.5,
                      neg_rate=1.,
                      pos_num=-1,
                      neg_num=-1,
                      reg_type='ltrb'):
    true_box_ = true_box[:, None, None]
    if 'ltrb' in reg_type:
        loc_target = np.zeros((4, score_size[0], score_size[1]), dtype=np.float32)
        loc_target[:2, ...] = points - true_box_[:2]
        loc_target[2:, ...] = true_box_[2:] - points
    elif 'xywh' in reg_type:
        loc_target = np.zeros((4, score_size[0], score_size[1]), dtype=np.float32)
        bbox_center = corner2center(true_box)[:, None, None]
        loc_target[:2, ...] = points - bbox_center[:2]
        loc_target[2:, ...] = bbox_center[2:]
    elif 'center' in reg_type:
        loc_target = corner2center(true_box)[:, None, None]

    if 'norm' in reg_type:
        loc_target = loc_target / np.concatenate([search_size, search_size]).reshape((-1, 1, 1))

    cls_label = -1. * np.ones((score_size[0], score_size[1]), dtype=np.float32)
    xs = points[0, 0, :]
    ys = points[1, :, 0]
    temp_label = np.zeros((score_size[0], score_size[1]), dtype=np.float32)

    if positive:
        max_neg_num = random.randint(neg_num - 5, neg_num + 5) if neg_num > 0 else -1
        max_pos_num = random.randint(pos_num - 2, pos_num + 2) if pos_num > 0 else -1
        neg, neg_num_ = neg_assign(temp_label.copy(), true_box, xs, ys, neg_rate, max_num=max_neg_num)
        pos, pos_num_ = pos_assign(temp_label.copy(), true_box, xs, ys, pos_rate, max_num=max_pos_num)

        cls_label[neg] = 0.
        cls_label[pos] = 1.
    else:
        if neg_num > 0:
            # 负样本对时，在原目标位置选取大量负样本点，迫使网络进行辨别，而不是记忆
            max_pos_num = random.randint(int(1.25 * pos_num) - 4, int(1.25 * pos_num) + 4) if pos_num > 0 else -1
            pos, pos_num_ = pos_assign(temp_label.copy(), true_box, xs, ys, pos_rate, max_num=max_pos_num)
            neg, neg_num_ = neg_assign(temp_label.copy(), true_box, xs, ys, neg_rate, max_num=int(pos_num_ // 2))

            cls_label[neg] = 0.
            cls_label[pos] = 0.
        else:
            cls_label = np.zeros((score_size[0], score_size[1]), dtype=np.float32)

    return cls_label, loc_target


def rectangle_mix_encoder(true_box,
                          mix_boxes,
                          positive,
                          search_size,
                          score_size,
                          points,
                          pos_rate=.5,
                          neg_rate=1.,
                          pos_num=-1,
                          neg_num=-1,
                          reg_type='ltrb'):
    true_box_ = true_box[:, None, None]
    if 'ltrb' in reg_type:
        loc_target = np.zeros((4, score_size[0], score_size[1]), dtype=np.float32)
        loc_target[:2, ...] = points - true_box_[:2]
        loc_target[2:, ...] = true_box_[2:] - points
    elif 'xywh' in reg_type:
        loc_target = np.zeros((4, score_size[0], score_size[1]), dtype=np.float32)
        bbox_center = corner2center(true_box)[:, None, None]
        loc_target[:2, ...] = points - bbox_center[:2]
        loc_target[2:, ...] = bbox_center[2:]
    elif 'center' in reg_type:
        loc_target = corner2center(true_box)[:, None, None]

    if 'norm' in reg_type:
        loc_target = loc_target / np.concatenate([search_size, search_size]).reshape((-1, 1, 1))

    cls_label = -1. * np.ones((score_size[0], score_size[1]), dtype=np.float32)
    xs = points[0, 0, :]
    ys = points[1, :, 0]
    temp_label = np.zeros((score_size[0], score_size[1]), dtype=np.float32)

    mix_neg_num = 0
    if mix_boxes is not None and not (not positive and (neg_num < 0)):
        mix_num = mix_boxes.shape[0]
        for i in range(mix_num):
            mix, mix_neg_num_ = pos_assign(temp_label.copy(), mix_boxes[i], xs, ys, pos_rate, max_num=random.randint(10, 14))
            cls_label[mix] = 0.
            mix_neg_num += mix_neg_num_

    if positive:
        if neg_num > 0:
            if mix_neg_num > 0:
                if mix_neg_num > 20:
                    max_neg_num = random.randint(10, 14)
                else:
                    max_neg_num = (neg_num - mix_neg_num) // 2
            else:
                max_neg_num = random.randint(neg_num - 5, neg_num + 5)
        else:
            max_neg_num = -1
        max_pos_num = random.randint(pos_num - 2, pos_num + 2) if pos_num > 0 else -1
        neg, neg_num_ = neg_assign(temp_label.copy(), true_box, xs, ys, neg_rate, max_num=max_neg_num)
        pos, pos_num_ = pos_assign(temp_label.copy(), true_box, xs, ys, pos_rate, max_num=max_pos_num)

        cls_label[neg] = 0.
        cls_label[pos] = 1.
    else:
        if neg_num > 0:
            # 负样本对时，在原目标位置选取大量负样本点，迫使网络进行辨别，而不是记忆
            max_pos_num = random.randint(int(1.25 * pos_num) - 4, int(1.25 * pos_num) + 4) if pos_num > 0 else -1
            pos, pos_num_ = pos_assign(temp_label.copy(), true_box, xs, ys, pos_rate, max_num=max_pos_num)
            neg, neg_num_ = neg_assign(temp_label.copy(), true_box, xs, ys, neg_rate, max_num=int(neg_num / 2.5))

            cls_label[neg] = 0.
            cls_label[pos] = 0.
        else:
            cls_label = np.zeros((score_size[0], score_size[1]), dtype=np.float32)

    return cls_label, loc_target


def rectangle_self_encoder(true_box,
                           mix_boxes,
                           positive,
                           search_size,
                           score_size,
                           points,
                           pos_rate=.5,
                           neg_rate=1.,
                           pos_num=-1,
                           reg_type='ltrb'):
    true_box_ = true_box[:, None, None]
    if 'ltrb' in reg_type:
        loc_target = np.zeros((4, score_size[0], score_size[1]), dtype=np.float32)
        loc_target[:2, ...] = points - true_box_[:2]
        loc_target[2:, ...] = true_box_[2:] - points
    elif 'xywh' in reg_type:
        loc_target = np.zeros((4, score_size[0], score_size[1]), dtype=np.float32)
        bbox_center = corner2center(true_box)[:, None, None]
        loc_target[:2, ...] = points - bbox_center[:2]
        loc_target[2:, ...] = bbox_center[2:]
    elif 'center' in reg_type:
        loc_target = corner2center(true_box)[:, None, None]

    if 'norm' in reg_type:
        loc_target = loc_target / np.concatenate([search_size, search_size]).reshape((-1, 1, 1))

    cls_label = -1. * np.ones((score_size[0], score_size[1]), dtype=np.float32)
    if positive:
        xs = points[0, 0, :]
        ys = points[1, :, 0]
        temp_label = np.zeros((score_size[0], score_size[1]), dtype=np.float32)

        max_pos_num = random.randint(pos_num - 2, pos_num + 2) if pos_num > 0 else -1
        neg, neg_num_ = pos_assign(temp_label.copy(), true_box, xs, ys, neg_rate, max_num=-1)
        pos, pos_num_ = pos_assign(temp_label.copy(), true_box, xs, ys, pos_rate, max_num=max_pos_num)

        cls_label[neg] = 0.
        cls_label[pos] = 1.

    return cls_label, loc_target
