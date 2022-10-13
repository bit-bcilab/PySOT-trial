

import numpy as np

from trial.utils.rand import select
from trial.utils.bbox import corner2center

import random

"""
Assign positive and negative samples based on ellipse
Regression target is the distance between the point on points and groundtruth bounding box (LTRB format) 

Refer to SiamBAN ———— Siamese Box Adaptive Network for Visual Tracking
Paper: https://arxiv.org/pdf/2003.06761.pdf
Code: https://github.com/hqucv/siamban
"""


def pos_assign(bbox, points, radium=4., max_num=-1):
    tcx, tcy, tw, th = bbox

    # 中心区域为正
    pos = np.where(((np.square(tcx - points[0, ...]) / (np.square(tw / radium) + 1e-5) +
                     np.square(tcy - points[1, ...]) / (np.square(th / radium) + 1e-5)) <= 1.))
    pos_num = pos[0].shape[0]

    if max_num > 0:
        # 数量超过上限时，随机挑选
        if pos_num > max_num:
            pos, _, _ = select(pos, max_num)
            pos_num = max_num
        # 没有满足条件的点时，选择离中心最近的点
        elif pos_num == 0:
            center_x = points[0, 0, :]
            center_y = points[1, :, 0]
            x_pos = np.argmin(np.abs(center_x - tcx))
            y_pos = np.argmin(np.abs(center_y - tcy))
            pos = (y_pos, x_pos)
            pos_num = 1
    return pos, pos_num


def neg_assign(bbox, points, radium=2., max_num=-1):
    tcx, tcy, tw, th = bbox

    neg = np.where((np.square(tcx - points[0, ...]) / (np.square(tw / radium) + 1e-5) +
                    np.square(tcy - points[1, ...]) / (np.square(th / radium) + 1e-5)) > 1.)
    neg_num = neg[0].shape[0]

    # 数量超过上限时，随机挑选
    if max_num > 0 and neg_num > max_num:
        neg, _, _ = select(neg, max_num)
        neg_num = max_num
    return neg, neg_num


def ellipse_mask(true_box, positive, points, radium=2.):
    mask = np.zeros(points.shape[1:], dtype=np.float32)
    if positive:
        bbox = corner2center(true_box)
        pos, _ = pos_assign(bbox, points, radium=radium, max_num=-1)
        mask[pos] = 1.
    return mask


def ellipse_encoder(true_box,
                    positive,
                    search_size,
                    score_size,
                    points,
                    pos_num,
                    neg_num,
                    pos_radium=4.,
                    neg_radium=2.,
                    reg_type='ltrb'):
    true_box_ = true_box[:, None, None]
    bbox = corner2center(true_box)

    if 'ltrb' in reg_type:
        loc_target = np.zeros((4, score_size[0], score_size[1]), dtype=np.float32)
        loc_target[:2, ...] = points - true_box_[:2]
        loc_target[2:, ...] = true_box_[2:] - points
    elif 'xywh' in reg_type:
        loc_target = np.zeros((4, score_size[0], score_size[1]), dtype=np.float32)
        bbox_center = bbox[:, None, None]
        loc_target[:2, ...] = points - bbox_center[:2]
        loc_target[2:, ...] = bbox_center[2:]
    elif 'center' in reg_type:
        loc_target = bbox[:, None, None]

    if 'norm' in reg_type:
        loc_target = loc_target / np.concatenate([search_size, search_size]).reshape((-1, 1, 1))

    cls_label = -1. * np.ones((score_size[0], score_size[1]), dtype=np.float32)

    if positive:
        max_neg_num = random.randint(neg_num - 5, neg_num + 5) if neg_num > 0 else -1
        max_pos_num = random.randint(pos_num - 2, pos_num + 2) if pos_num > 0 else -1
        neg, neg_num_ = neg_assign(bbox, points, radium=neg_radium, max_num=max_neg_num)
        pos, pos_num_ = pos_assign(bbox, points, radium=pos_radium, max_num=max_pos_num)

        cls_label[neg] = 0.
        cls_label[pos] = 1.
    else:
        if neg_num > 0:
            # 负样本对时，在原目标位置选取大量负样本点，迫使网络进行辨别，而不是记忆
            max_pos_num = random.randint(int(1.25 * pos_num) - 4, int(1.25 * pos_num) + 4) if pos_num > 0 else -1
            pos, pos_num_ = pos_assign(bbox, points, radium=neg_radium, max_num=max_pos_num)
            neg, neg_num_ = neg_assign(bbox, points, radium=neg_radium, max_num=int(pos_num_ // 2))

            cls_label[neg] = 0.
            cls_label[pos] = 0.
        else:
            cls_label = np.zeros((score_size[0], score_size[1]), dtype=np.float32)

    return cls_label, loc_target


def ellipse_mix_encoder(true_box,
                        mix_boxes,
                        positive,
                        search_size,
                        score_size,
                        points,
                        pos_num,
                        neg_num,
                        pos_radium=4.,
                        neg_radium=2.,
                        reg_type='ltrb'):
    true_box_ = true_box[:, None, None]
    bbox = corner2center(true_box)

    if 'ltrb' in reg_type:
        loc_target = np.zeros((4, score_size[0], score_size[1]), dtype=np.float32)
        loc_target[:2, ...] = points - true_box_[:2]
        loc_target[2:, ...] = true_box_[2:] - points
    elif 'xywh' in reg_type:
        loc_target = np.zeros((4, score_size[0], score_size[1]), dtype=np.float32)
        bbox_center = bbox[:, None, None]
        loc_target[:2, ...] = points - bbox_center[:2]
        loc_target[2:, ...] = bbox_center[2:]
    elif 'center' in reg_type:
        loc_target = bbox[:, None, None]

    if 'norm' in reg_type:
        loc_target = loc_target / np.concatenate([search_size, search_size]).reshape((-1, 1, 1))

    cls_label = -1. * np.ones((score_size[0], score_size[1]), dtype=np.float32)

    mix_radium = (pos_radium + neg_radium) / 2.
    mix_neg_num = 0
    if mix_boxes is not None and not (not positive and (neg_num < 0)):
        mix_num = mix_boxes.shape[0]
        mix_xywh = corner2center(mix_boxes)
        for i in range(mix_num):
            mix, mix_neg_num_ = pos_assign(mix_xywh[i], points, radium=mix_radium, max_num=random.randint(9, 12))

            cls_label[mix] = 0.
            mix_neg_num += mix_neg_num_

    if positive:
        if neg_num > 0:
            if mix_neg_num > 0:
                max_neg_num = random.randint(12, 16)
            else:
                max_neg_num = random.randint(neg_num - 5, neg_num + 5)
        else:
            max_neg_num = -1
        max_pos_num = random.randint(pos_num - 2, pos_num + 2) if pos_num > 0 else -1

        neg, neg_num_ = neg_assign(bbox, points, radium=neg_radium, max_num=max_neg_num)
        pos, pos_num_ = pos_assign(bbox, points, radium=pos_radium, max_num=max_pos_num)

        cls_label[neg] = 0.
        cls_label[pos] = 1.
    else:
        # 负样本对时，在原目标位置选取大量负样本点，迫使网络进行辨别，而不是记忆
        if neg_num > 0:
            # 负样本对时，在原目标位置选取大量负样本点，迫使网络进行辨别，而不是记忆
            max_pos_num = random.randint(pos_num - 4, pos_num + 4) if pos_num > 0 else -1
            pos, pos_num_ = pos_assign(bbox, points, radium=neg_radium, max_num=max_pos_num)
            neg, neg_num_ = neg_assign(bbox, points, radium=neg_radium, max_num=int(neg_num / 2.5))

            cls_label[neg] = 0.
            cls_label[pos] = 0.
        else:
            cls_label = np.zeros((score_size[0], score_size[1]), dtype=np.float32)

    return cls_label, loc_target


def ellipse_self_encoder(true_box,
                         mix_boxes,
                         positive,
                         search_size,
                         score_size,
                         points,
                         pos_num,
                         pos_radium=3.33,
                         neg_radium=1.75,
                         reg_type='ltrb'):
    true_box_ = true_box[:, None, None]
    bbox = corner2center(true_box)

    if 'ltrb' in reg_type:
        loc_target = np.zeros((4, score_size[0], score_size[1]), dtype=np.float32)
        loc_target[:2, ...] = points - true_box_[:2]
        loc_target[2:, ...] = true_box_[2:] - points
    elif 'xywh' in reg_type:
        loc_target = np.zeros((4, score_size[0], score_size[1]), dtype=np.float32)
        bbox_center = bbox[:, None, None]
        loc_target[:2, ...] = points - bbox_center[:2]
        loc_target[2:, ...] = bbox_center[2:]
    elif 'center' in reg_type:
        loc_target = bbox[:, None, None]

    if 'norm' in reg_type:
        loc_target = loc_target / np.concatenate([search_size, search_size]).reshape((-1, 1, 1))

    cls_label = -1. * np.ones((score_size[0], score_size[1]), dtype=np.float32)

    if positive:
        pos, pos_num_ = pos_assign(bbox, points, radium=pos_radium, max_num=pos_num)
        neg, neg_num_ = pos_assign(bbox, points, radium=neg_radium, max_num=-1)
        cls_label[neg] = 0.
        cls_label[pos] = 1.
    return cls_label, loc_target
