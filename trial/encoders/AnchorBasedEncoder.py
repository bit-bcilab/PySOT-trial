

from __future__ import division

import numpy as np

from trial.utils.bbox import corner2center
from trial.utils.rand import select

import random


"""
 lrtb形式的anchor用于计算与bbox(lrtb)之间的IOU，为positive与negative samples的assign提供判据
 xywh形式的anchor用于计算与bbox(xywh)之间的delta, 作为regression分支的target
"""


def box_iou(box1, box2):
    box1 = box1[:, None, None, None]

    box1_area = np.prod((box1[2:, :] - box1[:2, :]), axis=0)
    box2_area = np.prod((box2[2:, ...] - box2[:2, ...]), axis=0)

    left_top = np.maximum(box1[:2, :], box2[:2, ...])
    right_bottom = np.minimum(box1[2:, :], box2[2:, ...])
    inter = np.maximum(right_bottom - left_top, 0.)
    inter_are = np.prod(inter, axis=0)

    iou = inter_are / (box1_area + box2_area - inter_are)
    return iou


def multi_pos_ignore_multi_neg_encoder(true_box,
                                       positive,
                                       search_size,
                                       score_size,
                                       anchors,
                                       high_iou_threshold,
                                       pos_num,
                                       easy_pos_num,
                                       hard_pos_num,
                                       low_iou_threshold,
                                       easy_neg_num,
                                       mid_neg_num,
                                       hard_neg_num,
                                       k,
                                       radium):
    """
    Faster RCNN 中 训练 RPN 网络的编码方式
    大于高 IOU 阈值的 anchors 中抽样得到正样本
    小于低 IOU 阈值的 anchors 中抽样得到负样本
    其他 anchors 全部忽略

    :param true_boxes:
    :param positive:
    :param search_size:
    :param score_size:
    :param anchors:
    :param high_iou_threshold:
    :param pos_num:
    :param easy_pos_num:
    :param hard_pos_num:
    :param low_iou_threshold:
    :param easy_neg_num:
    :param mid_neg_num:
    :param hard_neg_num:
    :param k:
    :param radium:
    :return:
    """
    all_anchors = anchors[0]
    anchors_xywh = anchors[1]
    box_xywh = corner2center(true_box)
    box_xywh_ = box_xywh[:, None, None, None]
    anchor_num = anchors_xywh.shape[-2]
    stride = search_size[0] / score_size[0]

    cls_label = -1. * np.ones((anchor_num, score_size[0], score_size[1]), dtype=np.float32)
    loc_label = np.zeros((4, anchor_num, score_size[0], score_size[1]), dtype=np.float32)

    loc_label[0, ...] = (box_xywh_[0, ...] - anchors_xywh[0, ...]) / anchors_xywh[2, ...]
    loc_label[1, ...] = (box_xywh_[1, ...] - anchors_xywh[1, ...]) / anchors_xywh[3, ...]
    loc_label[2, ...] = np.log(np.maximum(box_xywh_[2, ...] / anchors_xywh[..., 2], 1e-6))
    loc_label[3, ...] = np.log(np.maximum(box_xywh_[3, ...] / anchors_xywh[..., 3], 1e-6))

    if positive:
        overlap = box_iou(true_box, all_anchors)
        neg = np.where(overlap < low_iou_threshold)
        pos = np.where(overlap > high_iou_threshold)
        pos_num_ = pos[0].shape[0]
        if pos_num_ > pos_num:
            pos_overlap = overlap[pos]
            easy_pos_index = np.argpartition(pos_overlap, -easy_pos_num, axis=0)[-easy_pos_num:]
            easy_pos = tuple(p[easy_pos_index] for p in pos)

            hard_pos_index = np.argpartition(pos_overlap, hard_pos_num, axis=0)[hard_pos_num:]
            hard_pos = tuple(p[hard_pos_index] for p in pos)
            pos_ = tuple(np.concatenate([easy_pos[c], hard_pos[c]]) for c in range(3))
        else:
            if pos_num_ == 0:
                max_iou = np.max(overlap)
                pos_ = np.where(overlap == max_iou)
            else:
                pos_ = pos

        if hard_neg_num > 0:
            neg_overlap = overlap[neg]
            # 从IOU < 0.3的前 neg_num * k 的anchor中随机选择难负样本
            hard_neg_index = np.argpartition(neg_overlap, -int(hard_neg_num * k), axis=0)[
                             -int(hard_neg_num * k):]
            hard_neg = tuple(n[hard_neg_index] for n in neg)
            hard_neg, _, hard_neg_index = select(hard_neg, hard_neg_num)

            # 选取分布目标附近可能存在相似物的潜在anchor作为mid负样本
            mid_neg_ = tuple(np.delete(n, hard_neg_index) for n in neg)
        else:
            mid_neg_ = neg

        if mid_neg_num > 0:
            lt = np.maximum(np.ceil((box_xywh[:2] - radium * box_xywh[2:]) / stride), 0)
            rb = np.minimum(np.floor((box_xywh[:2] + radium * box_xywh[2:]) / stride), np.array(score_size))
            mid_neg_list = np.stack((mid_neg_[0], mid_neg_[1]), axis=-1)
            mid_neg_index = np.where((mid_neg_list[1, :] > lt[0]) & (mid_neg_list[0, :] > lt[1]) &
                                     (mid_neg_list[1, :] < rb[0]) & (mid_neg_list[0, :] < rb[1]))

            mid_neg = tuple(n[mid_neg_index] for n in mid_neg_)
            mid_neg, _, mid_neg_index = select(mid_neg, mid_neg_num)

            # 从剩余位置（大部分是边界）随机选择易负样本
            easy_neg = tuple(np.delete(n, mid_neg_index) for n in mid_neg_)
        else:
            if hard_neg_num > 0:
                easy_neg = mid_neg_
            else:
                easy_neg = neg

        easy_neg, _, _ = select(easy_neg, easy_neg_num)

        if hard_neg_num > 0 and mid_neg_num > 0:
            neg_ = tuple(np.concatenate([easy_neg[c], mid_neg[c], hard_neg[c]]) for c in range(3))
        if hard_neg_num > 0 and mid_neg_num == 0:
            neg_ = tuple(np.concatenate([easy_neg[c], hard_neg[c]]) for c in range(3))
        if hard_neg_num == 0 and mid_neg_num > 0:
            neg_ = tuple(np.concatenate([easy_neg[c], mid_neg[c]]) for c in range(3))
        if hard_neg_num == 0 and mid_neg_num == 0:
            neg_ = easy_neg

        cls_label[neg_] = 0.
        cls_label[pos_] = 1.
    else:
        box = true_box / stride
        left = max(0, int(np.round(box[0] - 2)))
        up = max(0, int(np.round(box[1] - 2)))
        right = min(score_size[1], int(np.round(box[2] + 2)))
        down = min(score_size[0], int(np.round(box[3] + 2)))

        cls_label[:, up:down, left:right] = 0.
        hard_neg, _, _ = select(np.where(cls_label == 0.), hard_neg_num)
        cls_label = -1.
        cls_label[hard_neg] = 0.

        easy_neg, _, _ = select(np.where(cls_label == -1.), easy_neg_num)
        cls_label[easy_neg] = 0.

    return cls_label, loc_label


# def anchor_based_mix_encoder(true_box,
#                              mix_boxes,
#                              positive,
#                              search_size,
#                              score_size,
#                              anchors,
#                              high_iou_threshold,
#                              pos_num,
#                              easy_pos_num,
#                              hard_pos_num,
#                              low_iou_threshold,
#                              easy_neg_num,
#                              mid_neg_num,
#                              hard_neg_num,
#                              k,
#                              radium):
#     all_anchors = anchors[0]
#     anchors_xywh = anchors[1]
#     box_xywh = corner2center(true_box)
#     box_xywh_ = box_xywh[:, None, None, None]
#     anchor_num = anchors_xywh.shape[1]
#     stride = search_size[0] / score_size[0]
#
#     cls_label = -1. * np.ones((anchor_num, score_size[0], score_size[1]), dtype=np.float32)
#     loc_label = np.zeros((4, anchor_num, score_size[0], score_size[1]), dtype=np.float32)
#
#     loc_label[:2, ...] = (box_xywh_[:2, ...] - anchors_xywh[:2, ...]) / anchors_xywh[2:, ...]
#     loc_label[2:, ...] = np.log(np.maximum((box_xywh_[2:, ...] + 1e-6) / anchors_xywh[2:, ...], 1e-6))
#
#     overlap = box_iou(true_box, all_anchors)
#     neg = np.where(overlap < low_iou_threshold)
#     pos = np.where(overlap > high_iou_threshold)
#
#     mix_neg_num = 0
#     if mix_boxes is not None:
#         mix_num = mix_boxes.shape[0]
#         for i in range(mix_num):
#             max_num = random.randint(10, 14)
#             mix_box = mix_boxes[i, :]
#             mix_overlap = box_iou(mix_box, all_anchors)
#             mix = np.where(mix_overlap > 0.4)
#             mix_neg_num_ = mix[0].shape[0]
#
#             if mix_neg_num_ > max_num:
#                 pos_overlap_ = mix_overlap[mix]
#                 pos_index = np.argpartition(pos_overlap_, -max_num, axis=0)[-max_num:]
#                 mix = tuple(m[pos_index] for m in mix)
#             else:
#                 if mix_neg_num_ == 0:
#                     max_iou = np.max(mix_overlap)
#                     if max_iou > 0.2:
#                         mix = np.where(mix_overlap == max_iou)
#
#             cls_label[mix] = 0.
#
#             mix_neg_num += mix[0].shape[0]
#
#     pos_num_ = pos[0].shape[0]
#     if pos_num_ > pos_num:
#         pos_overlap = overlap[pos]
#         easy_pos_index = np.argpartition(pos_overlap, -easy_pos_num, axis=0)[-easy_pos_num:]
#         easy_pos = tuple(p[easy_pos_index] for p in pos)
#
#         hard_pos_index = np.argpartition(pos_overlap, hard_pos_num, axis=0)[hard_pos_num:]
#         hard_pos = tuple(p[hard_pos_index] for p in pos)
#         pos_ = tuple(np.concatenate([easy_pos[c], hard_pos[c]]) for c in range(3))
#     else:
#         if pos_num_ == 0:
#             max_iou = np.max(overlap)
#             pos_ = np.where(overlap == max_iou)
#         else:
#             pos_ = pos
#
#     if positive:
#         if hard_neg_num > 0:
#             neg_overlap = overlap[neg]
#             # 从IOU < 0.3的前 neg_num * k 的anchor中随机选择难负样本
#             hard_neg_index = np.argpartition(neg_overlap, -int(hard_neg_num * k), axis=0)[-int(hard_neg_num * k):]
#             hard_neg = tuple(n[hard_neg_index] for n in neg)
#             hard_neg, _, hard_neg_index = select(hard_neg, hard_neg_num)
#
#             # 选取分布目标附近可能存在相似物的潜在anchor作为mid负样本
#             mid_neg_ = tuple(np.delete(n, hard_neg_index) for n in neg)
#         else:
#             mid_neg_ = neg
#
#         if mid_neg_num > 0:
#             lt = np.maximum(np.ceil((box_xywh[:2] - radium * box_xywh[2:]) / stride), 0)
#             rb = np.minimum(np.floor((box_xywh[:2] + radium * box_xywh[2:]) / stride), np.array(score_size))
#             mid_neg_list = np.stack((mid_neg_[1], mid_neg_[2]), axis=-1)
#             mid_neg_index = np.where((mid_neg_list[:, 1] > lt[0]) & (mid_neg_list[:, 0] > lt[1]) &
#                                      (mid_neg_list[:, 1] < rb[0]) & (mid_neg_list[:, 0] < rb[1]))
#
#             mid_neg = tuple(n[mid_neg_index] for n in mid_neg_)
#             mid_neg, _, mid_neg_index = select(mid_neg, mid_neg_num)
#
#             # 从剩余位置（大部分是边界）随机选择易负样本
#             easy_neg = tuple(np.delete(n, mid_neg_index) for n in mid_neg_)
#         else:
#             if hard_neg_num > 0:
#                 easy_neg = mid_neg_
#             else:
#                 easy_neg = neg
#
#         easy_neg, _, _ = select(easy_neg, easy_neg_num)
#
#         if hard_neg_num > 0 and mid_neg_num > 0:
#             neg_ = tuple(np.concatenate([easy_neg[c], mid_neg[c], hard_neg[c]]) for c in range(3))
#         if hard_neg_num > 0 and mid_neg_num == 0:
#             neg_ = tuple(np.concatenate([easy_neg[c], hard_neg[c]]) for c in range(3))
#         if hard_neg_num == 0 and mid_neg_num > 0:
#             neg_ = tuple(np.concatenate([easy_neg[c], mid_neg[c]]) for c in range(3))
#         if hard_neg_num == 0 and mid_neg_num == 0:
#             neg_ = easy_neg
#
#         neg_num = neg_[0].shape[0]
#         neg_num_ = neg_num - mix_neg_num
#         if neg_num_ < 8:
#             neg_num_ = random.randint(8, 12)
#         neg_, _, _ = select(neg_, neg_num_)
#
#         cls_label[neg_] = 0.
#         cls_label[pos_] = 1.
#     else:
#         neg_num_ = hard_neg_num + mid_neg_num + easy_neg_num - pos_[0].shape[0] - mix_neg_num
#         if neg_num_ < 8:
#             neg_num_ = random.randint(8, 12)
#
#         cls_label[pos_] = 0.
#
#         easy_neg, _, _ = select(np.where(cls_label == -1.), neg_num_)
#         cls_label[easy_neg] = 0.
#     return cls_label, loc_label


def anchor_based_mix_encoder(true_box,
                             mix_boxes,
                             positive,
                             search_size,
                             score_size,
                             anchors,
                             high_iou_threshold,
                             pos_num,
                             low_iou_threshold,
                             neg_num):
    all_anchors = anchors[0]
    anchors_xywh = anchors[1]
    box_xywh = corner2center(true_box)
    box_xywh_ = box_xywh[:, None, None, None]
    anchor_num = anchors_xywh.shape[1]

    cls_label = -1. * np.ones((anchor_num, score_size[0], score_size[1]), dtype=np.float32)
    loc_target = np.zeros((4, anchor_num, score_size[0], score_size[1]), dtype=np.float32)

    loc_target[:2, ...] = (box_xywh_[:2, ...] - anchors_xywh[:2, ...]) / anchors_xywh[2:, ...]
    loc_target[2:, ...] = np.log(np.maximum(box_xywh_[2:, ...] / anchors_xywh[2:, ...], 1e-6))

    overlap = box_iou(true_box, all_anchors)
    neg = np.where(overlap < low_iou_threshold)
    pos = np.where(overlap > high_iou_threshold)

    mix_neg_num = 0
    if mix_boxes is not None:
        mix_num = mix_boxes.shape[0]
        for i in range(mix_num):
            max_num = random.randint(7, 10)
            mix_box = mix_boxes[i, :]
            mix_overlap = box_iou(mix_box, all_anchors)
            mix = np.where(mix_overlap > 0.4)
            mix_neg_num_ = mix[0].shape[0]

            if mix_neg_num_ > max_num:
                pos_overlap_ = mix_overlap[mix]
                pos_index = np.argpartition(pos_overlap_, -max_num, axis=0)[-max_num:]
                mix = tuple(m[pos_index] for m in mix)
            else:
                if mix_neg_num_ == 0:
                    max_iou = np.max(mix_overlap)
                    if max_iou > 0.2:
                        mix = np.where(mix_overlap == max_iou)

            cls_label[mix] = 0.

            mix_neg_num += mix[0].shape[0]

    pos_num_ = pos[0].shape[0]
    if pos_num_ > pos_num:
        pos_overlap = overlap[pos]
        pos_index = np.argpartition(pos_overlap, -pos_num, axis=0)[-pos_num:]
        pos_ = tuple(p[pos_index] for p in pos)
    else:
        if pos_num_ == 0:
            max_iou = np.max(overlap)
            pos_ = np.where(overlap == max_iou)
        else:
            pos_ = pos

    if positive:
        neg_num_ = neg_num - mix_neg_num
        if neg_num_ < 8:
            neg_num_ = random.randint(8, 12)
        neg_, _, _ = select(neg, neg_num_)

        cls_label[neg_] = 0.
        cls_label[pos_] = 1.
    else:
        neg_num_ = neg_num - pos_[0].shape[0] - mix_neg_num
        if neg_num_ < 8:
            neg_num_ = random.randint(8, 10)

        cls_label[pos_] = 0.

        neg, _, _ = select(np.where(cls_label == -1.), neg_num_)
        cls_label[neg] = 0.
    return cls_label, loc_target


def anchor_based_self_encoder(true_box,
                              mix_boxes,
                              positive,
                              search_size,
                              score_size,
                              anchors,
                              high_iou_threshold,
                              pos_num,
                              low_iou_threshold):
    all_anchors = anchors[0]
    anchors_xywh = anchors[1]
    box_xywh = corner2center(true_box)
    box_xywh_ = box_xywh[:, None, None, None]
    anchor_num = anchors_xywh.shape[1]

    cls_label = -1. * np.ones((anchor_num, score_size[0], score_size[1]), dtype=np.float32)
    loc_target = np.zeros((4, anchor_num, score_size[0], score_size[1]), dtype=np.float32)

    loc_target[:2, ...] = (box_xywh_[:2, ...] - anchors_xywh[:2, ...]) / anchors_xywh[2:, ...]
    loc_target[2:, ...] = np.log(np.maximum((box_xywh_[2:, ...] + 1e-6) / anchors_xywh[2:, ...], 1e-6))

    if positive:
        overlap = box_iou(true_box, all_anchors)
        pos = np.where(overlap > high_iou_threshold)
        pos_mask = np.where(overlap > low_iou_threshold)

        pos_num_ = pos[0].shape[0]
        if pos_num_ > pos_num:
            pos_overlap = overlap[pos]
            pos_index = np.argpartition(pos_overlap, -pos_num, axis=0)[-pos_num:]
            pos_ = tuple(p[pos_index] for p in pos)
        else:
            if pos_num_ == 0:
                max_iou = np.max(overlap)
                pos_ = np.where(overlap == max_iou)
            else:
                pos_ = pos
        cls_label[pos_mask] = 0.
        cls_label[pos_] = 1.
    return cls_label, loc_target
