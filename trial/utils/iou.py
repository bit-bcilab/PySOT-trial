

import numpy as np
import torch

import math


epsilon = 1e-7


def box_iou(box1, box2):
    num = box2.shape[-2]
    if len(box1.shape) == 1:
        box1 = np.tile(box1, (num, 1))

    box1_area = np.prod((box1[:, 2:] - box1[:, :2]), axis=-1)
    box2_area = np.prod((box2[..., 2:] - box2[..., :2]), axis=-1)

    left_top = np.maximum(box1[:, :2], box2[..., :2])
    right_bottom = np.minimum(box1[:, 2:], box2[..., 2:])
    inter = np.maximum(right_bottom - left_top, 0.)
    inter_are = np.prod(inter, axis=-1)

    iou = inter_are / (box1_area + box2_area - inter_are)
    return iou


def process_box(boxes, manner='corner'):
    if manner == 'center':
        # 变成左上角坐标、右下角坐标
        boxes = torch.cat([boxes[..., :2] - boxes[..., 2:] * 0.5, boxes[..., :2] + boxes[..., 2:] * 0.5], dim=-1)
    boxes = torch.cat([torch.min(boxes[..., :2], boxes[..., 2:]), torch.max(boxes[..., :2], boxes[..., 2:])], dim=-1)
    return boxes


def bbox_iou(boxes1, boxes2):
    # 两个矩形的面积
    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    # 相交矩形的左上角坐标、右下角坐标，shape 都是 (8, 13, 13, 3, 2)
    left_up = torch.max(boxes1[..., :2], boxes2[..., :2])
    right_down = torch.min(boxes1[..., 2:], boxes2[..., 2:])

    # 相交矩形的面积inter_area。iou
    inter_section = torch.max(right_down - left_up, torch.zeros_like(left_up))
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    iou = inter_area / (union_area + epsilon)
    return iou, union_area


def bbox_giou(boxes1, boxes2, iou=None, union_area=None):
    if iou is None:
        iou, union_area = bbox_iou(boxes1, boxes2)

    c_left_up = torch.min(boxes1[..., :2], boxes2[..., :2])
    c_right_down = torch.max(boxes1[..., 2:], boxes2[..., 2:])
    c_inter_section = torch.max(c_right_down - c_left_up, torch.zeros_like(c_right_down))
    c_inter_area = c_inter_section[..., 0] * c_inter_section[..., 1]

    giou = iou - (c_inter_area - union_area) / (c_inter_area + epsilon)
    return giou, iou


def bbox_diou(boxes1, boxes2, iou=None, union_area=None):
    """
    diou = iou - p2/c2
    """
    # 变成左上角坐标、右下角坐标
    if iou is None:
        iou, _ = bbox_iou(boxes1, boxes2)

    # 包围矩形的左上角坐标、右下角坐标，shape 都是 (8, 13, 13, 3, 2)
    enclose_left_up = torch.min(boxes1[..., :2], boxes2[..., :2])
    enclose_right_down = torch.max(boxes1[..., 2:], boxes2[..., 2:])

    # 包围矩形的对角线的平方
    enclose_wh = enclose_right_down - enclose_left_up
    enclose_c2 = torch.pow(enclose_wh[..., 0], 2) + torch.pow(enclose_wh[..., 1], 2)

    # 两矩形中心点距离的平方
    p2 = torch.pow(boxes1[..., 0] - boxes2[..., 0], 2) + torch.pow(boxes1[..., 1] - boxes2[..., 1], 2)

    diou = iou - 1.0 * p2 / (enclose_c2 + epsilon)
    return diou, iou


def bbox_ciou(boxes1, boxes2, iou=None, union_area=None):
    """
    ciou = iou - p2/c2 - av = diou - av
    """
    diou, iou = bbox_diou(boxes1, boxes2, iou=iou, union_area=union_area)
    # 增加av。加上除0保护防止nan。
    atan1 = torch.atan(boxes1[..., 2] / (boxes1[..., 3] + epsilon))
    atan2 = torch.atan(boxes2[..., 2] / (boxes2[..., 3] + epsilon))
    v = 4.0 * torch.pow(atan1 - atan2, 2) / (math.pi ** 2)
    a = v / (1. - iou + v + epsilon)

    ciou = diou - 1.0 * a * v
    return ciou, iou
