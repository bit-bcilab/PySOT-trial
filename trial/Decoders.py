

import torch


def TransTDecoder(cls, loc, ratio, boundary):
    score = cls.softmax(dim=-1)[..., 1]
    boxes_xywh = loc * ratio
    boxes = torch.cat([boxes_xywh[..., :2] - boxes_xywh[..., 2:] * 0.5,
                       boxes_xywh[..., :2] + boxes_xywh[..., 2:] * 0.5], dim=-1)
    boxes = torch.clamp(boxes, min=0., max=boundary)
    boxes_xywh_ = torch.cat([(boxes[..., :2] + boxes[..., 2:]) * 0.5, boxes[..., 2:] - boxes[..., :2]], dim=-1)
    return score, boxes, boxes_xywh_


def LTRBDecoder(cls, ltrb, points, boundary):
    boxes = torch.zeros_like(ltrb, dtype=torch.float32, device=ltrb.device)
    if cls.shape[1] == 1:
        score = cls.sigmoid()
    elif cls.shape[1] == 2:
        score = cls.softmax(dim=1)[:, 1, ...]
    boxes[:, :2, ...] = points - ltrb[:, :2, ...]
    boxes[:, 2:, ...] = points + ltrb[:, 2:, ...]
    boxes = torch.clamp(boxes, min=0., max=boundary)
    return score, boxes


def AnchorBasedDecoder(cls, loc, anchors, boundary):
    if cls.shape[1] == 1:
        score = cls.sigmoid()
    elif cls.shape[1] == 2:
        score = cls.softmax(dim=1)[:, 1, ...]

    boxes, boxes_xywh = torch.zeros_like(loc, dtype=torch.float32), torch.zeros_like(loc, dtype=torch.float32)
    boxes_xywh[:, :2, ...] = anchors[:, 2:, ...] * loc[:, :2, ...] + anchors[:, :2, ...]
    boxes_xywh[:, 2:, ...] = torch.exp(loc[:, 2:, ...]) * anchors[:, 2:, ...]
    boxes[:, :2, ...] = boxes_xywh[:, :2, ...] - boxes_xywh[:, 2:, ...] / 2.
    boxes[:, 2:, ...] = boxes_xywh[:, :2, ...] + boxes_xywh[:, 2:, ...] / 2.
    boxes = torch.clamp(boxes, min=0., max=boundary)
    return score, boxes
