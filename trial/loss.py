

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from trial.utils.iou import bbox_iou, bbox_ciou, bbox_diou, bbox_giou


epsilon = 1e-7


def centerness_target(label_loc):
    with torch.no_grad():
        left_right = label_loc[:, [0, 2], ...]
        top_bottom = label_loc[:, [1, 3], ...]
        centerness = (left_right.min(dim=1)[0] / (left_right.max(dim=1)[0] + epsilon)) * \
                     (top_bottom.min(dim=1)[0] / (top_bottom.max(dim=1)[0] + epsilon))

        center_mask = (centerness >= 0.).type(torch.float32)

        centerness = torch.sqrt(centerness * center_mask)
        return centerness, center_mask


def weighted_bce_loogits(target, pred, weight):
    loss = F.binary_cross_entropy_with_logits(pred, target, weight=weight, reduce=None, reduction='none')
    loss_ = loss.sum().div(weight.sum() + epsilon)
    return loss_


def weighted_l2_loss(target, delta, weight):
    loss = F.mse_loss(delta, target, reduce=None, reduction='none')
    loss_ = loss * weight
    loss_ = loss_.sum().div(weight.sum() + epsilon)
    return loss_


def weighted_l1_loss(target, delta, weight, smooth=False):
    if smooth:
        loss = F.smooth_l1_loss(delta, target, reduce=None, reduction='none')
    else:
        loss = F.l1_loss(delta, target, reduce=None, reduction='none')
    loss = loss.sum(dim=1)
    loss_ = loss * weight
    loss_ = loss_.sum().div(weight.sum() + epsilon)
    return loss_


def weighted_iou_loss(bbox, boxes, weight, loss_type='linear_iou', iou=None, union_area=None):
    if loss_type == 'iou' or loss_type == 'linear_iou':
        if iou is None:
            iou_ = iou = bbox_iou(bbox, boxes)[0]
        else:
            iou_ = iou

    elif loss_type == 'giou':
        iou_, iou = bbox_giou(bbox, boxes, iou=iou, union_area=union_area)

    elif loss_type == 'diou':
        iou_, iou = bbox_diou(bbox, boxes, iou=iou, union_area=union_area)

    elif loss_type == 'ciou':
        iou_, iou = bbox_ciou(bbox, boxes, iou=iou, union_area=union_area)

    if loss_type == 'iou':
        loss = -torch.log(iou_)
    else:
        loss = 1 - iou_

    loss_ = loss * weight
    loss_ = loss_.sum().div(weight.sum() + epsilon)
    return loss_, iou, iou_


def weighted_select_cross_entropy_loss(pred, label,
                                       pos_weight=None, neg_weight=None,
                                       pos_mod=False, neg_mod=False,
                                       pos_avg='batch', neg_avg='sample'):
    # pred 已经进行了log_sofmax处理，相当于可以直接作为CE损失的值
    # pred = torch.clamp(pred, min=epsilon, max=1e4)
    if pos_weight is None:
        with torch.no_grad():
            pos_mask = torch.eq(label, torch.ones_like(label)).type(torch.float32)
            if pos_mod:
                pl = - pos_mask * pred[..., 1]
                pos_weight = modulate_weight(pl, pos_mask, avg=pos_avg)
            else:
                pos_weight = label2weight(pos_mask, avg=pos_avg)

    pos_loss = - pos_weight * pred[..., 1]
    pos_loss_ = pos_loss.sum().div(pos_weight.sum() + epsilon)

    if neg_weight is None:
        neg_mask = torch.eq(label, torch.zeros_like(label)).type(torch.float32)
        if neg_mod:
            nl = - neg_mask * pred[..., 0]
            neg_weight = modulate_weight(nl, neg_mask, avg=neg_avg)
        else:
            neg_weight = label2weight(neg_mask, avg=neg_avg)
    neg_loss = - neg_weight * pred[..., 0]
    neg_loss_ = neg_loss.sum().div(neg_weight.sum() + epsilon)
    return pos_loss_, neg_loss_


def weighted_focal_loss(pred_log, pred, pos_weight, neg_weight, alpha=0.25, gamma=2.):
    pred_softmax = torch.softmax(pred, dim=-1)
    pos_loss = - pos_weight * alpha * torch.pow(1. - pred_softmax[..., 1], gamma) * pred_log[..., 1]
    neg_loss = - neg_weight * (1. - alpha) * torch.pow(1. - pred_softmax[..., 0], gamma) * pred_log[..., 0]

    pos_loss_ = pos_loss.sum().div(pos_weight.sum() + epsilon)
    neg_loss_ = neg_loss.sum().div(neg_weight.sum() + epsilon)
    return pos_loss_, neg_loss_


def sofTmax(x, T=1., b=0., mask=None, average='sample'):
    out = torch.exp((x - b) / T)
    if mask is not None:
        out = out * mask
    if average == 'batch':
        dim = [i for i in range(1, out.ndim)]
        out = out / (out.sum(dim=dim, keepdim=True) + epsilon)
    return out


def label2weight(label, avg='sample'):
    with torch.no_grad():
        if avg == 'batch':
            dim = [i for i in range(1, label.ndim)]
            weight = label / (label.sum(dim=dim, keepdim=True) + epsilon)
        elif avg == 'sample':
            weight = label
        return weight


def random_choice_batch(label, max_num):
    # shape = (b, 32, 32)
    labels = []
    label_ = np.zeros_like(label[0, ...])
    batch = label.shape[0]
    for i in range(batch):
        index = np.where(label[i, ...] == 1.)
        num = index[0].shape[0]
        if num > max_num:
            slt = np.arange(num)
            random.shuffle(slt)
            slt = slt[:max_num]
            index = tuple(p[slt] for p in index)
        if num > 0:
            label_[index] = 1.
        labels.append(label_)
    return np.array(labels)


def random_choice(label, max_num):
    label_ = np.zeros_like(label)
    index = np.where(label == 1.)
    num = index[0].shape[0]
    if num > max_num:
        slt = np.arange(num)
        random.shuffle(slt)
        slt = slt[:max_num]
        index = tuple(p[slt] for p in index)
    if num > 0:
        label_[index] = 1.
    return label_


def label_update(label, score, iou, positive, base='anchor',
                 pos_iou_thresh=0.75, pos_num=16,
                 neg_iou_thresh=0.4, neg_score_thresh=0.25, neg_num=48, easy_neg_num=16, hard_neg_num=32):
    batch = label.shape[0]
    new_label = np.ones_like(label)
    if base == 'anchor':
        positive_ = positive[:, None, None, None]
    elif base == 'point':
        positive_ = positive[:, None, None]
    negative = positive_ == 0.
    negative = negative.astype(np.float32)

    # 预备正样本
    pos_ = label == 1.
    pos_ = pos_.astype(np.float32)

    # pos范围
    pos_mask = label >= 0.
    pos_mask = pos_mask.astype(np.float32)

    # neg范围
    label_ = label * positive_ - 1. * negative * np.ones_like(label)
    neg_mask = label_ < 0.
    neg_mask = neg_mask.astype(np.float32)

    # 首先筛选出所有的正样本对内pos范围内点上的iou值
    pos_iou = iou * pos_mask
    # 进一步筛选出iou值大于阈值的点作为正样本
    pos = pos_iou >= pos_iou_thresh
    pos = pos.astype(np.float32)
    pos_label = pos * positive_

    # 检查每个batch是否有符合要求的正样本
    dim = [i for i in range(1, label.ndim)]
    pos_num_ = pos_label.sum(tuple(dim))
    if_pos = pos_num_ > 0.
    if_pos = if_pos.astype(np.float32)

    neg_label = np.zeros_like(pos_label)
    # 正样本范围外且得分高于阈值的box
    neg_score = score * neg_mask
    neg0 = neg_score >= neg_score_thresh
    neg0 = neg0.astype(np.float32)
    # 正样本范围外且IOU小于于阈值的box
    neg_iou = iou * neg_mask
    neg1 = neg_iou < neg_iou_thresh
    neg1 = neg1.astype(np.float32)
    # 两者的交集为异常高得分，将作为难负样本
    hard_neg = neg0 * neg1

    # 样本的清洗筛选
    for b in range(batch):
        hard_neg_ = random_choice(hard_neg[b], random.randint(hard_neg_num - 4, hard_neg_num + 4))
        easy_neg_ = random_choice(neg_mask[b], random.randint(easy_neg_num - 2, easy_neg_num + 2))

        if positive[b]:
            if if_pos[b]:
                # 当满足条件的box数超过总数时，选取满足iou条件的前K名box作为正样本
                max_pos_num = random.randint(pos_num-2, pos_num+2)
                if pos_num_[b] > max_pos_num:
                    p = np.where(pos_iou[b] > 0.)
                    pos_index = np.argpartition(pos_iou[b][p], -max_pos_num, axis=0)[-max_pos_num:]
                    p_ = tuple(i[pos_index] for i in p)
                    # 先清空，再填充
                    pos_label[b] = 0.
                    pos_label[b][p_] = 1.
            # 有的batch应该是正样本对，但是没有符合IOU阈值要求的box
            # 无满足条件的box时，使用原分配方式产生的预备正样本标签
            else:
                pos_label[b] = pos_[b]

        # 负样本对时，强制将box位置设置为负样本（基于预备label）
        else:
            hard_neg_ = hard_neg_ + pos_[b]
        neg_label_ = hard_neg_ + easy_neg_
        neg_label_ = neg_label_ > 0.
        neg_label_ = neg_label_.astype(np.float32)
        neg_label[b] = random_choice(neg_label_, random.randint(neg_num - 4, neg_num + 4))

    # 正Box位置标记为1，负样本位置标记为0，其余位置标记为-1，忽略
    new_label = -1. * new_label + 1. * neg_label + 2. * pos_label
    return torch.from_numpy(new_label).type(torch.int64).cuda()


def label_update_(label, score, iou, positive, base='point',
                  pos_iou_thresh=0.75, pos_num=12,
                  neg_iou_thresh=0.4, neg_score_thresh=0.25, neg_num=48, easy_neg_num=16, hard_neg_num=32):
    batch = label.shape[0]
    new_label = np.ones_like(label)
    if base == 'anchor':
        positive_ = positive[:, None, None, None]
    elif base == 'point':
        positive_ = positive[:, None, None]

    pos_label, neg_label = np.zeros_like(label), np.zeros_like(label)
    # 预备正样本
    pos_ = (label == 1.).astype(np.float32)
    # pos范围
    pos_mask = (label >= 0.).astype(np.float32)
    # neg范围
    neg_mask = (label < 0.).astype(np.float32)
    # 负样本对上的iou值置0
    iou = iou * positive_

    """正样本筛选"""
    # 首先筛选出所有的正样本对上的pos范围内候选点的iou值
    pos_iou = iou * pos_mask
    # 进一步筛选出iou值大于阈值的候选点
    pos = (pos_iou >= pos_iou_thresh).astype(np.float32)
    # 检查每个batch是否有符合要求的正样本
    dim = [i for i in range(1, label.ndim)]
    pos_num_ = pos.sum(tuple(dim))

    """负样本筛选"""
    # 正样本范围外且得分高于阈值的box
    neg_score = score * neg_mask
    neg = (neg_score >= neg_score_thresh).astype(np.float32)
    # 正样本范围外且IOU小于于阈值的box, 负样本对时不考虑IOU条件
    neg_iou = iou * neg_mask
    easy_neg = (neg_iou < neg_iou_thresh).astype(np.float32)
    # 两者的交集为异常高得分，将作为难负样本
    hard_neg = neg * easy_neg

    # 样本的清洗筛选
    for b in range(batch):
        hard_neg_ = random_choice(hard_neg[b], random.randint(hard_neg_num - 4, hard_neg_num + 4))
        easy_neg_ = random_choice(easy_neg[b], random.randint(easy_neg_num - 2, easy_neg_num + 2))

        neg_label_ = hard_neg_ + easy_neg_
        neg_label_ = (neg_label_ > 0.).astype(np.float32)
        neg_label[b] = random_choice(neg_label_, random.randint(neg_num - 4, neg_num + 4))

        if positive[b]:
            pos_label[b] = pos_[b]
        # if positive[b]:
        #     if pos_num_[b] > 0.:
        #         # 当满足条件的box数超过总数时，选取满足iou条件的前K名box作为正样本
        #         max_pos_num = random.randint(pos_num-2, pos_num+2)
        #         if pos_num_[b] > max_pos_num:
        #             p = np.where(pos_iou[b] > 0.)
        #             pos_index = np.argpartition(pos_iou[b][p], -max_pos_num, axis=0)[-max_pos_num:]
        #             p_ = tuple(i[pos_index] for i in p)
        #             pos_label[b][p_] = 1.
        #     # 有的batch应该是正样本对，但是没有符合IOU阈值要求的box
        #     # 无满足条件的box时，使用原分配方式产生的预备正样本标签
        #     else:
        #         pos_label[b] = pos_[b]

    # 正Box位置标记为1，负样本位置标记为0，其余位置标记为-1，忽略
    new_label = -1. * new_label + 1. * neg_label + 2. * pos_label
    return torch.from_numpy(new_label).type(torch.int64).cuda()


def modulate_weight(loss, mask, a=5.0, c=0.2, v=1.0, avg='sample'):
    with torch.no_grad():
        loss_ = loss.div(v)
        weight = 1. / (1. + torch.exp(a * (c - loss_)))
        weight_exp = torch.exp(weight) * mask
        if avg == 'sample':
            weight_softmax = weight_exp / (weight_exp.sum() + epsilon)
            return weight_softmax
        elif avg == 'batch':
            dim = [i for i in range(1, weight_exp.ndim)]
            weight_softmax = weight_exp / (weight_exp.sum(dim=dim, keepdim=True) + epsilon)
            return weight_softmax


if __name__ == '__main__':
    mask = torch.rand((8, 4, 4)) > 0.95
    mask = mask.type(torch.float32)
    modulate_weight(torch.rand((8, 4, 4)), mask, avg='batch')
    pass
