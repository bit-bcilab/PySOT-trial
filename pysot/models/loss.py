# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class IOULoss(nn.Module):
    def __init__(self, loc_loss_type):
        super(IOULoss, self).__init__()
        self.loc_loss_type = loc_loss_type

    def forward(self, pred, target, weight=None):
        pred_left = pred[:, 0]
        pred_top = pred[:, 1]
        pred_right = pred[:, 2]
        pred_bottom = pred[:, 3]

        target_left = target[:, 0]
        target_top = target[:, 1]
        target_right = target[:, 2]
        target_bottom = target[:, 3]

        pred_area = (pred_left + pred_right) * (pred_top + pred_bottom)
        target_area = (target_left + target_right) * (target_top + target_bottom)

        w_intersect = torch.min(pred_left, target_left) + torch.min(pred_right, target_right)
        h_intersect = torch.min(pred_bottom, target_bottom) + torch.min(pred_top, target_top)
        area_intersect = w_intersect * h_intersect
        area_union = target_area + pred_area - area_intersect
        ious = (area_intersect + 1.0) / (area_union + 1.0)

        if self.loc_loss_type == 'iou':
            losses = -torch.log(ious)
        elif self.loc_loss_type == 'linear_iou':
            losses = 1 - ious
        elif self.loc_loss_type == 'giou':
            g_w_intersect = torch.max(pred_left, target_left) + torch.max(pred_right, target_right)
            g_h_intersect = torch.max(pred_bottom, target_bottom) + torch.max(pred_top, target_top)
            ac_uion = g_w_intersect * g_h_intersect + 1e-7
            gious = ious - (ac_uion - area_union) / ac_uion
            losses = 1 - gious
        else:
            raise NotImplementedError

        if weight is not None and weight.sum() > 0:
            return (losses * weight).sum() / weight.sum()
        else:
            assert losses.numel() != 0
            return losses.mean()


class SiamCARLossComputation(object):
    """
    This class computes the SiamCAR losses.
    """

    def __init__(self, cfg):
        self.box_reg_loss_func = IOULoss(loc_loss_type='iou')
        self.centerness_loss_func = nn.BCEWithLogitsLoss()
        self.cfg = cfg

    def prepare_targets(self, points, labels, gt_bbox):

        labels, reg_targets = self.compute_targets_for_locations(
            points, labels, gt_bbox
        )

        return labels, reg_targets

    def compute_targets_for_locations(self, locations, labels, gt_bbox):
        # reg_targets = []
        xs, ys = locations[:, 0], locations[:, 1]

        bboxes = gt_bbox
        labels = labels.view(self.cfg.TRAIN.OUTPUT_SIZE ** 2, -1)

        l = xs[:, None] - bboxes[:, 0][None].float()
        t = ys[:, None] - bboxes[:, 1][None].float()
        r = bboxes[:, 2][None].float() - xs[:, None]
        b = bboxes[:, 3][None].float() - ys[:, None]
        reg_targets_per_im = torch.stack([l, t, r, b], dim=2)

        s1 = reg_targets_per_im[:, :, 0] > 0.6 * ((bboxes[:, 2] - bboxes[:, 0]) / 2).float()
        s2 = reg_targets_per_im[:, :, 2] > 0.6 * ((bboxes[:, 2] - bboxes[:, 0]) / 2).float()
        s3 = reg_targets_per_im[:, :, 1] > 0.6 * ((bboxes[:, 3] - bboxes[:, 1]) / 2).float()
        s4 = reg_targets_per_im[:, :, 3] > 0.6 * ((bboxes[:, 3] - bboxes[:, 1]) / 2).float()
        is_in_boxes = s1 * s2 * s3 * s4
        pos = np.where(is_in_boxes.cpu() == 1)
        labels[pos] = 1

        return labels.permute(1, 0).contiguous(), reg_targets_per_im.permute(1, 0, 2).contiguous()

    def compute_centerness_targets(self, reg_targets):
        left_right = reg_targets[:, [0, 2]]
        top_bottom = reg_targets[:, [1, 3]]
        centerness = (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * \
                     (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness)

    def __call__(self, locations, box_cls, box_regression, centerness, labels, reg_targets):
        """
        Arguments:
            locations (list[BoxList])
            box_cls (list[Tensor])
            box_regression (list[Tensor])
            centerness (list[Tensor])
            targets (list[BoxList])

        Returns:
            cls_loss (Tensor)
            reg_loss (Tensor)
            centerness_loss (Tensor)
        """

        label_cls, reg_targets = self.prepare_targets(locations, labels, reg_targets)
        box_regression_flatten = (box_regression.permute(0, 2, 3, 1).contiguous().view(-1, 4))
        labels_flatten = (label_cls.view(-1))
        reg_targets_flatten = (reg_targets.view(-1, 4))
        centerness_flatten = (centerness.view(-1))

        pos_inds = torch.nonzero(labels_flatten > 0).squeeze(1)

        box_regression_flatten = box_regression_flatten[pos_inds]
        reg_targets_flatten = reg_targets_flatten[pos_inds]
        centerness_flatten = centerness_flatten[pos_inds]
        cls_loss = select_cross_entropy_loss(box_cls, labels_flatten)

        if pos_inds.numel() > 0:
            centerness_targets = self.compute_centerness_targets(reg_targets_flatten)
            reg_loss = self.box_reg_loss_func(
                box_regression_flatten,
                reg_targets_flatten,
                centerness_targets
            )
            centerness_loss = self.centerness_loss_func(
                centerness_flatten,
                centerness_targets
            )
        else:
            reg_loss = box_regression_flatten.sum()
            centerness_loss = centerness_flatten.sum()

        return cls_loss, reg_loss, centerness_loss


class SiamGATLossComputation(object):
    """
    This class computes the SiamCAR losses.
    """

    def __init__(self, cfg):
        # self.box_reg_loss_func = DIOULoss()
        self.box_reg_loss_func = IOULoss('iou')
        self.centerness_loss_func = nn.BCEWithLogitsLoss()
        self.cfg = cfg

    def prepare_targets(self, points, labels, gt_bbox, neg):

        labels, reg_targets, pos_area = self.compute_targets_for_locations(
            points, labels, gt_bbox, neg
        )

        return labels, reg_targets, pos_area

    def compute_targets_for_locations(self, locations, labels, gt_bbox, neg):
        # reg_targets = []
        xs, ys = locations[:, 0], locations[:, 1]

        bboxes = gt_bbox
        labels = labels.view(self.cfg.TRAIN.OUTPUT_SIZE ** 2, -1)
        pos_area = torch.zeros_like(labels)

        l = xs[:, None] - bboxes[:, 0][None].float()
        t = ys[:, None] - bboxes[:, 1][None].float()
        r = bboxes[:, 2][None].float() - xs[:, None]
        b = bboxes[:, 3][None].float() - ys[:, None]
        reg_targets_per_im = torch.stack([l, t, r, b], dim=2)

        # #################################points in the gt_bbox area###################################
        all_s1 = reg_targets_per_im[:, :, 0] > 0
        all_s2 = reg_targets_per_im[:, :, 2] > 0
        all_s3 = reg_targets_per_im[:, :, 1] > 0
        all_s4 = reg_targets_per_im[:, :, 3] > 0
        all_in_boxes = all_s1 * all_s2 * all_s3 * all_s4
        all_pos = np.where(all_in_boxes.cpu() == 1)
        pos_area[all_pos] = 1

        # #########################################ignore labels###########################################
        ignore_s1 = reg_targets_per_im[:, :, 0] > 0.2 * ((bboxes[:, 2] - bboxes[:, 0]) / 2).float()
        ignore_s2 = reg_targets_per_im[:, :, 2] > 0.2 * ((bboxes[:, 2] - bboxes[:, 0]) / 2).float()
        ignore_s3 = reg_targets_per_im[:, :, 1] > 0.2 * ((bboxes[:, 3] - bboxes[:, 1]) / 2).float()
        ignore_s4 = reg_targets_per_im[:, :, 3] > 0.2 * ((bboxes[:, 3] - bboxes[:, 1]) / 2).float()
        ignore_in_boxes = ignore_s1 * ignore_s2 * ignore_s3 * ignore_s4
        ignore_pos = np.where(ignore_in_boxes.cpu() == 1)
        labels[ignore_pos] = -1

        s1 = reg_targets_per_im[:, :, 0] > 0.5 * ((bboxes[:, 2] - bboxes[:, 0]) / 2).float()
        s2 = reg_targets_per_im[:, :, 2] > 0.5 * ((bboxes[:, 2] - bboxes[:, 0]) / 2).float()
        s3 = reg_targets_per_im[:, :, 1] > 0.5 * ((bboxes[:, 3] - bboxes[:, 1]) / 2).float()
        s4 = reg_targets_per_im[:, :, 3] > 0.5 * ((bboxes[:, 3] - bboxes[:, 1]) / 2).float()
        is_in_boxes = s1 * s2 * s3 * s4
        pos = np.where(is_in_boxes.cpu() == 1)
        labels[pos] = 1
        labels = labels * (1 - neg.long())

        return labels.permute(1, 0).contiguous(), reg_targets_per_im.permute(1, 0, 2).contiguous(), pos_area.permute(1, 0).contiguous()

    def compute_centerness_targets(self, reg_targets):
        left_right = reg_targets[:, [0, 2]]
        top_bottom = reg_targets[:, [1, 3]]
        centerness = (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * \
                     (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness)

    def __call__(self, locations, box_cls, box_regression, centerness, labels, reg_targets, neg):
        """
        Arguments:
            locations (list[BoxList])
            box_cls (list[Tensor])
            box_regression (list[Tensor])
            centerness (list[Tensor])
            targets (list[BoxList])

        Returns:
            cls_loss (Tensor)
            reg_loss (Tensor)
            centerness_loss (Tensor)
        """

        label_cls, reg_targets, pos_area = self.prepare_targets(locations, labels, reg_targets, neg)
        box_regression_flatten = (box_regression.permute(0, 2, 3, 1).contiguous().view(-1, 4))
        labels_flatten = (label_cls.view(-1))
        reg_targets_flatten = (reg_targets.view(-1, 4))
        centerness_flatten = (centerness.view(-1))

        pos_inds = torch.nonzero(labels_flatten > 0).squeeze(1)

        ###########################change cen and reg area###################################
        pos_area_flatten = (pos_area.view(-1))
        all_pos_idx = torch.nonzero(pos_area_flatten > 0).squeeze(1)

        box_regression_flatten = box_regression_flatten[all_pos_idx]
        reg_targets_flatten = reg_targets_flatten[all_pos_idx]
        centerness_flatten = centerness_flatten[all_pos_idx]

        #####################################################################################
        # box_regression_flatten = box_regression_flatten[pos_inds]
        # reg_targets_flatten = reg_targets_flatten[pos_inds]
        # centerness_flatten = centerness_flatten[pos_inds]
        cls_loss = select_cross_entropy_loss(box_cls, labels_flatten)

        if pos_inds.numel() > 0:
            centerness_targets = self.compute_centerness_targets(reg_targets_flatten)
            reg_loss = self.box_reg_loss_func(
                box_regression_flatten,
                reg_targets_flatten,
                centerness_targets
            )
            centerness_loss = self.centerness_loss_func(
                centerness_flatten,
                centerness_targets
            )
        else:
            reg_loss = box_regression_flatten.sum()
            centerness_loss = centerness_flatten.sum()

        return cls_loss, reg_loss, centerness_loss


def get_cls_loss(pred, label, select):
    if len(select.size()) == 0 or select.size() == torch.Size([0]):
        return 0
    pred = torch.index_select(pred, 0, select)
    label = torch.index_select(label, 0, select)
    return F.nll_loss(pred, label)


def select_cross_entropy_loss(pred_, label_, add=True):
    pred = pred_.view(-1, 2)
    label = label_.view(-1)
    pos = label.data.eq(1).nonzero().squeeze().cuda()
    neg = label.data.eq(0).nonzero().squeeze().cuda()
    loss_pos = get_cls_loss(pred, label, pos)
    loss_neg = get_cls_loss(pred, label, neg)
    if add:
        return loss_pos * 0.5 + loss_neg * 0.5
    else:
        return loss_pos, loss_neg


def weight_l1_loss(pred_loc, label_loc, loss_weight):
    b, _, sh, sw = pred_loc.size()
    pred_loc = pred_loc.view(b, 4, -1, sh, sw)
    diff = (pred_loc - label_loc).abs()
    diff = diff.sum(dim=1).view(b, -1, sh, sw)
    loss = diff * loss_weight
    return loss.sum().div(b)


def select_iou_loss(pred_loc, label_loc, label_cls):
    label_cls = label_cls.reshape(-1)
    pos = label_cls.data.eq(1).nonzero().squeeze().cuda()

    pred_loc = pred_loc.permute(0, 2, 3, 1).reshape(-1, 4)
    pred_loc = torch.index_select(pred_loc, 0, pos)

    label_loc = label_loc.permute(0, 2, 3, 1).reshape(-1, 4)
    label_loc = torch.index_select(label_loc, 0, pos)

    return IOULoss(loc_loss_type='linear_iou')(pred_loc, label_loc)


def mask_loss_bce(masks_pred, masks_gt, mask_weight, ohem=True):
    """Mask R-CNN specific losses."""
    mask_weight = mask_weight.view(-1)
    pos = mask_weight.data.eq(1).nonzero().squeeze()
    if pos.nelement() == 0:
        return masks_pred.sum() * 0, masks_pred.sum() * 0, masks_pred.sum() * 0, masks_pred.sum() * 0

    masks_pred = torch.index_select(masks_pred, 0, pos)
    masks_gt = torch.index_select(masks_gt, 0, pos)

    _, _, h, w = masks_pred.size()
    masks_pred = masks_pred.view(-1, h*w)
    masks_gt = Variable(masks_gt.view(-1, h*w), requires_grad=False)

    if ohem:
        top_k = 0.7
        loss = F.binary_cross_entropy_with_logits(masks_pred, masks_gt, reduction='none')
        loss = loss.view(-1)
        index = torch.topk(loss, int(top_k * loss.size()[0]))
        loss = torch.mean(loss[index[1]])
    else:
        loss = F.binary_cross_entropy_with_logits(masks_pred, masks_gt)

    iou_m, iou_5, iou_7 = iou_measure(masks_pred, masks_gt)
    return loss, iou_m, iou_5, iou_7


def iou_measure(pred, label):
    pred = pred.ge(0)
    mask_sum = pred.eq(1).add(label.eq(1))
    intxn = torch.sum(mask_sum == 2, dim=1).float()
    union = torch.sum(mask_sum > 0, dim=1).float()
    iou = intxn / (union + 1e-5)
    return torch.mean(iou), (torch.sum(iou > 0.5).float()/iou.shape[0]), (torch.sum(iou > 0.7).float()/iou.shape[0])


def det_loss_smooth_l1(bboxes_pred, bboxes_gt, bbox_weight):
    bbox_weight = bbox_weight.view(-1)
    pos = bbox_weight.data.eq(1).nonzero().squeeze()
    if pos.nelement() == 0:
        return bboxes_pred.sum() * 0

    bboxes_pred = torch.index_select(bboxes_pred, 0, pos)
    bboxes_gt = torch.index_select(bboxes_gt, 0, pos)
    bboxes_gt = Variable(bboxes_gt, requires_grad=False)

    bbox_loss = F.smooth_l1_loss(bboxes_pred, bboxes_gt)

    return bbox_loss
