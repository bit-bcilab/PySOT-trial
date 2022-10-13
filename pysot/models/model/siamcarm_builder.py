# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import math
import torch
import torch.nn as nn

from pysot.models.model.base_model_builder import BaseModelBuilder
from pysot.models.xcorr import xcorr_depthwise
from pysot.models.loss import SiamCARLossComputation
from trial.loss import sofTmax, label2weight, label_update_
from trial.loss import weighted_select_cross_entropy_loss, weighted_iou_loss, weighted_l1_loss
from trial.utils.iou import process_box, bbox_iou
from trial.Decoders import LTRBDecoder


class CARHead(torch.nn.Module):
    def __init__(self, cfg, in_channels):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(CARHead, self).__init__()
        # TODO: Implement the sigmoid version first.
        num_classes = cfg.TRAIN.NUM_CLASSES

        cls_tower = []
        bbox_tower = []
        for i in range(cfg.TRAIN.NUM_CONVS):
            cls_tower.append(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            )
            cls_tower.append(nn.GroupNorm(32, in_channels))
            cls_tower.append(nn.ReLU())
            bbox_tower.append(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            )
            bbox_tower.append(nn.GroupNorm(32, in_channels))
            bbox_tower.append(nn.ReLU())

        self.add_module('cls_tower', nn.Sequential(*cls_tower))
        self.add_module('bbox_tower', nn.Sequential(*bbox_tower))
        self.cls_logits = nn.Conv2d(
            in_channels, num_classes, kernel_size=3, stride=1,
            padding=1
        )
        self.bbox_pred = nn.Conv2d(
            in_channels, 4, kernel_size=3, stride=1,
            padding=1
        )

        # initialization
        for modules in [self.cls_tower, self.bbox_tower,
                        self.cls_logits, self.bbox_pred]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

        # initialize the bias for focal loss
        prior_prob = cfg.TRAIN.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)

    def forward(self, x):
        cls_tower = self.cls_tower(x)
        logits = self.cls_logits(cls_tower)
        bbox_reg = torch.exp(self.bbox_pred(self.bbox_tower(x)))
        return logits, bbox_reg


class SiamCARMBuilder(BaseModelBuilder):
    def __init__(self, cfg):
        super(SiamCARMBuilder, self).__init__(cfg)

        # build car head
        self.car_head = CARHead(cfg, 256)

        # build response map
        self.xcorr_depthwise = xcorr_depthwise

        # build loss
        self.loss_evaluator = SiamCARLossComputation(cfg)

        self.down = nn.ConvTranspose2d(256 * 3, 256, 1, 1)

        self.T = 0.5
        self.ti = 0.3
        self.ts = 0.3

    def get_head_parameters(self):
        head_params = [self.car_head.parameters(), self.down.parameters()]
        return head_params

    def template(self, z):
        zf = self.backbone(z)
        if self.cfg.ADJUST.ADJUST:
            zf = self.neck(zf)
        self.zf = zf

    def track(self, x):
        xf = self.backbone(x)
        if self.cfg.ADJUST.ADJUST:
            xf = self.neck(xf)

        features = self.xcorr_depthwise(xf[0], self.zf[0])
        for i in range(len(xf) - 1):
            features_new = self.xcorr_depthwise(xf[i + 1], self.zf[i + 1])
            features = torch.cat([features, features_new], 1)
        features = self.down(features)

        cls, loc = self.car_head(features)
        return {
            'cls': cls,
            'loc': loc
        }

    def forward_original(self, data):
        """ only used in training
        """
        template = data['template'].cuda()
        search = data['search'].cuda()
        label_cls = data['label_cls'].cuda()
        label_loc = data['bbox'].cuda()

        # get feature
        zf = self.backbone(template)
        xf = self.backbone(search)
        if self.cfg.ADJUST.ADJUST:
            zf = self.neck(zf)
            xf = self.neck(xf)

        features = self.xcorr_depthwise(xf[0], zf[0])
        for i in range(len(xf) - 1):
            features_new = self.xcorr_depthwise(xf[i + 1], zf[i + 1])
            features = torch.cat([features, features_new], 1)
        features = self.down(features)

        cls_, loc, cen = self.car_head(features)

        cls = self.log_softmax(cls_)
        locations = compute_locations(cls_, self.cfg.TRACK.STRIDE, self.cfg.TRACK.OFFSET)
        cls_loss, loc_loss, cen_loss = self.loss_evaluator(
            locations,
            cls,
            loc,
            cen, label_cls, label_loc
        )

        # get loss
        outputs = {}
        outputs['total_loss'] = self.cfg.TRAIN.CLS_WEIGHT * cls_loss + \
                                self.cfg.TRAIN.LOC_WEIGHT * loc_loss + self.cfg.TRAIN.CEN_WEIGHT * cen_loss
        outputs['cls_loss'] = cls_loss
        outputs['loc_loss'] = loc_loss
        outputs['cen_loss'] = cen_loss
        return outputs

    def forward_trial(self, data):
        # return self.forward_trial_old(data)
        return self.forward_trial_new(data)

    def forward_trial_new(self, data):
        template = data['template'].cuda()
        search = data['search'].cuda()
        label_cls = data['label_cls'].cuda()
        label_loc = data['label_loc'].cuda()
        bbox = data['bbox'].cuda()[:, None, None, :]
        positive = data['pos'].cuda()

        # get feature
        zf = self.backbone(template)
        xf = self.backbone(search)
        if self.cfg.ADJUST.ADJUST:
            zf = self.neck(zf)
            xf = self.neck(xf)

        features = self.xcorr_depthwise(xf[0], zf[0])
        for i in range(len(xf) - 1):
            features_new = self.xcorr_depthwise(xf[i + 1], zf[i + 1])
            features = torch.cat([features, features_new], 1)
        features = self.down(features)

        cls_, loc = self.car_head(features)

        loc = torch.clamp(loc, max=1e4)
        loc = torch.where(torch.isinf(loc), 1e4 * torch.ones_like(loc), loc)

        cls = self.log_softmax(cls_)
        score, boxes = LTRBDecoder(cls_, loc, self.points, self.cfg.TRAIN.SEARCH_SIZE)
        boxes = boxes.permute(0, 2, 3, 1).contiguous()
        bbox = process_box(bbox)
        boxes = process_box(boxes)
        iou, union_area = bbox_iou(bbox, boxes)

        with torch.no_grad():
            # 基于self-training方式，根据pred boxes与GT直接的iou以及boxes的score，重新划分正负样本
            if self.update_settings is not None and not self.validate:
                label_cls = label_update_(label_cls.cpu().detach().numpy(),
                                          score.cpu().detach().numpy(), iou.cpu().detach().numpy(),
                                          positive.cpu().detach().numpy(), **self.update_settings)
            pos_mask = (label_cls == 1.).type(torch.float32)
            neg_mask = (label_cls == 0.).type(torch.float32)

            if 'weighted' in self.cfg.MODE and self.train_epoch > 0 and not self.validate:
                iou_weight = sofTmax(-iou * positive[:, None, None], T=self.T, b=-self.ti, mask=neg_mask, average='batch')
                neg_weight = sofTmax(score, T=self.T, b=self.ts, mask=neg_mask * iou_weight, average='batch')
                pos_weight_cls = sofTmax(iou, T=self.T, b=(1 - self.ti), mask=pos_mask, average='batch')
                pos_weight_l1 = sofTmax(score, T=self.T, b=(1 - self.ts), mask=pos_mask, average='batch')
            else:
                neg_weight = label2weight(neg_mask, avg='batch')
                pos_weight_cls = label2weight(pos_mask, avg='batch')
                pos_weight_l1 = pos_weight_cls
            pos_weight_iou = pos_weight_l1

        l1_loss = weighted_l1_loss(label_loc, loc, pos_weight_l1, smooth=True)

        # cross entropy loss
        pos_loss, neg_loss = weighted_select_cross_entropy_loss(cls, label_cls,
                                                                pos_weight=pos_weight_cls, neg_weight=neg_weight)
        cls_loss = pos_loss * self.weights['pos_weight'] + neg_loss * self.weights['neg_weight']

        iou_loss = weighted_iou_loss(bbox, boxes, weight=pos_weight_iou,
                                     iou=iou, union_area=union_area, loss_type='ciou')[0]

        loc_loss = iou_loss * self.weights['iou_weight']
        if self.train_epoch > 0 and self.weights['l1_weight']:
            loc_loss += l1_loss * self.weights['l1_weight']

        # a0 = data['label_cls'].numpy()
        # a1 = iou.cpu().detach().numpy()
        # a2 = score.cpu().detach().numpy()
        # a3 = neg_weight.cpu().detach().numpy()
        # import cv2
        # search_img = search.permute((0, 2, 3, 1)).contiguous().type(torch.uint8).cpu().detach().numpy()
        # template_img = template.permute((0, 2, 3, 1)).contiguous().type(torch.uint8).cpu().detach().numpy()
        # # j = 2
        # # cv2.imshow('0', template_img[j])
        # # box = list(map(int, data['bbox'].numpy()[j]))
        # # img = search_img[j]
        # # cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        # # cv2.imshow('1', img)
        # # cv2.waitKey()
        # for j in range(search_img.shape[0]):
        #     cv2.imshow(str(j) + '-template', template_img[j])
        #     box = list(map(int, data['bbox'].numpy()[j]))
        #     img = search_img[j]
        #     cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        #     cv2.imshow(str(j) + '-search', img)
        # cv2.waitKey()

        outputs = {}
        outputs['total_loss'] = cls_loss + loc_loss
        outputs['cls_loss'] = cls_loss
        outputs['pos_loss'] = pos_loss
        outputs['neg_loss'] = neg_loss
        outputs['loc_loss'] = loc_loss
        outputs['iou_loss'] = iou_loss
        outputs['l1_loss'] = l1_loss
        return outputs

    def forward_trial_old(self, data):
        """ only used in training
        """
        template = data['template'].cuda()
        search = data['search'].cuda()
        label_cls = data['label_cls'].cuda()
        label_loc = data['label_loc'].cuda()
        bbox = data['bbox'].cuda()[:, None, None, :]
        positive = data['pos'].cuda()

        # get feature
        zf = self.backbone(template)
        xf = self.backbone(search)
        if self.cfg.ADJUST.ADJUST:
            zf = self.neck(zf)
            xf = self.neck(xf)

        features = self.xcorr_depthwise(xf[0], zf[0])
        for i in range(len(xf) - 1):
            features_new = self.xcorr_depthwise(xf[i + 1], zf[i + 1])
            features = torch.cat([features, features_new], 1)
        features = self.down(features)

        cls_, loc = self.car_head(features)

        loc = torch.clamp(loc, max=1e4)
        loc = torch.where(torch.isinf(loc), 1e4 * torch.ones_like(loc), loc)

        cls = self.log_softmax(cls_)
        score, boxes = LTRBDecoder(cls_, loc, self.points, self.cfg.TRAIN.SEARCH_SIZE)
        boxes = boxes.permute(0, 2, 3, 1).contiguous()
        bbox = process_box(bbox)
        boxes = process_box(boxes)
        iou, union_area = bbox_iou(bbox, boxes)

        with torch.no_grad():
            # 基于self-training方式，根据pred boxes与GT直接的iou以及boxes的score，重新划分正负样本
            if self.update_settings is not None and not self.validate:
                label_cls = label_update_(label_cls.cpu().detach().numpy(),
                                          score.cpu().detach().numpy(), iou.cpu().detach().numpy(),
                                          positive.cpu().detach().numpy(), **self.update_settings)

            pos_mask = (label_cls == 1.).type(torch.float32)
            neg_mask = (label_cls == 0.).type(torch.float32)
            _pos_mask = (pos_mask == 0.).type(torch.float32)

            # neg_weight = label2weight(neg_mask, avg='sample')
            # pos_weight_cls = label2weight(pos_mask, avg='batch')
            # pos_weight_iou = label2weight(pos_mask, avg='batch')
            # pos_weight_l1 = label2weight(pos_mask, avg='batch')
            if self.train_epoch > 0 and not self.validate:
                iou_weight = sofTmax(-iou * positive[:, None, None], T=0.5, b=-0.20, mask=neg_mask, average='batch')
                if self.train_epoch > 4:
                    neg_weight = sofTmax(score, T=0.5, b=0.25, mask=neg_mask * iou_weight, average='batch')
                else:
                    neg_weight = sofTmax(score, T=0.5, b=0.40, mask=neg_mask * iou_weight, average='batch')
            else:
                neg_weight = label2weight(neg_mask, avg='batch')

            if self.train_epoch > 20 and not self.validate:
                pos_weight_cls = sofTmax(iou, T=0.5, b=0., mask=pos_mask, average='batch')
            else:
                pos_weight_cls = label2weight(pos_mask, avg='batch')
            # pos_weight_cls = label2weight(pos_mask, avg='batch')

            pos_weight_l1 = label2weight(pos_mask, avg='batch')
            pos_weight_iou = pos_weight_l1

        l1_loss = weighted_l1_loss(label_loc, loc, pos_weight_l1, smooth=True)

        # cross entropy loss
        pos_loss, neg_loss = weighted_select_cross_entropy_loss(cls, label_cls,
                                                                pos_weight=pos_weight_cls, neg_weight=neg_weight)
        cls_loss = pos_loss * self.weights['pos_weight'] + neg_loss * self.weights['neg_weight']

        iou_loss = weighted_iou_loss(bbox, boxes, weight=pos_weight_iou,
                                     iou=iou, union_area=union_area, loss_type='ciou')[0]

        loc_loss = iou_loss * self.weights['iou_weight']
        if self.train_epoch > 0 and self.weights['l1_weight']:
            loc_loss += l1_loss * self.weights['l1_weight']

        # a0 = data['label_cls'].numpy()
        # a1 = iou.cpu().detach().numpy()
        # a2 = score.cpu().detach().numpy()
        # a3 = neg_weight.cpu().detach().numpy()
        # import cv2
        # search_img = search.permute((0, 2, 3, 1)).contiguous().type(torch.uint8).cpu().detach().numpy()
        # template_img = template.permute((0, 2, 3, 1)).contiguous().type(torch.uint8).cpu().detach().numpy()
        # j = 2
        # # cv2.imshow('0', template_img[j])
        # # box = list(map(int, data['bbox'].numpy()[j]))
        # # img = search_img[j]
        # # cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        # # cv2.imshow('1', img)
        # # cv2.waitKey()

        # get loss
        outputs = {}
        outputs['total_loss'] = cls_loss + loc_loss
        outputs['cls_loss'] = cls_loss
        outputs['loc_loss'] = loc_loss
        outputs['pos_loss'] = pos_loss
        outputs['neg_loss'] = neg_loss
        outputs['iou_loss'] = iou_loss
        outputs['l1_loss'] = l1_loss
        return outputs


def compute_locations(features, stride, offset):
    h, w = features.size()[-2:]
    locations_per_level = compute_locations_per_level(
        h, w, stride, offset,
        features.device
    )
    return locations_per_level


def compute_locations_per_level(h, w, stride, offset, device):
    shifts_x = torch.arange(
        0, w * stride, step=stride,
        dtype=torch.float32, device=device
    )
    shifts_y = torch.arange(
        0, h * stride, step=stride,
        dtype=torch.float32, device=device
    )
    shift_y, shift_x = torch.meshgrid((shifts_y, shifts_x))
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    locations = torch.stack((shift_x, shift_y), dim=1) + offset  # 32/31?
    return locations


if __name__ == '__main__':
    f = torch.ones((4, 256, 25, 25))
    a = compute_locations(f, 8, 32)
    import torchvision.transforms as trans
    trans.RandomRotation()
    pass
