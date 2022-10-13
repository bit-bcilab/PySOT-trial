# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn

from pysot.models.model.base_model_builder import BaseModelBuilder
from pysot.models.xcorr import xcorr_depthwise
from pysot.models.head.siamcar_head import CARHead
from pysot.models.loss import SiamCARLossComputation
from trial.loss import centerness_target, sofTmax, label2weight, label_update_
from trial.loss import weighted_select_cross_entropy_loss, weighted_iou_loss, weighted_l1_loss, weighted_bce_loogits
from trial.utils.iou import process_box, bbox_iou
from trial.Decoders import LTRBDecoder


class SiamCARBuilder(BaseModelBuilder):
    def __init__(self, cfg):
        super(SiamCARBuilder, self).__init__(cfg)

        # build car head
        self.car_head = CARHead(cfg, 256)

        # build response map
        self.xcorr_depthwise = xcorr_depthwise

        # build loss
        self.loss_evaluator = SiamCARLossComputation(cfg)

        self.down = nn.ConvTranspose2d(256 * 3, 256, 1, 1)

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

        cls, loc, cen = self.car_head(features)
        return {
            'cls': cls,
            'loc': loc,
            'cen': cen
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

    # def forward_trial(self, data):
    #     """ only used in training
    #     """
    #     template = data['template'].cuda()
    #     search = data['search'].cuda()
    #     bbox_mask = data['bbox_mask'].cuda()
    #     label_cls = data['label_cls'].cuda()
    #     label_loc = data['label_loc'].cuda()
    #     bbox = data['bbox'].cuda()[:, None, None, :]
    #     positive = data['pos'].cuda()
    #     centerness = centerness_target(label_loc)
    #
    #     # get feature
    #     zf = self.backbone(template)
    #     xf = self.backbone(search)
    #     if self.cfg.ADJUST.ADJUST:
    #         zf = self.neck(zf)
    #         xf = self.neck(xf)
    #
    #     features = self.xcorr_depthwise(xf[0], zf[0])
    #     for i in range(len(xf) - 1):
    #         features_new = self.xcorr_depthwise(xf[i + 1], zf[i + 1])
    #         features = torch.cat([features, features_new], 1)
    #     features = self.down(features)
    #
    #     cls_, loc, cen = self.car_head(features)
    #
    #     cls = self.log_softmax(cls_)
    #     score, boxes = LTRBDecoder(cls_, loc, self.points)
    #     boxes = boxes.permute(0, 2, 3, 1).contiguous()
    #     bbox = process_box(bbox)
    #     boxes = process_box(boxes)
    #     iou, union_area = bbox_iou(bbox, boxes)
    #
    #     with torch.no_grad():
    #         # 基于self-training方式，根据pred boxes与GT直接的iou以及boxes的score，重新划分正负样本
    #         if self.update_settings is not None:
    #             # label_cls = label_update(label_cls.cpu().detach().numpy(), score.cpu().detach().numpy(),
    #             #                          iou.cpu().detach().numpy(), positive.cpu().detach().numpy(),
    #             #                          base=self.cfg.BASE, **self.update_settings)
    #             label_cls = label_update_point(label_cls.cpu().detach().numpy(), bbox_mask.cpu().detach().numpy(),
    #                                            score.cpu().detach().numpy(), iou.cpu().detach().numpy(),
    #                                            positive.cpu().detach().numpy(), **self.update_settings)
    #
    #         pos_mask = torch.eq(label_cls, torch.ones_like(label_cls)).type(torch.float32)
    #         neg_mask = torch.eq(label_cls, torch.zeros_like(label_cls)).type(torch.float32)
    #         _pos_mask = torch.eq(pos_mask, torch.zeros_like(label_cls)).type(torch.float32)
    #         loc_weight = pos_mask + _pos_mask * iou
    #
    #         neg_weight = sofTmax(score, T=0.2, b=0.2, mask=neg_mask, average='sample')
    #         pos_weight_cls = sofTmax(iou, T=0.25, b=0.7, mask=pos_mask, average='batch')
    #         pos_weight_loc = sofTmax(loc_weight, T=0.10, b=0., mask=bbox_mask, average='batch')
    #
    #         l1_loss = weighted_l1_loss(label_loc, loc, pos_mask, smooth=True)
    #
    #     # cross entropy loss
    #     pos_loss, neg_loss = weighted_select_cross_entropy_loss(cls, label_cls,
    #                                                             pos_weight=pos_weight_cls, neg_weight=neg_weight)
    #     cls_loss = pos_loss * 0.5 + neg_loss * 0.5
    #
    #     iou_loss = weighted_iou_loss(bbox, boxes, weight=pos_weight_loc,
    #                                  iou=iou, union_area=union_area, loss_type='ciou')[0]
    #
    #     cen_loss = weighted_bce_loogits(centerness, cen[:, 0, ...], pos_mask)
    #
    #     # loc_loss = 0.5 * iou_loss + 0.01 * l1_loss
    #     loc_loss = iou_loss
    #
    #     import cv2
    #     search_img = search.permute((0, 2, 3, 1)).contiguous().type(torch.uint8).cpu().detach().numpy()
    #     template_img = template.permute((0, 2, 3, 1)).contiguous().type(torch.uint8).cpu().detach().numpy()
    #     a0 = label_cls.cpu().detach().numpy()
    #     a1 = iou.cpu().detach().numpy()
    #     a2 = score.cpu().detach().numpy()
    #     a3 = bbox_mask.cpu().detach().numpy()
    #     # j = 2
    #     # cv2.imshow('0', template_img[j])
    #     # box = list(map(int, data['bbox'].numpy()[j]))
    #     # img = search_img[j]
    #     # cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
    #     # cv2.imshow('1', img)
    #     # cv2.waitKey()
    #
    #     # get loss
    #     outputs = {}
    #     outputs['total_loss'] = self.cfg.TRAIN.CLS_WEIGHT * cls_loss + \
    #                             self.cfg.TRAIN.LOC_WEIGHT * loc_loss + self.cfg.TRAIN.CEN_WEIGHT * cen_loss
    #     outputs['cls_loss'] = cls_loss
    #     outputs['loc_loss'] = loc_loss
    #     outputs['cen_loss'] = cen_loss
    #     outputs['pos_loss'] = pos_loss
    #     outputs['neg_loss'] = neg_loss
    #     outputs['iou_loss'] = iou_loss
    #     outputs['l1_loss'] = l1_loss
    #     return outputs

    def forward_trial(self, data):
        """ only used in training
        """
        template = data['template'].cuda()
        search = data['search'].cuda()
        label_cls = data['label_cls'].cuda()
        label_loc = data['label_loc'].cuda()
        bbox = data['bbox'].cuda()[:, None, None, :]
        positive = data['pos'].cuda()
        centerness, center_mask = centerness_target(label_loc)

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
                iou_weight = sofTmax(-iou * positive[:, None, None], T=0.5, b=-0.15, mask=neg_mask, average='batch')
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

        cen_loss = weighted_bce_loogits(centerness, cen[:, 0, ...], pos_mask * center_mask)

        # a0 = data['label_cls'].numpy()
        # a1 = iou.cpu().detach().numpy()
        # a2 = score.cpu().detach().numpy()
        # a3 = neg_weight.cpu().detach().numpy()
        # a4 = cen.cpu().detach().numpy()[:, 0, ...]
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
        outputs['total_loss'] = cls_loss + loc_loss + self.weights['cen_weight'] * cen_loss
        outputs['cls_loss'] = cls_loss
        outputs['loc_loss'] = loc_loss
        outputs['cen_loss'] = cen_loss
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
