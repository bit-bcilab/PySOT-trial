# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch

from pysot.models.model.base_model_builder import BaseModelBuilder
from pysot.models.head.siamban_head import get_ban_head
from pysot.models.loss import select_cross_entropy_loss, select_iou_loss
from trial.loss import sofTmax, label2weight, label_update_
from trial.loss import weighted_select_cross_entropy_loss, weighted_iou_loss, weighted_l1_loss, weighted_focal_loss
from trial.utils.iou import process_box, bbox_iou
from trial.Decoders import LTRBDecoder


class SiamBANBuilder(BaseModelBuilder):
    def __init__(self, cfg):
        super(SiamBANBuilder, self).__init__(cfg)
        self.head = get_ban_head(cfg.BAN.TYPE, **cfg.BAN.KWARGS)

        self.T = 0.4
        self.ti = 0.3
        self.ts = 0.3

        # if self.cfg.MODE == 'teacher':
        #     pass
        # if self.cfg.MODE == 'student':
        #     adaption_s = ...
        #     adaption_t = ...
        #     pass

    def template(self, z):
        zf = self.backbone(z)
        if self.cfg.ADJUST.ADJUST:
            zf = self.neck(zf)
        self.zf = zf

    def track(self, x):
        xf = self.backbone(x)
        if self.cfg.ADJUST.ADJUST:
            xf = self.neck(xf)
        cls, loc = self.head(self.zf, xf)
        return {
                'cls': cls,
                'loc': loc
               }

    def get_head_parameters(self):
        head_params = [self.head.parameters()]
        return head_params

    def forward_original(self, data):
        template = data['template'].cuda()
        search = data['search'].cuda()
        label_cls = data['label_cls'].cuda()
        label_loc = data['label_loc'].cuda()

        # get feature
        zf = self.backbone(template)
        xf = self.backbone(search)
        if self.cfg.ADJUST.ADJUST:
            zf = self.neck(zf)
            xf = self.neck(xf)
        cls_, loc = self.head(zf, xf)

        # get loss

        # cls loss with cross entropy loss
        cls = self.log_softmax(cls_)
        cls_loss = select_cross_entropy_loss(cls, label_cls)

        # loc loss with iou loss
        loc_loss = select_iou_loss(loc, label_loc, label_cls)

        outputs = {}
        outputs['total_loss'] = self.cfg.TRAIN.CLS_WEIGHT * cls_loss + self.cfg.TRAIN.LOC_WEIGHT * loc_loss
        outputs['cls_loss'] = cls_loss
        outputs['loc_loss'] = loc_loss

        # score, boxes = LTRBDecoder(cls_, loc, self.points)
        # boxes = boxes.permute(0, 2, 3, 1)
        # bbox = data['bbox'].cuda()[:, None, None, :]
        # iou = bbox_iou(bbox, boxes)[0]
        # a1 = iou.cpu().detach().numpy()
        # a2 = score.cpu().detach().numpy()
        # import cv2
        # search_img = search.permute((0, 2, 3, 1)).type(torch.uint8).cpu().detach().numpy()
        # template_img = template.permute((0, 2, 3, 1)).type(torch.uint8).cpu().detach().numpy()
        # # j = 2
        # # cv2.imshow('0', template_img[j])
        # # box = list(map(int, data['bbox'].numpy()[j]))
        # # cv2.rectangle(search_img[j], (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        # # cv2.imshow('1', search_img[j])
        # # cv2.waitKey()
        return outputs

    def forward_trial(self, data):
        # return self.forward_trial_focal_loss(data)
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

        cls_, loc = self.head(zf, xf)

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

    def forward_trial_focal_loss(self, data):
        template = data['template'].cuda()
        search = data['search'].cuda()
        label_cls = data['label_cls'].cuda()
        label_loc = data['label_loc'].cuda()
        bbox = data['bbox'].cuda()[:, None, None, :]

        # get feature
        zf = self.backbone(template)
        xf = self.backbone(search)
        if self.cfg.ADJUST.ADJUST:
            zf = self.neck(zf)
            xf = self.neck(xf)

        cls_, loc = self.head(zf, xf)

        loc = torch.clamp(loc, max=1e4)
        loc = torch.where(torch.isinf(loc), 1e4 * torch.ones_like(loc), loc)

        cls, cls0 = self.log_softmax(cls_, return_pred=True)
        score, boxes = LTRBDecoder(cls_, loc, self.points, self.cfg.TRAIN.SEARCH_SIZE)
        boxes = boxes.permute(0, 2, 3, 1).contiguous()
        bbox = process_box(bbox)
        boxes = process_box(boxes)
        iou, union_area = bbox_iou(bbox, boxes)

        with torch.no_grad():
            pos_mask = (label_cls == 1.).type(torch.float32)
            neg_mask = (label_cls == 0.).type(torch.float32)

            neg_weight = label2weight(neg_mask, avg='batch')
            pos_weight_cls = label2weight(pos_mask, avg='batch')
            pos_weight_iou = pos_weight_cls
            pos_weight_l1 = pos_weight_iou

        # cross entropy loss
        pos_loss, neg_loss = weighted_focal_loss(pred_log=cls, pred=cls0,
                                                 pos_weight=pos_weight_cls, neg_weight=neg_weight,
                                                 alpha=0.25, gamma=2.)
        # pos_loss_ce, neg_loss_ce = weighted_select_cross_entropy_loss(cls, label_cls,
        #                                                               pos_weight=pos_weight_cls, neg_weight=neg_weight)
        cls_loss = pos_loss * self.weights['pos_weight'] + neg_loss * self.weights['neg_weight']

        l1_loss = weighted_l1_loss(label_loc, loc, pos_weight_l1, smooth=True)
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
        # cv2.imshow('0', template_img[j])
        # box = list(map(int, data['bbox'].numpy()[j]))
        # img = search_img[j]
        # cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        # cv2.imshow('1', img)
        # cv2.waitKey()

        outputs = {}
        outputs['total_loss'] = cls_loss + loc_loss
        outputs['cls_loss'] = cls_loss
        outputs['pos_loss'] = pos_loss
        outputs['neg_loss'] = neg_loss
        # outputs['pos_loss_ce'] = pos_loss_ce
        # outputs['neg_loss_ce'] = neg_loss_ce
        outputs['loc_loss'] = loc_loss
        outputs['iou_loss'] = iou_loss
        outputs['l1_loss'] = l1_loss
        return outputs

    def forward_teacher(self, data):
        pass

    def forward_student(self, data):
        pass
