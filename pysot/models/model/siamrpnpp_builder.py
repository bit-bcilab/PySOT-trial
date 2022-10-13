# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch

from pysot.models.model.base_model_builder import BaseModelBuilder
from pysot.models.head.rpn_head import get_rpn_head
from pysot.models.head.mask_head import get_mask_head, get_refine_head
from pysot.models.loss import select_cross_entropy_loss, weight_l1_loss

from trial.loss import label_update, label2weight, label_update_, sofTmax
from trial.loss import weighted_select_cross_entropy_loss, weighted_iou_loss, weighted_l1_loss
from trial.utils.iou import bbox_iou, process_box
from trial.Decoders import AnchorBasedDecoder


class SiamRPNppBuilder(BaseModelBuilder):
    def __init__(self, cfg):
        super(SiamRPNppBuilder, self).__init__(cfg)

        # build rpn head
        self.rpn_head = get_rpn_head(cfg.RPN.TYPE, **cfg.RPN.KWARGS)

        # build mask head
        if cfg.MASK.MASK:
            self.mask_head = get_mask_head(cfg.MASK.TYPE, **cfg.MASK.KWARGS)

            if cfg.REFINE.REFINE:
                self.refine_head = get_refine_head(cfg.REFINE.TYPE)

        self.T = 0.4
        self.ti = 0.10
        self.ts = 0.35

    def template(self, z):
        zf = self.backbone(z)
        if self.cfg.MASK.MASK:
            zf = zf[-1]
        if self.cfg.ADJUST.ADJUST:
            zf = self.neck(zf)
        self.zf = zf

    def track(self, x):
        xf = self.backbone(x)
        if self.cfg.MASK.MASK:
            self.xf = xf[:-1]
            xf = xf[-1]
        if self.cfg.ADJUST.ADJUST:
            xf = self.neck(xf)
        cls, loc = self.rpn_head(self.zf, xf)
        if self.cfg.MASK.MASK:
            mask, self.mask_corr_feature = self.mask_head(self.zf, xf)
        return {
                'cls': cls,
                'loc': loc,
                'mask': mask if self.cfg.MASK.MASK else None
               }

    def mask_refine(self, pos):
        return self.refine_head(self.xf, self.mask_corr_feature, pos)

    def get_head_parameters(self):
        head_params = [self.rpn_head.parameters()]
        if self.cfg.MASK.MASK:
            head_params.append(self.mask_head.parameters())
            if self.cfg.REFINE.REFINE:
                head_params.append(self.refine_head.parameters())
        return head_params

    def forward_original(self, data):
        """ only used in training
        """
        template = data['template'].cuda()
        search = data['search'].cuda()
        label_cls = data['label_cls'].cuda()
        label_loc = data['label_loc'].cuda()
        label_loc_weight = data['label_loc_weight'].cuda()

        # get feature
        zf = self.backbone(template)
        xf = self.backbone(search)
        if self.cfg.MASK.MASK:
            zf = zf[-1]
            self.xf_refine = xf[:-1]
            xf = xf[-1]
        if self.cfg.ADJUST.ADJUST:
            zf = self.neck(zf)
            xf = self.neck(xf)
        cls_, loc_ = self.rpn_head(zf, xf)

        # get loss
        cls = self.log_softmax(cls_)
        cls_loss = select_cross_entropy_loss(cls, label_cls)
        loc_loss = weight_l1_loss(loc_, label_loc, label_loc_weight)

        # bbox = data['bbox'].cuda()[:, None, None, None, :].type(torch.float32)
        # b, _, sh, sw = loc_.size()
        # loc = loc_.view(b, 4, -1, sh, sw)
        # cls_ = cls_.view(b, 2, -1, sh, sw)
        # score, boxes = AnchorBasedDecoder(cls_, loc, self.anchors)
        # boxes = boxes.permute(0, 2, 3, 4, 1)
        # iou = bbox_iou(bbox, boxes)[0]
        # iou_loss_, iou, _ = weighted_iou_loss(bbox, boxes, label_loc_weight, loss_type='linear_iou')
        # l1_loss = weighted_l1_loss(label_loc, loc, label_loc_weight, smooth=False)
        # a0 = label_cls.cpu().detach().numpy()
        # a1 = iou.cpu().detach().numpy()
        # a2 = score.cpu().detach().numpy()
        # a3 = label_loc_weight.cpu().detach().numpy()

        outputs = {}
        outputs['total_loss'] = self.cfg.TRAIN.CLS_WEIGHT * cls_loss + self.cfg.TRAIN.LOC_WEIGHT * loc_loss
        outputs['cls_loss'] = cls_loss
        outputs['loc_loss'] = loc_loss

        # if self.cfg.MASK.MASK:
        #     # TODO
        #     mask, self.mask_corr_feature = self.mask_head(zf, xf)
        #     mask_loss = None
        #     outputs['total_loss'] += self.cfg.TRAIN.MASK_WEIGHT * mask_loss
        #     outputs['mask_loss'] = mask_loss
        return outputs

    def forward_trial(self, data):
        # return self.forward_trial_old(data)
        # return self.forward_trial_abl(data)
        # return self.forward_trial_focal_loss(data)
        return self.forward_trial_new(data)

    def forward_trial_new(self, data):
        template = data['template'].cuda()
        search = data['search'].cuda()
        label_cls = data['label_cls'].cuda()
        label_loc = data['label_loc'].cuda()
        bbox = data['bbox'].cuda()[:, None, None, None, :]
        positive = data['pos'].cuda()

        # get feature
        zf = self.backbone(template)
        xf = self.backbone(search)
        if self.cfg.MASK.MASK:
            zf = zf[-1]
            self.xf_refine = xf[:-1]
            xf = xf[-1]
        if self.cfg.ADJUST.ADJUST:
            zf = self.neck(zf)
            xf = self.neck(xf)
        cls_, loc_ = self.rpn_head(zf, xf)

        cls = self.log_softmax(cls_)
        b, _, sh, sw = loc_.size()
        loc = loc_.view(b, 4, -1, sh, sw)
        cls_ = cls_.view(b, 2, -1, sh, sw)
        score, boxes = AnchorBasedDecoder(cls_, loc, self.anchors, self.cfg.TRAIN.SEARCH_SIZE)
        boxes = boxes.permute(0, 2, 3, 4, 1).contiguous()
        bbox = process_box(bbox)
        boxes = process_box(boxes)
        iou, union_area = bbox_iou(bbox, boxes)

        with torch.no_grad():
            # 基于self-training方式，根据pred boxes与GT直接的iou以及boxes的score，重新划分正负样本
            if self.update_settings is not None and not self.validate:
                label_cls = label_update_(label_cls.cpu().detach().numpy(),
                                          score.cpu().detach().numpy(), iou.cpu().detach().numpy(),
                                          positive.cpu().detach().numpy(), base=self.cfg.BASE, **self.update_settings)
            pos_mask = (label_cls == 1.).type(torch.float32)
            neg_mask = (label_cls == 0.).type(torch.float32)

            if 'weighted' in self.cfg.MODE and self.train_epoch > 0 and not self.validate:
                iou_weight = sofTmax(-iou * positive[:, None, None, None], T=self.T, b=-self.ti, mask=neg_mask, average='batch')
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
        bbox = data['bbox'].cuda()[:, None, None, None, :]
        positive = data['pos'].cuda()

        # get feature
        zf = self.backbone(template)
        xf = self.backbone(search)
        if self.cfg.MASK.MASK:
            zf = zf[-1]
            self.xf_refine = xf[:-1]
            xf = xf[-1]
        if self.cfg.ADJUST.ADJUST:
            zf = self.neck(zf)
            xf = self.neck(xf)
        cls_, loc_ = self.rpn_head(zf, xf)

        cls = self.log_softmax(cls_)
        b, _, sh, sw = loc_.size()
        loc = loc_.view(b, 4, -1, sh, sw)
        cls_ = cls_.view(b, 2, -1, sh, sw)
        score, boxes = AnchorBasedDecoder(cls_, loc, self.anchors, self.cfg.TRAIN.SEARCH_SIZE)
        boxes = boxes.permute(0, 2, 3, 4, 1).contiguous()
        bbox = process_box(bbox)
        boxes = process_box(boxes)
        iou, union_area = bbox_iou(bbox, boxes)

        with torch.no_grad():
            # 基于self-training方式，根据pred boxes与GT直接的iou以及boxes的score，划分正负样本
            if self.update_settings is not None and not self.validate:
                label_cls = label_update_(label_cls.cpu().detach().numpy(), score.cpu().detach().numpy(),
                                          iou.cpu().detach().numpy(), positive.cpu().detach().numpy(),
                                          base=self.cfg.BASE, **self.update_settings)
            pos_mask = (label_cls == 1.).type(torch.float32)
            neg_mask = (label_cls == 0.).type(torch.float32)
            _pos_mask = (pos_mask == 0.).type(torch.float32)

            if self.train_epoch > 0 and not self.validate:
                iou_weight = sofTmax(-iou * positive[:, None, None, None], T=0.4, b=-0.15, mask=neg_mask, average='batch')
                if self.train_epoch > 4:
                    neg_weight = sofTmax(score, T=0.4, b=0.25, mask=neg_mask * iou_weight, average='batch')
                else:
                    neg_weight = sofTmax(score, T=0.4, b=0.40, mask=neg_mask * iou_weight, average='batch')
            else:
                neg_weight = label2weight(neg_mask, avg='batch')

            if self.train_epoch > 0 and not self.validate:
                pos_weight_cls = sofTmax(iou, T=0.4, b=0., mask=pos_mask, average='batch')
            else:
                pos_weight_cls = label2weight(pos_mask, avg='batch')

            pos_weight_l1 = label2weight(pos_mask, avg='batch')
            pos_weight_iou = pos_weight_l1

        l1_loss = weighted_l1_loss(label_loc, loc, pos_weight_l1, smooth=False)

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

        outputs = {}
        outputs['total_loss'] = cls_loss + loc_loss
        outputs['cls_loss'] = cls_loss
        outputs['pos_loss'] = pos_loss
        outputs['neg_loss'] = neg_loss
        outputs['loc_loss'] = loc_loss
        outputs['iou_loss'] = iou_loss
        outputs['l1_loss'] = l1_loss
        return outputs
