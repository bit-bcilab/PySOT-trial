# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import torch

from pysot.models.model.base_model_builder import BaseModelBuilder
from pysot.models.head.rpn_head import get_rpn_head
from pysot.models.head.mask_head import FusedSemanticHead
from pysot.models.head.siamatt_head import FeatureEnhance, FeatureFusionNeck, FCx2DetHead
from pysot.models.loss import select_cross_entropy_loss, weight_l1_loss, mask_loss_bce, det_loss_smooth_l1
from pysot.utils.mask_target_builder import build_proposal_target_, build_mask_target_, convert_loc_to_bbox_

from trial.loss import label_update, label2weight
from trial.loss import weighted_select_cross_entropy_loss, weighted_iou_loss, weighted_l1_loss
from trial.utils.iou import bbox_iou
from trial.Decoders import AnchorBasedDecoder


class SiamAttBuilder(BaseModelBuilder):
    def __init__(self, cfg):
        super(SiamAttBuilder, self).__init__(cfg)

        # build rpn head
        self.rpn_head = get_rpn_head(cfg.RPN.TYPE, **cfg.RPN.KWARGS)
        self.feature_enhance = FeatureEnhance(in_channels=256, out_channels=256)

        # build mask head
        if cfg.MASK.MASK:
            self.feature_fusion = FeatureFusionNeck(num_ins=5, fusion_level=1,
                                                    in_channels=[64, 256, 256, 256, 256], conv_out_channels=256)
            self.mask_head = FusedSemanticHead(cfg=cfg, pooling_func=None,
                                               num_convs=4, in_channels=256,
                                               upsample_ratio=(cfg.MASK.MASK_OUTSIZE // cfg.TRAIN.ROIPOOL_OUTSIZE))
            self.bbox_head = FCx2DetHead(cfg=cfg, pooling_func=None,
                                         in_channels=256 * (cfg.TRAIN.ROIPOOL_OUTSIZE // 4)**2)

    def template(self, z):
        zf = self.backbone(z)
        if self.cfg.ADJUST.ADJUST:
            zf[2:] = self.neck(zf[2:])
        self.zf = zf

    def track(self, x):
        xf = self.backbone(x)
        if self.cfg.ADJUST.ADJUST:
            xf[2:] = self.neck(xf[2:])

        zf, xf[2:] = self.feature_enhance(self.zf[2:], xf[2:])
        cls, loc = self.rpn_head(zf, xf[2:])
        enhanced_zf = self.zf[:2] + zf
        if self.cfg.MASK.MASK:
            self.b_fused_features, self.m_fused_features = self.feature_fusion(enhanced_zf, xf)
        return {
            'cls': cls,
            'loc': loc
        }

    def mask_refine(self, roi):
        mask_pred = self.mask_head(self.m_fused_features, roi)
        return mask_pred

    def bbox_refine(self, roi):
        bbox_pred = self.bbox_head(self.b_fused_features, roi)
        return bbox_pred

    def get_head_parameters(self):
        head_params = [self.rpn_head.parameters()]
        head_params.append(self.feature_enhance.parameters())
        if self.cfg.MASK.MASK:
            head_params.append(self.feature_fusion.parameters())
            head_params.append(self.mask_head.parameters())
            head_params.append(self.bbox_head.parameters())
        return head_params

    def forward_trial(self, data):
        # loss of Siamese network
        template = data['template'].cuda()
        search = data['search'].cuda()
        label_cls = data['label_cls'].cuda()
        label_loc = data['label_loc'].cuda()
        bbox = data['bbox'].cuda()[:, None, None, None, :]
        positive = data['pos'].cuda()

        # get feature
        zf = self.backbone(template)
        xf = self.backbone(search)
        if self.cfg.ADJUST.ADJUST:
            zf[2:] = self.neck(zf[2:])
            xf[2:] = self.neck(xf[2:])
        zf[2:], xf[2:] = self.feature_enhance(zf[2:], xf[2:])
        cls_, loc_ = self.rpn_head(zf[2:], xf[2:])

        cls = self.log_softmax(cls_)
        b, _, sh, sw = loc_.size()
        loc = loc_.view(b, 4, -1, sh, sw)
        cls_ = cls_.view(b, 2, -1, sh, sw)
        score, boxes = AnchorBasedDecoder(cls_, loc, self.anchors)
        boxes = boxes.permute(0, 2, 3, 4, 1)

        # get loss
        with torch.no_grad():
            # 基于self-training方式，根据pred boxes与GT直接的iou以及boxes的score，划分正负样本
            if self.update_settings is not None:
                iou = bbox_iou(bbox, boxes)[0]
                label_cls = label_update(label_cls.cpu().detach().numpy(), score.cpu().detach().numpy(),
                                         iou.cpu().detach().numpy(), positive.cpu().detach().numpy(),
                                         base=self.cfg.BASE, **self.update_settings)
            pos_mask = torch.eq(label_cls, torch.ones_like(label_cls)).type(torch.float32)
            neg_mask = torch.eq(label_cls, torch.zeros_like(label_cls)).type(torch.float32)
            pos_weight = label2weight(pos_mask, avg='batch')
            neg_weight = label2weight(neg_mask, avg='sample')

        iou_loss, iou, _ = weighted_iou_loss(bbox, boxes, pos_weight, loss_type='linear_iou')
        l1_loss = weighted_l1_loss(label_loc, loc, pos_weight, smooth=False)
        loc_loss = 0.5 * iou_loss + 0.5 * l1_loss

        # 以IOU值作为正类损失的权重
        with torch.no_grad():
            pw = torch.exp(10. * (iou - 0.75)) * pos_mask
            dim = [i for i in range(1, pw.ndim)]
            pw = pw / (pw.sum(dim=dim, keepdim=True) + 1e-9)

        # cross entropy loss
        pos_loss, neg_loss = weighted_select_cross_entropy_loss(cls, label_cls, pos_weight=pw, neg_weight=neg_weight)
        cls_loss = pos_loss * 0.5 + neg_loss * 0.5

        # pos_loss_, neg_loss_ = select_cross_entropy_loss(cls, label_cls, add=False)
        # cls_loss_ = pos_loss_ * 0.5 + neg_loss_ * 0.5
        # loc_loss_ = weight_l1_loss(loc_, label_loc, pos_weight)
        # import cv2
        # search_img = search.permute((0, 2, 3, 1)).type(torch.uint8).cpu().detach().numpy()
        # template_img = template.permute((0, 2, 3, 1)).type(torch.uint8).cpu().detach().numpy()
        # a0 = label_cls.cpu().detach().numpy()
        # a1 = iou.cpu().detach().numpy()
        # a2 = score.cpu().detach().numpy()
        # a3 = data['label_cls'].numpy()
        # a4 = pw.cpu().detach().numpy()
        # # j = 2
        # # cv2.imshow('0', template_img[j])
        # # box = list(map(int, data['bbox'].numpy()[j]))
        # # cv2.rectangle(search_img[j], (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        # # cv2.imshow('1', search_img[j])
        # # cv2.waitKey()

        outputs = {}
        outputs['total_loss'] = self.cfg.TRAIN.CLS_WEIGHT * cls_loss + self.cfg.TRAIN.LOC_WEIGHT * loc_loss
        outputs['cls_loss'] = cls_loss
        outputs['loc_loss'] = loc_loss

        # if self.cfg.MASK.MASK:
        #     # loss of box refinement
        #     gt_bboxes = data['bbox'].cuda()
        #     bbox_weight = data['bbox_weight'].cuda()
        #
        #     # Convert loc coordinate to (x1,y1,x2,y2)
        #     loc = loc.detach()
        #     bbox = convert_loc_to_bbox_(self.cfg, loc)
        #
        #     rois, cls_ind, regression_target = build_proposal_target_(self.cfg, bbox, gt_bboxes)
        #     select_roi_list = build_roi(rois, cls_ind)
        #
        #     # for deformable roi pooling
        #     batch_inds = torch.from_numpy(np.arange(self.cfg.TRAIN.BATCH_SIZE).repeat(self.cfg.TRAIN.ROI_PER_IMG).
        #         reshape(self.cfg.TRAIN.BATCH_SIZE*self.cfg.TRAIN.ROI_PER_IMG, 1)).cuda().float()
        #     rois = torch.cat((batch_inds, torch.stack(select_roi_list).view(-1, 4)), dim=1)
        #
        #     b_fused_features, m_fused_features = self.feature_fusion(zf, xf)
        #     bbox_pred = self.bbox_head(b_fused_features, rois)
        #     bbox_pred = bbox_pred.view_as(regression_target)
        #     bbox_loss = det_loss_smooth_l1(bbox_pred, regression_target, bbox_weight)
        #     outputs['bbox_loss'] = bbox_loss
        #     outputs['total_loss'] += self.cfg.TRAIN.BBOX_WEIGHT * bbox_loss
        #
        #     # loss of mask prediction
        #     search_mask = data['search_mask'].cuda()
        #     mask_weight = data['mask_weight'].cuda()
        #
        #     mask_targets, select_roi_list = build_mask_target_(self.cfg, rois, cls_ind, search_mask)
        #     mask_pred = self.mask_head(m_fused_features, rois)
        #     mask_pred = mask_pred.view_as(mask_targets)
        #     mask_loss, iou_m, iou_5, iou_7 = mask_loss_bce(mask_pred, mask_targets, mask_weight)
        #
        #     outputs['mask_loss'] = mask_loss
        #     outputs['total_loss'] += self.cfg.TRAIN.MASK_WEIGHT * mask_loss
        #
        #     outputs['mask_labels'] = mask_targets
        #     outputs['mask_preds'] = mask_pred
        #     outputs['iou_m'] = iou_m
        #     outputs['iou_5'] = iou_5
        #     outputs['iou_7'] = iou_7
        return outputs

    def forward_original(self, data):
        # loss of Siamese network
        template = data['template'].cuda()
        search = data['search'].cuda()
        label_cls = data['label_cls'].cuda()
        label_loc = data['label_loc'].cuda()
        label_loc_weight = data['label_loc_weight'].cuda()

        # get feature
        zf = self.backbone(template)
        xf = self.backbone(search)
        if self.cfg.ADJUST.ADJUST:
            zf[2:] = self.neck(zf[2:])
            xf[2:] = self.neck(xf[2:])
        zf[2:], xf[2:] = self.feature_enhance(zf[2:], xf[2:])
        cls, loc = self.rpn_head(zf[2:], xf[2:])

        # get loss
        cls = self.log_softmax(cls)
        cls_loss = select_cross_entropy_loss(cls, label_cls)
        loc_loss = weight_l1_loss(loc, label_loc, label_loc_weight)

        outputs = {}
        outputs['total_loss'] = self.cfg.TRAIN.CLS_WEIGHT * cls_loss + self.cfg.TRAIN.LOC_WEIGHT * loc_loss
        outputs['cls_loss'] = cls_loss
        outputs['loc_loss'] = loc_loss

        # if self.cfg.MASK.MASK:
        #     # loss of box refinement
        #     gt_bboxes = data['bbox'].cuda()
        #     bbox_weight = data['bbox_weight'].cuda()
        #
        #     # Convert loc coordinate to (x1,y1,x2,y2)
        #     loc = loc.detach()
        #     bbox = convert_loc_to_bbox_(self.cfg, loc)
        #
        #     rois, cls_ind, regression_target = build_proposal_target_(self.cfg, bbox, gt_bboxes)
        #     select_roi_list = build_roi(rois, cls_ind)
        #
        #     # for deformable roi pooling
        #     batch_inds = torch.from_numpy(np.arange(self.cfg.TRAIN.BATCH_SIZE).repeat(self.cfg.TRAIN.ROI_PER_IMG).
        #         reshape(self.cfg.TRAIN.BATCH_SIZE*self.cfg.TRAIN.ROI_PER_IMG, 1)).cuda().float()
        #     rois = torch.cat((batch_inds, torch.stack(select_roi_list).view(-1, 4)), dim=1)
        #
        #     b_fused_features, m_fused_features = self.feature_fusion(zf, xf)
        #     bbox_pred = self.bbox_head(b_fused_features, rois)
        #     bbox_pred = bbox_pred.view_as(regression_target)
        #     bbox_loss = det_loss_smooth_l1(bbox_pred, regression_target, bbox_weight)
        #     outputs['bbox_loss'] = bbox_loss
        #     outputs['total_loss'] += self.cfg.TRAIN.BBOX_WEIGHT * bbox_loss
        #
        #     # loss of mask prediction
        #     search_mask = data['search_mask'].cuda()
        #     mask_weight = data['mask_weight'].cuda()
        #
        #     mask_targets, select_roi_list = build_mask_target_(self.cfg, rois, cls_ind, search_mask)
        #     mask_pred = self.mask_head(m_fused_features, rois)
        #     mask_pred = mask_pred.view_as(mask_targets)
        #     mask_loss, iou_m, iou_5, iou_7 = mask_loss_bce(mask_pred, mask_targets, mask_weight)
        #
        #     outputs['mask_loss'] = mask_loss
        #     outputs['total_loss'] += self.cfg.TRAIN.MASK_WEIGHT * mask_loss
        #
        #     outputs['mask_labels'] = mask_targets
        #     outputs['mask_preds'] = mask_pred
        #     outputs['iou_m'] = iou_m
        #     outputs['iou_5'] = iou_5
        #     outputs['iou_7'] = iou_7
        return outputs


def build_roi(boxes, label):
    num_batches = boxes.shape[0]
    select_rois = []
    for i in range(num_batches):
        batch_label = label[i]
        num_rois = boxes[i].shape[0]
        batch_select_rois = []
        for j in range(num_rois):
            if batch_label[j] <= 0:
                continue
            roi = boxes[i, j, :]
            batch_select_rois.append(roi)
        batch_select_rois = torch.stack(batch_select_rois, dim=0)
        select_rois.append(batch_select_rois)
    return select_rois
