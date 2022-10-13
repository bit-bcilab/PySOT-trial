# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import torch
import torch.nn as nn
import torch.nn.functional as F

from pysot.models.model.base_model_builder import BaseModelBuilder
from pysot.models.head.siamcar_head import CARHead
from pysot.models.loss import SiamGATLossComputation
from trial.loss import centerness_target, label2weight, label_update
from trial.loss import weighted_select_cross_entropy_loss, weighted_iou_loss, weighted_l1_loss, weighted_bce_loogits
from trial.utils.iou import bbox_iou
from trial.Decoders import LTRBDecoder


class Graph_Attention_Union(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Graph_Attention_Union, self).__init__()

        # search region nodes linear transformation
        self.support = nn.Conv2d(in_channel, in_channel, 1, 1)

        # target template nodes linear transformation
        self.query = nn.Conv2d(in_channel, in_channel, 1, 1)

        # linear transformation for message passing
        self.g = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 1, 1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True),
        )

        # aggregated feature
        self.fi = nn.Sequential(
            nn.Conv2d(in_channel*2, out_channel, 1, 1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )

    def forward(self, zf, xf):
        # linear transformation
        xf_trans = self.query(xf)
        zf_trans = self.support(zf)

        # linear transformation for message passing
        xf_g = self.g(xf)
        zf_g = self.g(zf)

        # calculate similarity
        shape_x = xf_trans.shape
        shape_z = zf_trans.shape

        zf_trans_plain = zf_trans.view(-1, shape_z[1], shape_z[2] * shape_z[3])
        zf_g_plain = zf_g.view(-1, shape_z[1], shape_z[2] * shape_z[3]).permute(0, 2, 1)
        xf_trans_plain = xf_trans.view(-1, shape_x[1], shape_x[2] * shape_x[3]).permute(0, 2, 1)

        similar = torch.matmul(xf_trans_plain, zf_trans_plain)
        similar = F.softmax(similar, dim=2)

        embedding = torch.matmul(similar, zf_g_plain).permute(0, 2, 1)
        embedding = embedding.view(-1, shape_x[1], shape_x[2], shape_x[3])

        # aggregated feature
        output = torch.cat([embedding, xf_g], 1)
        output = self.fi(output)
        return output


class SiamGATBuilder(BaseModelBuilder):
    def __init__(self, cfg):
        super(SiamGATBuilder, self).__init__(cfg)

        # build car head
        self.car_head = CARHead(cfg, 256)

        # build response map
        self.attention = Graph_Attention_Union(256, 256)

        # build loss
        self.loss_evaluator = SiamGATLossComputation(cfg)

    def get_head_parameters(self):
        head_params = [self.car_head.parameters(), self.attention.parameters()]
        return head_params

    def template(self, z, roi):
        self.zf = self.backbone(z, roi)

    def track(self, x):
        xf = self.backbone(x)

        features = self.attention(self.zf, xf)

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
        target_box = data['target_box'].cuda()
        neg = data['neg'].cuda()

        # get feature
        zf = self.backbone(template, target_box)
        xf = self.backbone(search)
        features = self.attention(zf, xf)
        cls, loc, cen = self.car_head(features)

        # score = cls.softmax(dim=1)[:, 1, ...]
        # a2 = score.cpu().detach().numpy()

        locations = compute_locations(cls, self.cfg.TRACK.STRIDE, self.cfg.TRACK.OFFSET)
        cls = self.log_softmax(cls)
        cls_loss, loc_loss, cen_loss = self.loss_evaluator(
            locations,
            cls,
            loc,
            cen, label_cls, label_loc, neg
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
        """ only used in training
        """
        template = data['template'].cuda()
        search = data['search'].cuda()
        label_cls = data['label_cls'].cuda()
        label_loc = data['label_loc'].cuda()
        z_box = data['z_box'].cuda()
        bbox = data['bbox'].cuda()[:, None, None, :]
        positive = data['pos'].cuda()
        centerness, center_mask = centerness_target(label_loc)

        # get feature
        zf = self.backbone(template, z_box)
        xf = self.backbone(search)
        features = self.attention(zf, xf)
        cls_, loc, cen = self.car_head(features)

        cls = self.log_softmax(cls_)
        score, boxes = LTRBDecoder(cls_, loc, self.points)
        boxes = boxes.permute(0, 2, 3, 1)

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

        # get loss
        iou_loss, iou, _ = weighted_iou_loss(bbox, boxes, pos_weight, loss_type='ciou')
        l1_loss = weighted_l1_loss(label_loc, loc, pos_weight, smooth=True)
        # loc_loss = 0.5 * iou_loss + 0.01 * l1_loss
        loc_loss = iou_loss

        # 以IOU值作为正类损失的权重
        with torch.no_grad():
            pw = torch.exp(10. * (iou - 0.75)) * pos_mask
            dim = [i for i in range(1, pw.ndim)]
            pw = pw / (pw.sum(dim=dim, keepdim=True) + 1e-9)

        # cross entropy loss
        pos_loss, neg_loss = weighted_select_cross_entropy_loss(cls, label_cls, pos_weight=pw, neg_weight=neg_weight)
        cls_loss = pos_loss * 0.5 + neg_loss * 0.5
        cen_loss = weighted_bce_loogits(centerness, cen[:, 0, ...], pos_weight * center_mask)

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

        # get loss
        outputs = {}
        outputs['total_loss'] = self.cfg.TRAIN.CLS_WEIGHT * cls_loss + \
                                self.cfg.TRAIN.LOC_WEIGHT * loc_loss + self.cfg.TRAIN.CEN_WEIGHT * cen_loss
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
    locations = torch.stack((shift_x, shift_y), dim=1) + offset
    return locations


if __name__ == '__main__':
    f = torch.ones((4, 256, 25, 25))
    a = compute_locations(f, 8, 47)
    pass
