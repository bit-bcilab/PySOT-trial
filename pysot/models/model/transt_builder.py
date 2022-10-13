# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F

from pysot.models.model.base_model_builder import BaseModelBuilder
from pysot.utils.misc_transt import NestedTensor, nested_tensor_from_tensor, nested_tensor_from_tensor_2
from pysot.models.transformer.transt_transformer import build_featurefusion_network

from trial.loss import sofTmax, label2weight, label_update_
from trial.loss import weighted_select_cross_entropy_loss, weighted_iou_loss, weighted_l1_loss
from trial.utils.iou import process_box, bbox_iou
from trial.Decoders import TransTDecoder
from trial.utils.image import normalize_batch_cuda


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class TransTBuilder(BaseModelBuilder):
    def __init__(self, cfg):
        super(TransTBuilder, self).__init__(cfg)

        self.featurefusion_network = build_featurefusion_network(cfg.TRANS)
        hidden_dim = cfg.TRANS.hidden_dim
        self.class_embed = MLP(hidden_dim, hidden_dim, cfg.NUM_CLASSES + 1, 3)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.input_proj = nn.Conv2d(self.backbone.num_channels, hidden_dim, kernel_size=1)

        self.ratio = torch.tensor(self.cfg.TRAIN.SEARCH_SIZE, dtype=torch.float32).repeat(4).view((1, 1, 1, -1)).cuda()
        self.mean = torch.tensor(cfg.NORMALIZE_MEAN, dtype=torch.float32).view((1, -1, 1, 1)).cuda()
        self.std = torch.tensor(cfg.NORMALIZE_STD, dtype=torch.float32).view((1, -1, 1, 1)).cuda()

        self.T = 0.4
        self.ti = 0.10
        self.ts = 0.35

    def get_head_parameters(self):
        head_params = [self.featurefusion_network.parameters(), self.input_proj.parameters(),
                       self.class_embed.parameters(), self.bbox_embed.parameters()]
        return head_params

    def template(self, z):
        z = normalize_batch_cuda(z, self.mean, self.std, False)
        if not isinstance(z, NestedTensor):
            z = nested_tensor_from_tensor_2(z)
        zf, pos_template = self.backbone(z)
        self.zf = zf
        self.pos_template = pos_template

    def track(self, x):
        x = normalize_batch_cuda(x, self.mean, self.std, False)
        if not isinstance(x, NestedTensor):
            x = nested_tensor_from_tensor_2(x)
        features_search, pos_search = self.backbone(x)
        feature_template = self.zf
        pos_template = self.pos_template
        src_search, mask_search = features_search[-1].decompose()
        assert mask_search is not None
        src_template, mask_template = feature_template[-1].decompose()
        assert mask_template is not None
        hs = self.featurefusion_network(self.input_proj(src_template), mask_template, self.input_proj(src_search), mask_search, pos_template[-1], pos_search[-1])

        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        return out

    def forward_original(self, data):
        """ only used in training
        """
        template = data['template'].cuda()
        search = data['search'].cuda()
        bbox_mask = data['bbox_mask'].cuda()
        label_cls = data['label_cls'].cuda()
        label_loc = data['label_loc'].cuda()
        bbox = data['bbox'].cuda()[:, None, None, :]
        positive = data['pos'].cuda()

        if not isinstance(search, NestedTensor):
            search = nested_tensor_from_tensor(search)
        if not isinstance(template, NestedTensor):
            template = nested_tensor_from_tensor(template)

        feature_search, pos_search = self.backbone(search)
        feature_template, pos_template = self.backbone(template)
        src_search, mask_search = feature_search[-1].decompose()
        assert mask_search is not None
        src_template, mask_template = feature_template[-1].decompose()
        assert mask_template is not None
        hs = self.featurefusion_network(self.input_proj(src_template), mask_template, self.input_proj(src_search),
                                        mask_search, pos_template[-1], pos_search[-1])

        # shape = (1, b, N, 2),
        cls_ = self.class_embed(hs)[0]
        loc = self.bbox_embed(hs).sigmoid()[0]
        loc = loc.view((-1, self.cfg.TRAIN.OUTPUT_SIZ, self.cfg.TRAIN.OUTPUT_SIZE, 4))
        cls_ = torch.cat([cls_[..., 1], cls_[..., 0]], dim=-1)
        cls_ = cls_.view((-1, self.cfg.TRAIN.OUTPUT_SIZ, self.cfg.TRAIN.OUTPUT_SIZE, 2))
        cls = F.log_softmax(cls_, dim=-1)
        cls = cls.view

        score, boxes = TransTDecoder(cls_, loc, (self.cfg.TRAIN.SEARCH_SIZE, self.cfg.TRAIN.SEARCH_SIZE))
        bbox = process_box(bbox)
        boxes = process_box(boxes)
        iou, union_area = bbox_iou(bbox, boxes)

        with torch.no_grad():
            pos_mask = (label_cls == 1.).type(torch.float32)
            neg_mask = (label_cls == 0.).type(torch.float32)
            _pos_mask = (pos_mask == 0.).type(torch.float32)

            neg_weight = label2weight(neg_mask, avg='sample')
            pos_weight_cls = label2weight(pos_mask, avg='batch')
            pos_weight_loc = label2weight(pos_mask, avg='batch')
            l1_loss = weighted_l1_loss(label_loc, loc, pos_mask, smooth=True)

        # cross entropy loss
        pos_loss, neg_loss = weighted_select_cross_entropy_loss(cls, label_cls,
                                                                pos_weight=pos_weight_cls, neg_weight=neg_weight)
        cls_loss = pos_loss * self.weights['pos_weight'] + neg_loss * self.weights['neg_weight']

        iou_loss = weighted_iou_loss(bbox, boxes, weight=pos_weight_loc,
                                     iou=iou, union_area=union_area, loss_type='ciou')[0]

        loc_loss = iou_loss
        # a0 = label_cls.cpu().detach().numpy()
        # a1 = iou.cpu().detach().numpy()
        # a2 = score.cpu().detach().numpy()
        # a3 = data['label_cls'].numpy()
        # a4 = pw.cpu().detach().numpy()
        # import cv2
        # search_img = search.permute((0, 2, 3, 1)).type(torch.uint8).cpu().detach().numpy()
        # template_img = template.permute((0, 2, 3, 1)).type(torch.uint8).cpu().detach().numpy()
        # j = 2
        # cv2.imshow('0', template_img[j])
        # box = list(map(int, data['bbox'].numpy()[j]))
        # cv2.rectangle(search_img[j], (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        # cv2.imshow('1', search_img[j])
        # cv2.waitKey()

        outputs = {}
        outputs['total_loss'] = self.cfg.TRAIN.CLS_WEIGHT * cls_loss + self.cfg.TRAIN.LOC_WEIGHT * loc_loss
        outputs['cls_loss'] = cls_loss
        outputs['pos_loss'] = pos_loss
        outputs['neg_loss'] = neg_loss
        outputs['loc_loss'] = loc_loss
        outputs['iou_loss'] = iou_loss
        outputs['l1_loss'] = l1_loss
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

        template = normalize_batch_cuda(template, self.mean, self.std, False)
        search = normalize_batch_cuda(search, self.mean, self.std, False)

        if not isinstance(search, NestedTensor):
            search = nested_tensor_from_tensor(search)
        if not isinstance(template, NestedTensor):
            template = nested_tensor_from_tensor(template)

        feature_search, pos_search = self.backbone(search)
        feature_template, pos_template = self.backbone(template)
        src_search, mask_search = feature_search[-1].decompose()
        assert mask_search is not None
        src_template, mask_template = feature_template[-1].decompose()
        assert mask_template is not None
        hs = self.featurefusion_network(self.input_proj(src_template), mask_template, self.input_proj(src_search),
                                        mask_search, pos_template[-1], pos_search[-1])

        # shape = (1, b, N, 2),
        cls_ = self.class_embed(hs)[0]
        loc_ = self.bbox_embed(hs).sigmoid()[0]
        loc_ = loc_.view((-1, self.cfg.TRAIN.OUTPUT_SIZE, self.cfg.TRAIN.OUTPUT_SIZE, 4))
        cls_ = torch.cat([cls_[..., 1:2], cls_[..., 0:1]], dim=-1)
        cls_ = cls_.view((-1, self.cfg.TRAIN.OUTPUT_SIZE, self.cfg.TRAIN.OUTPUT_SIZE, 2))
        cls = F.log_softmax(cls_, dim=-1)
        loc = loc_.permute((0, 3, 1, 2)).contiguous()

        score, boxes, _ = TransTDecoder(cls_, loc_, self.ratio, self.cfg.TRAIN.SEARCH_SIZE)
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

        l1_loss = weighted_l1_loss(label_loc, loc, pos_weight_l1, smooth=False)

        # cross entropy loss
        pos_loss, neg_loss = weighted_select_cross_entropy_loss(cls, label_cls, pos_weight=pos_weight_cls, neg_weight=neg_weight)
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
        # search_img = search.tensors.permute((0, 2, 3, 1)).contiguous().type(torch.uint8).cpu().detach().numpy()
        # template_img = template.tensors.permute((0, 2, 3, 1)).contiguous().type(torch.uint8).cpu().detach().numpy()
        # j = 2
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
