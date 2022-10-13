# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch.cuda.amp
import torch.nn as nn
import torch.nn.functional as F

from pysot.models.backbone import get_backbone
from pysot.models.neck import get_neck


class BaseModelBuilder(nn.Module):
    def __init__(self, cfg):
        super(BaseModelBuilder, self).__init__()
        self.T = 0.4
        self.ti = 0.3
        self.ts = 0.3

        self.train_epoch = 0
        self.update_settings = None
        self.weights = None
        self.logger = None
        self.validate = False

        self.cfg = cfg

        # build backbone
        self.backbone = get_backbone(cfg.BACKBONE.TYPE, **cfg.BACKBONE.KWARGS)

        # build adjust layer
        if cfg.ADJUST.ADJUST:
            self.neck = get_neck(cfg.ADJUST.TYPE, **cfg.ADJUST.KWARGS)

    def track(self, x):
        raise NotImplementedError

    def forward(self, data):
        if self.cfg.AMP:
            with torch.cuda.amp.autocast():
                if 'trial' in self.cfg.MODE:
                    return self.forward_trial(data)
                elif self.cfg.MODE == 'teacher':
                    return self.forward_teacher(data)
                elif self.cfg.MODE == 'student':
                    return self.forward_teacher(data)
                else:
                    return self.forward_original(data)
        else:
            if 'trial' in self.cfg.MODE:
                return self.forward_trial(data)
            elif self.cfg.MODE == 'teacher':
                return self.forward_teacher(data)
            elif self.cfg.MODE == 'student':
                return self.forward_student(data)
            else:
                return self.forward_original(data)

    def forward_original(self, data):
        raise NotImplementedError

    def forward_trial(self, data):
        raise NotImplementedError

    def get_head_parameters(self):
        head_params = []
        return head_params

    def forward_teacher(self, data):
        return None

    def forward_student(self, data):
        return None

    def log_softmax(self, cls, return_pred=False):
        b, a2, h, w = cls.size()
        if self.cfg.BASE == 'anchor':
            cls = cls.view(b, 2, a2 // 2, h, w)
            cls = cls.permute(0, 2, 3, 4, 1).contiguous()
        elif self.cfg.BASE == 'point':
            cls = cls.permute(0, 2, 3, 1).contiguous()

        cls_ = F.log_softmax(cls, dim=-1)
        if not return_pred:
            return cls_
        else:
            return cls_, cls
