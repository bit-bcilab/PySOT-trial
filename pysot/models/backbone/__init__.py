# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from pysot.models.backbone.alexnet import alexnetlegacy, alexnet
from pysot.models.backbone.mobile_v2 import mobilenetv2
from pysot.models.backbone.resnet_atrous import resnet18, resnet34, resnet50
from pysot.models.backbone.googlenet import Inception3
from pysot.models.backbone.googlenet_ou import Inception3_ou
from pysot.models.backbone.trans_resnet import transt_resnet18, transt_resnet50, transt_resnet18_pe, transt_resnet50_pe
from pysot.models.backbone.light_backbone import light_backbone
from pysot.models.backbone.repvgg import func_dict as repvggs, create_RepVGG_Stark
from pysot.models.backbone.stark_resnet import backbone_dict as starks


BACKBONES = {
              'alexnetlegacy': alexnetlegacy,
              'mobilenetv2': mobilenetv2,
              'resnet18': resnet18,
              'resnet34': resnet34,
              'resnet50': resnet50,
              'alexnet': alexnet,

              'googlenet': Inception3,
              'googlenet_ou': Inception3_ou,

              'transt_resnet18': transt_resnet18,
              'transt_resnet50': transt_resnet50,
              'transt_resnet18_pe': transt_resnet18_pe,
              'transt_resnet50_pe': transt_resnet50_pe,

              'light_backbone': light_backbone,
              'repvgg_stark': create_RepVGG_Stark,
            }
BACKBONES.update(repvggs)
BACKBONES.update(starks)


def get_backbone(name, **kwargs):
    return BACKBONES[name](**kwargs)
