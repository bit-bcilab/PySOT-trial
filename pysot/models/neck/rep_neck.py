# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch.nn as nn
from pysot.models.backbone.repvgg import RepVGGBlock as RepBlock


class RepAdjustLayer(nn.Module):
    def __init__(self, in_channels, out_channels, crop_z=False):
        super(RepAdjustLayer, self).__init__()
        self.downsample = RepBlock(in_channels, out_channels, kernel_size=3, padding=1)
        self.crop_z = crop_z

    def forward(self, x):
        x = self.downsample(x)
        if x.size(3) < 20 and self.crop_z:
            l = 4
            r = l + 7
            x = x[:, :, l:r, l:r]
        return x


class RepAdjustAll(nn.Module):
    def __init__(self, in_channels, out_channels, crop_z=False):
        super(RepAdjustAll, self).__init__()
        self.num = len(out_channels)
        if self.num == 1:
            self.downsample = RepAdjustLayer(in_channels[0], out_channels[0], crop_z)
        else:
            for i in range(self.num):
                self.add_module('downsample' + str(i+2),
                                RepAdjustLayer(in_channels[i], out_channels[i], crop_z))

    def forward(self, features):
        if self.num == 1:
            return self.downsample(features)
        else:
            out = []
            for i in range(self.num):
                adj_layer = getattr(self, 'downsample'+str(i+2))
                out.append(adj_layer(features[i]))
            return out
