

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import BasicBlock, Bottleneck, conv1x1
from torchvision.models._utils import IntermediateLayerGetter

from pysot.utils.misc_transt import NestedTensor
from pysot.models.transformer.PositionEmbedding import build_position_encoding
from pysot.models.transformer.utils import FrozenBatchNorm2d


class ResNet(nn.Module):
    def __init__(self, block, layers, zero_init_residual=False, groups=1, width_per_group=64,
                 replace_stride_with_dilation=None, freeze_bn=False, last_layer=None):
        super(ResNet, self).__init__()
        # assert last_layer in ['layer1', 'layer2', 'layer3', 'layer4']
        self.last_layer = last_layer
        if freeze_bn:
            self._norm_layer = FrozenBatchNorm2d
        else:
            self._norm_layer = nn.BatchNorm2d

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = self._norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        if 'layer2' in self.last_layer:
            self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                           dilate=replace_stride_with_dilation[0])
            if 'layer3' in self.last_layer:
                self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                               dilate=replace_stride_with_dilation[1])
                if 'layer4' in self.last_layer:
                    self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                                   dilate=replace_stride_with_dilation[2])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)

        out = []
        if 'layer2' in self.last_layer:
            x = out2 = self.layer2(x)
            out.append(out2)
            if 'layer3' in self.last_layer:
                x = out3 = self.layer3(x)
                out.append(out3)
                if 'layer4' in self.last_layer:
                    x = out4 = self.layer4(x)
                    out.append(out4)
        return out


def stark_resnet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def stark_resnet34(**kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)


def stark_resnet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


def stark_resnet101(**kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)


def stark_resnet152(**kwargs):
    return ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)


backbone_dict = {
    'stark_resnet18': stark_resnet18,
    'stark_resnet34': stark_resnet34,
    'stark_resnet50': stark_resnet50,
    'stark_resnet101': stark_resnet101,
    'stark_resnet152': stark_resnet152
}


if __name__ == '__main__':
    net = stark_resnet50(replace_stride_with_dilation=[False, False, False],
                         last_layer=['layer2', 'layer3', 'layer4'], freeze_bn=False)
    a = net.state_dict()
    pass
