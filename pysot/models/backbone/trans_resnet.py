

import math
from collections import OrderedDict
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from pysot.utils.misc_transt import NestedTensor
from pysot.models.transformer.utils import FrozenBatchNorm2d
from pysot.models.transformer.PositionEmbedding import build_position_encoding


def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, bias=False, dilation=dilation)


class BackboneNet(nn.Module):
    """Base class for backbone networks. Handles freezing layers etc.
    args:
        frozen_layers  -  Name of layers to freeze. Either list of strings, 'none' or 'all'. Default: 'none'.
    """

    def __init__(self, frozen_layers=()):
        super().__init__()

        if isinstance(frozen_layers, str):
            if frozen_layers.lower() == 'none':
                frozen_layers = ()
            elif frozen_layers.lower() != 'all':
                raise ValueError(
                    'Unknown option for frozen layers: \"{}\". Should be \"all\", \"none\" or list of layer names.'.format(
                        frozen_layers))

        self.frozen_layers = frozen_layers
        self._is_frozen_nograd = False

    def train(self, mode=True):
        super().train(mode)
        if mode == True:
            self._set_frozen_to_eval()
        if not self._is_frozen_nograd:
            self._set_frozen_to_nograd()
            self._is_frozen_nograd = True

    def _set_frozen_to_eval(self):
        if isinstance(self.frozen_layers, str) and self.frozen_layers.lower() == 'all':
            self.eval()
        else:
            for layer in self.frozen_layers:
                getattr(self, layer).eval()

    def _set_frozen_to_nograd(self):
        if isinstance(self.frozen_layers, str) and self.frozen_layers.lower() == 'all':
            for p in self.parameters():
                p.requires_grad_(False)
        else:
            for layer in self.frozen_layers:
                for p in getattr(self, layer).parameters():
                    p.requires_grad_(False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1, use_bn=True, bn_fn=nn.BatchNorm2d):
        super(BasicBlock, self).__init__()
        self.use_bn = use_bn
        self.conv1 = conv3x3(inplanes, planes, stride, dilation=dilation)

        if use_bn:
            self.bn1 = bn_fn(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, dilation=dilation)

        if use_bn:
            self.bn2 = bn_fn(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)

        if self.use_bn:
            out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)

        if self.use_bn:
            out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1, bn_fn=nn.BatchNorm2d):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = bn_fn(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation, bias=False, dilation=dilation)
        self.bn2 = bn_fn(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = bn_fn(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(BackboneNet):
    """ ResNet network module. Allows extracting specific feature blocks."""
    def __init__(self, block, layers, output_layers, inplanes=64, dilation_factor=1,
                 frozen_layers=(), freeze_bn=False, return_type='dict'):
        self.inplanes = inplanes
        if freeze_bn:
            self.bn_fn = FrozenBatchNorm2d
        else:
            self.bn_fn = nn.BatchNorm2d
        self.return_type = return_type

        super(ResNet, self).__init__(frozen_layers=frozen_layers)
        self.output_layers = output_layers
        self.conv1 = nn.Conv2d(3, inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = self.bn_fn(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        stride = [1 + (dilation_factor < l) for l in (8, 4, 2)]
        self.layer1 = self._make_layer(block, inplanes, layers[0], dilation=max(dilation_factor//8, 1))
        self.layer2 = self._make_layer(block, inplanes*2, layers[1], stride=stride[0], dilation=max(dilation_factor//4, 1))
        # self.layer3 = self._make_layer(block, inplanes*4, layers[2], stride=stride[1], dilation=max(dilation_factor//2, 1))
        self.layer3 = self._make_layer(block, inplanes*4, layers[2], stride=1, dilation=2)
        # self.layer4 = self._make_layer(block, inplanes*8, layers[3], stride=stride[2], dilation=dilation_factor)

        out_feature_strides = {'conv1': 4, 'layer1': 4, 'layer2': 4*stride[0], 'layer3': 4*stride[0]*stride[1],
                               'layer4': 4*stride[0]*stride[1]*stride[2]}

        # TODO better way?
        if isinstance(self.layer1[0], BasicBlock):
            out_feature_channels = {'conv1': inplanes, 'layer1': inplanes,
                                    'layer2': inplanes * 2, 'layer3': inplanes * 4, 'layer4': inplanes * 8}
        elif isinstance(self.layer1[0], Bottleneck):
            base_num_channels = 4 * inplanes
            out_feature_channels = {'conv1': inplanes, 'layer1': base_num_channels, 'layer2': base_num_channels * 2,
                                    'layer3': base_num_channels * 4, 'layer4': base_num_channels * 8}
        else:
            raise Exception('block not supported')

        self._out_feature_strides = out_feature_strides
        self._out_feature_channels = out_feature_channels

        # self.avgpool = nn.AvgPool2d(7, stride=1)
        # self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        # self.fc = nn.Linear(inplanes*8 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, FrozenBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def out_feature_strides(self, layer=None):
        if layer is None:
            return self._out_feature_strides
        else:
            return self._out_feature_strides[layer]

    def out_feature_channels(self, layer=None):
        if layer is None:
            return self._out_feature_channels
        else:
            return self._out_feature_channels[layer]

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                self.bn_fn(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, dilation=dilation, bn_fn=self.bn_fn))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, bn_fn=self.bn_fn))

        return nn.Sequential(*layers)

    def _add_output_and_check(self, name, x, outputs, output_layers):
        if self.return_type == 'dict':
            if name in output_layers:
                outputs[name] = x
            return len(output_layers) == len(outputs)
        elif self.return_type == 'list':
            if name in output_layers:
                outputs.append(x)
            return len(output_layers) == len(outputs)
        else:
            return name == output_layers[0]

    def forward(self, x, output_layers=None):
        """ Forward pass with input x. The output_layers specify the feature blocks which must be returned """
        if self.return_type == 'dict':
            outputs = OrderedDict()
        elif self.return_type == 'list':
            outputs = []
        else:
            outputs = None

        if output_layers is None:
            output_layers = self.output_layers

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        if self._add_output_and_check('conv1', x, outputs, output_layers):
            if self.return_type != 'list' and self.return_type != 'dict':
                outputs = x
            return outputs

        x = self.maxpool(x)

        x = self.layer1(x)

        if self._add_output_and_check('layer1', x, outputs, output_layers):
            if self.return_type != 'list' and self.return_type != 'dict':
                outputs = x
            return outputs

        x = self.layer2(x)

        if self._add_output_and_check('layer2', x, outputs, output_layers):
            if self.return_type != 'list' and self.return_type != 'dict':
                outputs = x
            return outputs

        x = self.layer3(x)

        if self._add_output_and_check('layer3', x, outputs, output_layers):
            if self.return_type != 'list' and self.return_type != 'dict':
                outputs = x
            return outputs

        # x = self.layer4(x)

        # if self._add_output_and_check('layer4', x, outputs, output_layers):
        #     return outputs
        #
        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)
        #
        # if self._add_output_and_check('fc', x, outputs, output_layers):
        #     return outputs
        #
        # if len(output_layers) == 1 and output_layers[0] == 'default':
        #     return x

        raise ValueError('output_layer is wrong.')


def resnet_baby(output_layers=None, pretrained=False, inplanes=16, **kwargs):
    """Constructs a ResNet-18 model.
    """

    if output_layers is None:
        output_layers = ['default']
    else:
        for l in output_layers:
            if l not in ['conv1', 'layer1', 'layer2', 'layer3', 'layer4', 'fc']:
                raise ValueError('Unknown layer: {}'.format(l))

    model = ResNet(BasicBlock, [2, 2, 2, 2], output_layers, inplanes=inplanes, **kwargs)

    if pretrained:
        raise NotImplementedError
    return model


def transt_resnet18(output_layers=None, **kwargs):
    """Constructs a ResNet-18 model.
    """

    if output_layers is None:
        output_layers = ['default']
    else:
        for l in output_layers:
            if l not in ['conv1', 'layer1', 'layer2', 'layer3', 'layer4', 'fc']:
                raise ValueError('Unknown layer: {}'.format(l))

    model = ResNet(BasicBlock, [2, 2, 2, 2], output_layers, **kwargs)
    return model


def transt_resnet50(output_layers=None, **kwargs):
    """Constructs a ResNet-50 model.
    """
    if output_layers is None:
        output_layers = ['default']
    else:
        for l in output_layers:
            if l not in ['conv1', 'layer1', 'layer2', 'layer3', 'layer4', 'fc']:
                raise ValueError('Unknown layer: {}'.format(l))

    model = ResNet(Bottleneck, [3, 4, 6, 3], output_layers, **kwargs)
    return model


class BackboneBase(nn.Module):
    def __init__(self, backbone: nn.Module, num_channels: int):
        super().__init__()
        self.body = backbone
        self.num_channels = num_channels

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, net_type, output_layers, frozen_layers):
        if net_type == 'resnet50':
            backbone = transt_resnet50(output_layers=output_layers, frozen_layers=frozen_layers)
        if net_type == 'resnet18':
            backbone = transt_resnet18(output_layers=output_layers, frozen_layers=frozen_layers)
        num_channels = 1024
        super().__init__(backbone, num_channels)


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos


def transt_resnet_pe(output_layers, frozen_layers='all', net_type='resnet50', hidden_dim=256, position_embedding='sine'):
    position_embedding = build_position_encoding(hidden_dim, position_embedding)
    backbone = Backbone(net_type=net_type, output_layers=output_layers, frozen_layers=frozen_layers)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model


def transt_resnet18_pe(output_layers, frozen_layers='all', hidden_dim=256, position_embedding='sine'):
    position_embedding = build_position_encoding(hidden_dim, position_embedding)
    backbone = Backbone(net_type='resnet18', output_layers=output_layers, frozen_layers=frozen_layers)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model


def transt_resnet50_pe(output_layers, frozen_layers='all', hidden_dim=256, position_embedding='sine'):
    position_embedding = build_position_encoding(hidden_dim, position_embedding)
    backbone = Backbone(net_type='resnet50', output_layers=output_layers, frozen_layers=frozen_layers)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model

