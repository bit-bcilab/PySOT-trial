

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pysot.models.transformer.utils import FrozenBatchNorm2d


class SEBlock(nn.Module):
    def __init__(self, input_channels, internal_neurons):
        super(SEBlock, self).__init__()
        self.down = nn.Conv2d(in_channels=input_channels, out_channels=internal_neurons, kernel_size=1, stride=1)
        self.up = nn.Conv2d(in_channels=internal_neurons, out_channels=input_channels, kernel_size=1, stride=1)
        self.input_channels = input_channels

    def forward(self, inputs):
        x = F.avg_pool2d(inputs, kernel_size=inputs.size(3))
        x = self.down(x)
        x = F.relu(x)
        x = self.up(x)
        x = torch.sigmoid(x)
        x = x.view(-1, self.input_channels, 1, 1)
        return inputs * x


def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1, freeze_bn=False):
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                        stride=stride, padding=padding, groups=groups, bias=False))
    if freeze_bn:
        result.add_module('bn', FrozenBatchNorm2d(out_channels))
    else:
        result.add_module('bn', nn.BatchNorm2d(out_channels))
    return result


class RepVGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 padding_mode='zeros', deploy=False, use_se=False, freeze_bn=False):
        super(RepVGGBlock, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels

        assert kernel_size == 3
        assert padding == 1

        padding_11 = padding - kernel_size // 2

        self.nonlinearity = nn.ReLU()

        if use_se:
            self.se = SEBlock(out_channels, internal_neurons=out_channels // 16)
        else:
            self.se = nn.Identity()

        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                         dilation=dilation, groups=groups, bias=True, padding_mode=padding_mode)

        else:
            if out_channels == in_channels and stride == 1:
                if freeze_bn:
                    self.rbr_identity = FrozenBatchNorm2d(in_channels)
                else:
                    self.rbr_identity = nn.BatchNorm2d(in_channels)
            else:
                self.rbr_identity = None

            self.rbr_dense = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                     stride=stride, padding=padding, groups=groups, freeze_bn=freeze_bn)
            self.rbr_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride,
                                   padding=padding_11, groups=groups, freeze_bn=freeze_bn)

    def forward(self, inputs):
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.se(self.rbr_reparam(inputs)))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)
        return self.nonlinearity(self.se(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out))

    #   This func derives the equivalent kernel and bias in a DIFFERENTIABLE way.
    #   You can get the equivalent kernel and bias at any time and do whatever you want,
    #   for example, apply some penalties or constraints during training, just like you do to the other models.
    #   May be useful for quantization or pruning.
    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d) or isinstance(branch, FrozenBatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        if hasattr(self, 'rbr_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(in_channels=self.rbr_dense.conv.in_channels,
                                     out_channels=self.rbr_dense.conv.out_channels,
                                     kernel_size=self.rbr_dense.conv.kernel_size, stride=self.rbr_dense.conv.stride,
                                     padding=self.rbr_dense.conv.padding, dilation=self.rbr_dense.conv.dilation,
                                     groups=self.rbr_dense.conv.groups, bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')


class RepVGG(nn.Module):
    def __init__(self, num_blocks, width_multiplier=None, override_groups_map=None, deploy=False,
                 use_se=False, last_layer='stage3', freeze_bn=False, strides=None):
        super(RepVGG, self).__init__()
        if strides is None:
            strides = [2, 2, 2, 2, 2]
        self.last_layer = last_layer

        assert len(width_multiplier) == 4

        self.deploy = deploy
        self.override_groups_map = override_groups_map or dict()
        self.use_se = use_se

        assert 0 not in self.override_groups_map

        self.in_planes = min(64, int(64 * width_multiplier[0]))

        self.stage0 = RepVGGBlock(in_channels=3, out_channels=self.in_planes, kernel_size=3, stride=strides[0], 
                                  padding=1, deploy=self.deploy, use_se=self.use_se, freeze_bn=freeze_bn)
        self.cur_layer_idx = 1
        self.stage1 = self._make_stage(int(64 * width_multiplier[0]), num_blocks[0], 
                                       stride=strides[1], freeze_bn=freeze_bn)

        if isinstance(self.last_layer, list):
            self.stage2 = self._make_stage(int(128 * width_multiplier[1]), num_blocks[1],
                                           stride=strides[2], freeze_bn=freeze_bn)
            self.stage3 = self._make_stage(int(256 * width_multiplier[2]), num_blocks[2],
                                           stride=strides[3], freeze_bn=freeze_bn)
            self.stage4 = self._make_stage(int(512 * width_multiplier[3]), num_blocks[3],
                                           stride=strides[4], freeze_bn=freeze_bn)
        else:
            if self.last_layer != 'stage1':
                self.stage2 = self._make_stage(int(128 * width_multiplier[1]), num_blocks[1],
                                               stride=strides[2], freeze_bn=freeze_bn)
                if self.last_layer != 'stage2':
                    self.stage3 = self._make_stage(int(256 * width_multiplier[2]), num_blocks[2],
                                                   stride=strides[3], freeze_bn=freeze_bn)
                    if self.last_layer != 'stage3':
                        self.stage4 = self._make_stage(int(512 * width_multiplier[3]), num_blocks[3],
                                                       stride=strides[4], freeze_bn=freeze_bn)

    def _make_stage(self, planes, num_blocks, stride, freeze_bn=False):
        strides = [stride] + [1] * (num_blocks - 1)
        blocks = []
        for stride in strides:
            cur_groups = self.override_groups_map.get(self.cur_layer_idx, 1)
            blocks.append(RepVGGBlock(in_channels=self.in_planes, out_channels=planes, kernel_size=3,
                                      stride=stride, padding=1, groups=cur_groups, deploy=self.deploy,
                                      use_se=self.use_se, freeze_bn=freeze_bn))
            self.in_planes = planes
            self.cur_layer_idx += 1
        return nn.Sequential(*blocks)

    def forward(self, x):
        out = self.stage0(x)
        out = self.stage1(out)
        if not isinstance(self.last_layer, list):
            if self.last_layer != 'stage1':
                out = self.stage2(out)
                if self.last_layer != 'stage2':
                    out = self.stage3(out)
                    if self.last_layer != 'stage3':
                        out = self.stage4(out)
            return out
        else:
            outs = []
            if 'stage1' in self.last_layer:
                outs.append(out)
            if 'stage2' in self.last_layer:
                out = self.stage2(out)
                outs.append(out)
            if 'stage3' in self.last_layer:
                out = self.stage3(out)
                outs.append(out)
            if 'stage4' in self.last_layer:
                out = self.stage4(out)
                outs.append(out)
            return outs


optional_groupwise_layers = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]
g2_map = {l: 2 for l in optional_groupwise_layers}
g4_map = {l: 4 for l in optional_groupwise_layers}


def create_RepVGG_Stark(deploy=False, last_layer='stage3', freeze_bn=False, strides=None, num_blocks=None):
    if num_blocks is None:
        num_blocks = [2, 4, 4, 1]
    return RepVGG(num_blocks=num_blocks, width_multiplier=[0.75, 0.75, 0.75, 2.5], override_groups_map=None,
                  deploy=deploy, last_layer=last_layer, freeze_bn=freeze_bn, strides=strides)


def create_RepVGG_Stark_(deploy=False, last_layer='stage3', freeze_bn=False, strides=None, num_blocks=None):
    if num_blocks is None:
        num_blocks = [2, 4, 8, 1]
    return RepVGG(num_blocks=num_blocks, width_multiplier=[0.75, 0.75, 0.75, 2.5], override_groups_map=None,
                  deploy=deploy, last_layer=last_layer, freeze_bn=freeze_bn, strides=strides)


def create_RepVGG_A0(deploy=False, last_layer='stage3', freeze_bn=False, strides=None, num_blocks=None):
    if num_blocks is None:
        num_blocks = [2, 4, 14, 1]
    return RepVGG(num_blocks=num_blocks, width_multiplier=[0.75, 0.75, 0.75, 2.5], override_groups_map=None,
                  deploy=deploy, last_layer=last_layer, freeze_bn=freeze_bn, strides=strides)


def create_RepVGG_A1(deploy=False, last_layer='stage3', freeze_bn=False, strides=None, num_blocks=None):
    if num_blocks is None:
        num_blocks = [2, 4, 14, 1]
    return RepVGG(num_blocks=num_blocks, width_multiplier=[1, 1, 1, 2.5], override_groups_map=None,
                  deploy=deploy, last_layer=last_layer, freeze_bn=freeze_bn, strides=strides)


def create_RepVGG_A2(deploy=False, last_layer='stage3', freeze_bn=False, strides=None, num_blocks=None):
    if num_blocks is None:
        num_blocks = [2, 4, 14, 1]
    return RepVGG(num_blocks=num_blocks, width_multiplier=[1.5, 1.5, 1.5, 2.75],  override_groups_map=None,
                  deploy=deploy, last_layer=last_layer, freeze_bn=freeze_bn, strides=strides)


def create_RepVGG_B0(deploy=False, last_layer='stage3', freeze_bn=False, strides=None, num_blocks=[4, 6, 16, 1]):
    return RepVGG(num_blocks=num_blocks, width_multiplier=[1, 1, 1, 2.5], override_groups_map=None,
                  deploy=deploy, last_layer=last_layer, freeze_bn=freeze_bn, strides=strides)


def create_RepVGG_B1(deploy=False, last_layer='stage3', freeze_bn=False, strides=None, num_blocks=[4, 6, 16, 1]):
    return RepVGG(num_blocks=num_blocks, width_multiplier=[2, 2, 2, 4], override_groups_map=None,
                  deploy=deploy, last_layer=last_layer, freeze_bn=freeze_bn, strides=strides)


def create_RepVGG_B1g2(deploy=False, last_layer='stage3', freeze_bn=False, strides=None):
    return RepVGG(num_blocks=[4, 6, 16, 1], width_multiplier=[2, 2, 2, 4], override_groups_map=g2_map,
                  deploy=deploy, last_layer=last_layer, freeze_bn=freeze_bn, strides=strides)


def create_RepVGG_B1g4(deploy=False, last_layer='stage3', freeze_bn=False, strides=None):
    return RepVGG(num_blocks=[4, 6, 16, 1], width_multiplier=[2, 2, 2, 4], override_groups_map=g4_map,
                  deploy=deploy, last_layer=last_layer, freeze_bn=freeze_bn, strides=strides)


def create_RepVGG_B2(deploy=False, last_layer='stage3', freeze_bn=False, strides=None):
    return RepVGG(num_blocks=[4, 6, 16, 1], width_multiplier=[2.5, 2.5, 2.5, 5], override_groups_map=None,
                  deploy=deploy, last_layer=last_layer, freeze_bn=freeze_bn, strides=strides)


def create_RepVGG_B2g2(deploy=False, last_layer='stage3', freeze_bn=False, strides=None):
    return RepVGG(num_blocks=[4, 6, 16, 1], width_multiplier=[2.5, 2.5, 2.5, 5], override_groups_map=g2_map,
                  deploy=deploy, last_layer=last_layer, freeze_bn=freeze_bn, strides=strides)


def create_RepVGG_B2g4(deploy=False, last_layer='stage3', freeze_bn=False, strides=None):
    return RepVGG(num_blocks=[4, 6, 16, 1], width_multiplier=[2.5, 2.5, 2.5, 5], override_groups_map=g4_map,
                  deploy=deploy, last_layer=last_layer, freeze_bn=freeze_bn, strides=strides)


def create_RepVGG_B3(deploy=False, last_layer='stage3', freeze_bn=False, strides=None):
    return RepVGG(num_blocks=[4, 6, 16, 1], width_multiplier=[3, 3, 3, 5], override_groups_map=None,
                  deploy=deploy, last_layer=last_layer, freeze_bn=freeze_bn, strides=strides)


def create_RepVGG_B3g2(deploy=False, last_layer='stage3', freeze_bn=False, strides=None):
    return RepVGG(num_blocks=[4, 6, 16, 1], width_multiplier=[3, 3, 3, 5], override_groups_map=g2_map,
                  deploy=deploy, last_layer=last_layer, freeze_bn=freeze_bn, strides=strides)


def create_RepVGG_B3g4(deploy=False, last_layer='stage3', freeze_bn=False, strides=None):
    return RepVGG(num_blocks=[4, 6, 16, 1], width_multiplier=[3, 3, 3, 5], override_groups_map=g4_map,
                  deploy=deploy, last_layer=last_layer, freeze_bn=freeze_bn, strides=strides)


def create_RepVGG_D2se(deploy=False, last_layer='stage3', freeze_bn=False, strides=None):
    return RepVGG(num_blocks=[8, 14, 24, 1], width_multiplier=[2.5, 2.5, 2.5, 5], use_se=True, override_groups_map=None,
                  deploy=deploy, last_layer=last_layer, freeze_bn=freeze_bn, strides=strides)


func_dict = {
    'RepVGG-A0': create_RepVGG_A0,
    'RepVGG-A1': create_RepVGG_A1,
    'RepVGG-A2': create_RepVGG_A2,
    'RepVGG-B0': create_RepVGG_B0,
    'RepVGG-B1': create_RepVGG_B1,
    'RepVGG-B1g2': create_RepVGG_B1g2,
    'RepVGG-B1g4': create_RepVGG_B1g4,
    'RepVGG-B2': create_RepVGG_B2,
    'RepVGG-B2g2': create_RepVGG_B2g2,
    'RepVGG-B2g4': create_RepVGG_B2g4,
    'RepVGG-B3': create_RepVGG_B3,
    'RepVGG-B3g2': create_RepVGG_B3g2,
    'RepVGG-B3g4': create_RepVGG_B3g4,
    'RepVGG-D2se': create_RepVGG_D2se,  # Updated at April 25, 2021. This is not reported in the CVPR paper.
}


def get_RepVGG_func_by_name(name):
    return func_dict[name]


def repvgg_model_convert(model: torch.nn.Module, save_path=None, do_copy=True):
    if do_copy:
        model = copy.deepcopy(model)
    for module in model.modules():
        if hasattr(module, 'switch_to_deploy'):
            module.switch_to_deploy()
    if save_path is not None:
        torch.save(model.state_dict(), save_path)
    return model


if __name__ == '__main__':
    net1 = create_RepVGG_A0(freeze_bn=False)
    # std1 = torch.load('E://Stark-main/RepVGG-A0-train.pth')
    # net1.load_state_dict(std1, strict=False)

    net2 = repvgg_model_convert(net1)

    test = torch.rand((4, 3, 256, 256))
    o1 = net1(test)
    o2 = net2(test)
    err = (o1 - o2).abs().sum()

    net2 = create_RepVGG_Stark(freeze_bn=True, strides=[2, 2, 2, 1, 2])
    std2 = torch.load('E://Stark-main/repvgg-a0.pth')['state_dict']

    a = 0
    pass
