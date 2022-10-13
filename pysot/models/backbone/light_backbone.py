
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch._six import container_abcs

from collections.__init__ import OrderedDict
from functools import partial
from typing import Tuple, Optional, List
from itertools import repeat


def resolve_se_args(kwargs, in_chs, act_layer=None):
    se_kwargs = kwargs.copy() if kwargs is not None else {}
    # fill in args that aren't specified with the defaults
    for k, v in _SE_ARGS_DEFAULT.items():
        se_kwargs.setdefault(k, v)
    # some models, like MobilNetV3, calculate SE reduction chs from the containing block's mid_ch instead of in_ch
    if not se_kwargs.pop('reduce_mid'):
        se_kwargs['reduced_base_chs'] = in_chs
    # act_layer override, if it remains None, the containing block's act_layer will be used
    if se_kwargs['act_layer'] is None:
        assert act_layer is not None
        se_kwargs['act_layer'] = act_layer
    return se_kwargs


def swish(x, inplace: bool = False):
    """Swish - Described in: https://arxiv.org/abs/1710.05941
    """
    return x.mul_(x.sigmoid()) if inplace else x.mul(x.sigmoid())


class Swish(nn.Module):
    def __init__(self, inplace: bool = False):
        super(Swish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return swish(x, self.inplace)


def sigmoid(x, inplace: bool = False):
    return x.sigmoid_() if inplace else x.sigmoid()


class Sigmoid(nn.Module):
    def __init__(self, inplace: bool = False):
        super(Sigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x.sigmoid_() if self.inplace else x.sigmoid()


def hard_sigmoid(x, inplace: bool = False):
    if inplace:
        return x.add_(3.).clamp_(0., 6.).div_(6.)
    else:
        return F.relu6(x + 3.) / 6.


class HardSigmoid(nn.Module):
    def __init__(self, inplace: bool = False):
        super(HardSigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return hard_sigmoid(x, self.inplace)


def _ntuple(n):
    # def parse(x):
    #     if isinstance(x, container_abcs.Iterable):
    #         return x
    #     return tuple(repeat(x, n))

    return


tup_pair = _ntuple(2)

_SE_ARGS_DEFAULT = dict(
    gate_fn=sigmoid,
    act_layer=None,
    reduce_mid=False,
    divisor=1)


def make_divisible(v, divisor=8, min_value=None):
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def round_channels(channels, multiplier=1.0, divisor=8, channel_min=None):
    """Round number of filters based on depth multiplier."""
    if not multiplier:
        return channels
    channels *= multiplier
    return make_divisible(channels, divisor, channel_min)


def create_conv2d_pad(in_chs, out_chs, kernel_size, **kwargs):
    padding = kwargs.pop('padding', '')
    kwargs.setdefault('bias', False)
    padding, is_dynamic = get_padding_value(padding, kernel_size, **kwargs)
    return nn.Conv2d(in_chs, out_chs, kernel_size, padding=padding, **kwargs)


def create_conv2d(in_chs, out_chs, kernel_size, **kwargs):
    """ Select a 2d convolution implementation based on arguments
    Creates and returns one of torch.nn.Conv2d, Conv2dSame, MixedConv2d, or CondConv2d.
    Used extensively by EfficientNet, MobileNetv3 and related networks.
    """
    assert 'groups' not in kwargs  # only use 'depthwise' bool arg
    depthwise = kwargs.pop('depthwise', False)
    groups = out_chs if depthwise else 1
    m = create_conv2d_pad(in_chs, out_chs, kernel_size, groups=groups, **kwargs)
    return m


def get_condconv_initializer(initializer, num_experts, expert_shape):
    def condconv_initializer(weight):
        """CondConv initializer function."""
        num_params = np.prod(expert_shape)
        if (len(weight.shape) != 2 or weight.shape[0] != num_experts or
                weight.shape[1] != num_params):
            raise (ValueError(
                'CondConv variables must have shape [num_experts, num_params]'))
        for i in range(num_experts):
            initializer(weight[i].view(expert_shape))

    return condconv_initializer


def _init_weight_goog(m, n='', fix_group_fanout=True, last_bn=None):
    """ Weight initialization as per Tensorflow official implementations.
    Args:
        m (nn.Module): module to init
        n (str): module name
        fix_group_fanout (bool): enable correct (matching Tensorflow TPU impl) fanout calculation w/ group convs
    Handles layers in EfficientNet, EfficientNet-CondConv, MixNet, MnasNet, MobileNetV3, etc:
    * https://github.com/tensorflow/tpu/blob/master/models/official/mnasnet/mnasnet_model.py
    * https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/efficientnet_model.py
    """
    if isinstance(m, CondConv2d):
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        if fix_group_fanout:
            fan_out //= m.groups
        init_weight_fn = get_condconv_initializer(
            lambda w: w.data.normal_(0, math.sqrt(2.0 / fan_out)), m.num_experts, m.weight_shape)
        init_weight_fn(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.Conv2d):
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        if fix_group_fanout:
            fan_out //= m.groups
        m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        if n in last_bn:
            m.weight.data.zero_()
            m.bias.data.zero_()
        else:
            m.weight.data.fill_(1.0)
            m.bias.data.zero_()
        m.weight.data.fill_(1.0)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        fan_out = m.weight.size(0)  # fan-out
        fan_in = 0
        if 'routing_fn' in n:
            fan_in = m.weight.size(1)
        init_range = 1.0 / math.sqrt(fan_in + fan_out)
        m.weight.data.uniform_(-init_range, init_range)
        m.bias.data.zero_()


def efficientnet_init_weights(model: nn.Module, init_fn=None, zero_gamma=False):
    last_bn = []
    if zero_gamma:
        prev_n = ''
        for n, m in model.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                if ''.join(prev_n.split('.')[:-1]) != ''.join(n.split('.')[:-1]):
                    last_bn.append(prev_n)
                prev_n = n
        last_bn.append(prev_n)

    init_fn = init_fn or _init_weight_goog
    for n, m in model.named_modules():
        init_fn(m, n, last_bn=last_bn)


def get_padding(kernel_size: int, stride: int = 1, dilation: int = 1, **_) -> int:
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding


def is_static_pad(kernel_size: int, stride: int = 1, dilation: int = 1, **_):
    return stride == 1 and (dilation * (kernel_size - 1)) % 2 == 0


def get_padding_value(padding, kernel_size, **kwargs) -> Tuple[Tuple, bool]:
    dynamic = False
    if isinstance(padding, str):
        # for any string padding, the padding will be calculated for you, one of three ways
        padding = padding.lower()
        if padding == 'same':
            # TF compatible 'SAME' padding, has a performance and GPU memory allocation impact
            if is_static_pad(kernel_size, **kwargs):
                # static case, no extra overhead
                padding = get_padding(kernel_size, **kwargs)
            else:
                # dynamic 'SAME' padding, has runtime/GPU memory overhead
                padding = 0
                dynamic = True
        elif padding == 'valid':
            # 'VALID' padding, same as padding=0
            padding = 0
        else:
            # Default to PyTorch style 'same'-ish symmetric padding
            padding = get_padding(kernel_size, **kwargs)
    return padding, dynamic


def get_same_padding(x: int, k: int, s: int, d: int):
    return max((math.ceil(x / s) - 1) * s + (k - 1) * d + 1 - x, 0)


def pad_same(x, k: List[int], s: List[int], d: List[int] = (1, 1), value: float = 0):
    ih, iw = x.size()[-2:]
    pad_h, pad_w = get_same_padding(ih, k[0], s[0], d[0]), get_same_padding(iw, k[1], s[1], d[1])
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2], value=value)
    return x


def conv2d_same(
        x, weight: torch.Tensor, bias: Optional[torch.Tensor] = None, stride: Tuple[int, int] = (1, 1),
        padding: Tuple[int, int] = (0, 0), dilation: Tuple[int, int] = (1, 1), groups: int = 1):
    x = pad_same(x, weight.shape[-2:], stride, dilation)
    return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)


class CondConv2d(nn.Module):
    """ Conditionally Parameterized Convolution
    Inspired by: https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/condconv/condconv_layers.py
    Grouped convolution hackery for parallel execution of the per-sample kernel filters inspired by this discussion:
    https://github.com/pytorch/pytorch/issues/17983
    """
    __constants__ = ['bias', 'in_channels', 'out_channels', 'dynamic_padding']

    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding='', dilation=1, groups=1, bias=False, num_experts=4):
        super(CondConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = tup_pair(kernel_size)
        self.stride = tup_pair(stride)
        padding_val, is_padding_dynamic = get_padding_value(
            padding, kernel_size, stride=stride, dilation=dilation)
        self.dynamic_padding = is_padding_dynamic  # if in forward to work with torchscript
        self.padding = tup_pair(padding_val)
        self.dilation = tup_pair(dilation)
        self.groups = groups
        self.num_experts = num_experts

        self.weight_shape = (self.out_channels, self.in_channels // self.groups) + self.kernel_size
        weight_num_param = 1
        for wd in self.weight_shape:
            weight_num_param *= wd
        self.weight = torch.nn.Parameter(torch.Tensor(self.num_experts, weight_num_param))

        if bias:
            self.bias_shape = (self.out_channels,)
            self.bias = torch.nn.Parameter(torch.Tensor(self.num_experts, self.out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        init_weight = get_condconv_initializer(
            partial(nn.init.kaiming_uniform_, a=math.sqrt(5)), self.num_experts, self.weight_shape)
        init_weight(self.weight)
        if self.bias is not None:
            fan_in = np.prod(self.weight_shape[1:])
            bound = 1 / math.sqrt(fan_in)
            init_bias = get_condconv_initializer(
                partial(nn.init.uniform_, a=-bound, b=bound), self.num_experts, self.bias_shape)
            init_bias(self.bias)

    def forward(self, x, routing_weights):
        B, C, H, W = x.shape
        weight = torch.matmul(routing_weights, self.weight)
        new_weight_shape = (B * self.out_channels, self.in_channels // self.groups) + self.kernel_size
        weight = weight.view(new_weight_shape)
        bias = None
        if self.bias is not None:
            bias = torch.matmul(routing_weights, self.bias)
            bias = bias.view(B * self.out_channels)
        # move batch elements with channels so each batch element can be efficiently convolved with separate kernel
        x = x.view(1, B * C, H, W)
        if self.dynamic_padding:
            out = conv2d_same(
                x, weight, bias, stride=self.stride, padding=self.padding,
                dilation=self.dilation, groups=self.groups * B)
        else:
            out = F.conv2d(
                x, weight, bias, stride=self.stride, padding=self.padding,
                dilation=self.dilation, groups=self.groups * B)
        out = out.permute([1, 0, 2, 3]).view(B, self.out_channels, out.shape[-2], out.shape[-1])

        # Literal port (from TF definition)
        # x = torch.split(x, 1, 0)
        # weight = torch.split(weight, 1, 0)
        # if self.bias is not None:
        #     bias = torch.matmul(routing_weights, self.bias)
        #     bias = torch.split(bias, 1, 0)
        # else:
        #     bias = [None] * B
        # out = []
        # for xi, wi, bi in zip(x, weight, bias):
        #     wi = wi.view(*self.weight_shape)
        #     if bi is not None:
        #         bi = bi.view(*self.bias_shape)
        #     out.append(self.conv_fn(
        #         xi, wi, bi, stride=self.stride, padding=self.padding,
        #         dilation=self.dilation, groups=self.groups))
        # out = torch.cat(out, 0)
        return out


class SqueezeExcite(nn.Module):
    def __init__(self, in_chs, se_ratio=0.25, reduced_base_chs=None,
                 act_layer=nn.ReLU, gate_fn=sigmoid, divisor=1, **_):
        super(SqueezeExcite, self).__init__()
        self.gate_fn = gate_fn
        reduced_chs = make_divisible((reduced_base_chs or in_chs) * se_ratio, divisor)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        self.act1 = act_layer(inplace=True)
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)

    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        x = x * self.gate_fn(x_se)
        return x


class DepthwiseSeparableConv(nn.Module):
    """ DepthwiseSeparable block
    Used for DS convs in MobileNet-V1 and in the place of IR blocks that have no expansion
    (factor of 1.0). This is an alternative to having a IR with an optional first pw conv.
    """

    def __init__(self, in_chs, out_chs, dw_kernel_size=3,
                 stride=1, dilation=1, pad_type='', act_layer=nn.ReLU, noskip=False,
                 pw_kernel_size=1, pw_act=False, se_ratio=0., se_kwargs=None,
                 norm_layer=nn.BatchNorm2d, norm_kwargs=None, drop_path_rate=0.):
        super(DepthwiseSeparableConv, self).__init__()
        norm_kwargs = norm_kwargs or {}
        has_se = se_ratio is not None and se_ratio > 0.
        self.has_residual = (stride == 1 and in_chs == out_chs) and not noskip
        self.has_pw_act = pw_act  # activation after point-wise conv
        self.drop_path_rate = drop_path_rate

        self.conv_dw = create_conv2d(
            in_chs, in_chs, dw_kernel_size, stride=stride, dilation=dilation, padding=pad_type, depthwise=True)
        self.bn1 = norm_layer(in_chs, **norm_kwargs)
        self.act1 = act_layer(inplace=True)

        # Squeeze-and-excitation
        if has_se:
            se_kwargs = resolve_se_args(se_kwargs, in_chs, act_layer)
            self.se = SqueezeExcite(in_chs, se_ratio=se_ratio, **se_kwargs)
        else:
            self.se = None

        self.conv_pw = create_conv2d(in_chs, out_chs, pw_kernel_size, padding=pad_type)
        self.bn2 = norm_layer(out_chs, **norm_kwargs)
        self.act2 = act_layer(inplace=True) if self.has_pw_act else nn.Identity()

    def feature_info(self, location):
        if location == 'expansion':
            # no expansion in this block, use depthwise, before SE
            info = dict(module='act1', hook_type='forward', num_chs=self.conv_pw.in_channels)
        elif location == 'depthwise':  # after SE
            info = dict(module='conv_pw', hook_type='forward_pre', num_chs=self.conv_pw.in_channels)
        else:  # location == 'bottleneck'
            info = dict(module='', hook_type='', num_chs=self.conv_pw.out_channels)
        return info

    def forward(self, x):
        residual = x

        x = self.conv_dw(x)
        x = self.bn1(x)
        x = self.act1(x)

        if self.se is not None:
            x = self.se(x)

        x = self.conv_pw(x)
        x = self.bn2(x)
        x = self.act2(x)

        if self.has_residual:
            x += residual
        return x


class ConvBnAct(nn.Module):
    def __init__(self, in_chs, out_chs, kernel_size,
                 stride=1, dilation=1, pad_type='', act_layer=nn.ReLU,
                 norm_layer=nn.BatchNorm2d, norm_kwargs=None):
        super(ConvBnAct, self).__init__()
        norm_kwargs = norm_kwargs or {}
        self.conv = create_conv2d(in_chs, out_chs, kernel_size, stride=stride, dilation=dilation, padding=pad_type)
        self.bn1 = norm_layer(out_chs, **norm_kwargs)
        self.act1 = act_layer(inplace=True)

    def feature_info(self, location):
        if location == 'expansion' or location == 'depthwise':
            # no expansion or depthwise this block, use act after conv
            info = dict(module='act1', hook_type='forward', num_chs=self.conv.out_channels)
        else:  # location == 'bottleneck'
            info = dict(module='', hook_type='', num_chs=self.conv.out_channels)
        return info

    def forward(self, x):
        x = self.conv(x)
        x = self.bn1(x)
        x = self.act1(x)
        return x


class InvertedResidual(nn.Module):
    """ Inverted residual block w/ optional SE and CondConv routing"""

    def __init__(self, in_chs, out_chs, dw_kernel_size=3,
                 stride=1, dilation=1, pad_type='', act_layer=nn.ReLU, noskip=False,
                 exp_ratio=1.0, exp_kernel_size=1, pw_kernel_size=1,
                 se_ratio=0., se_kwargs=None, norm_layer=nn.BatchNorm2d, norm_kwargs=None,
                 conv_kwargs=None, drop_path_rate=0.):
        super(InvertedResidual, self).__init__()
        norm_kwargs = norm_kwargs or {}
        conv_kwargs = conv_kwargs or {}
        mid_chs = make_divisible(in_chs * exp_ratio)
        has_se = se_ratio is not None and se_ratio > 0.
        self.has_residual = (in_chs == out_chs and stride == 1) and not noskip
        self.drop_path_rate = drop_path_rate

        # Point-wise expansion
        self.conv_pw = create_conv2d(in_chs, mid_chs, exp_kernel_size, padding=pad_type, **conv_kwargs)
        self.bn1 = norm_layer(mid_chs, **norm_kwargs)
        self.act1 = act_layer(inplace=True)

        # Depth-wise convolution
        self.conv_dw = create_conv2d(
            mid_chs, mid_chs, dw_kernel_size, stride=stride, dilation=dilation,
            padding=pad_type, depthwise=True, **conv_kwargs)
        self.bn2 = norm_layer(mid_chs, **norm_kwargs)
        self.act2 = act_layer(inplace=True)

        # Squeeze-and-excitation
        if has_se:
            se_kwargs = resolve_se_args(se_kwargs, in_chs, act_layer)
            self.se = SqueezeExcite(mid_chs, se_ratio=se_ratio, **se_kwargs)
        else:
            self.se = None

        # Point-wise linear projection
        self.conv_pwl = create_conv2d(mid_chs, out_chs, pw_kernel_size, padding=pad_type, **conv_kwargs)
        self.bn3 = norm_layer(out_chs, **norm_kwargs)

    def feature_info(self, location):
        if location == 'expansion':
            info = dict(module='act1', hook_type='forward', num_chs=self.conv_pw.in_channels)
        elif location == 'depthwise':  # after SE
            info = dict(module='conv_pwl', hook_type='forward_pre', num_chs=self.conv_pwl.in_channels)
        else:  # location == 'bottleneck'
            info = dict(module='', hook_type='', num_chs=self.conv_pwl.out_channels)
        return info

    def forward(self, x):
        residual = x

        # Point-wise expansion
        x = self.conv_pw(x)
        x = self.bn1(x)
        x = self.act1(x)

        # Depth-wise convolution
        x = self.conv_dw(x)
        x = self.bn2(x)
        x = self.act2(x)

        # Squeeze-and-excitation
        if self.se is not None:
            x = self.se(x)

        # Point-wise linear projection
        x = self.conv_pwl(x)
        x = self.bn3(x)

        if self.has_residual:
            x += residual

        return x


class ChildNetBuilder:
    """ Build Trunk Blocks
    """

    def __init__(self, channel_multiplier=1.0, channel_divisor=8, channel_min=None,
                 output_stride=32, pad_type='', act_layer=None, se_kwargs=None,
                 norm_layer=nn.BatchNorm2d, norm_kwargs=None, drop_path_rate=0., feature_location='',
                 verbose=False):
        self.channel_multiplier = channel_multiplier
        self.channel_divisor = channel_divisor
        self.channel_min = channel_min
        self.output_stride = output_stride
        self.pad_type = pad_type
        self.act_layer = act_layer
        self.se_kwargs = se_kwargs
        self.norm_layer = norm_layer
        self.norm_kwargs = norm_kwargs
        self.drop_path_rate = drop_path_rate
        self.feature_location = feature_location
        assert feature_location in ('pre_pwl', 'post_exp', '')
        self.verbose = verbose

        # state updated during build, consumed by model
        self.in_chs = None
        self.features = OrderedDict()

    def _round_channels(self, chs):
        return round_channels(chs, self.channel_multiplier, self.channel_divisor, self.channel_min)

    def _make_block(self, ba, block_idx, block_count):
        drop_path_rate = self.drop_path_rate * block_idx / block_count
        bt = ba.pop('block_type')
        ba['in_chs'] = self.in_chs
        ba['out_chs'] = self._round_channels(ba['out_chs'])
        if 'fake_in_chs' in ba and ba['fake_in_chs']:
            # FIXME this is a hack to work around mismatch in origin impl input filters
            ba['fake_in_chs'] = self._round_channels(ba['fake_in_chs'])
        ba['norm_layer'] = self.norm_layer
        ba['norm_kwargs'] = self.norm_kwargs
        ba['pad_type'] = self.pad_type
        # block act fn overrides the model default
        ba['act_layer'] = ba['act_layer'] if ba['act_layer'] is not None else self.act_layer
        assert ba['act_layer'] is not None
        if bt == 'ir':
            ba['drop_path_rate'] = drop_path_rate
            ba['se_kwargs'] = self.se_kwargs
            block = InvertedResidual(**ba)
        elif bt == 'ds' or bt == 'dsa':
            ba['drop_path_rate'] = drop_path_rate
            ba['se_kwargs'] = self.se_kwargs
            block = DepthwiseSeparableConv(**ba)
        elif bt == 'cn':
            block = ConvBnAct(**ba)
        else:
            assert False, 'Uknkown block type (%s) while building model.' % bt
        self.in_chs = ba['out_chs']  # update in_chs for arg of next block

        return block

    def __call__(self, in_chs, model_block_args):
        """ Build the blocks
        Args:
            in_chs: Number of input-channels passed to first block
            model_block_args: A list of lists, outer list defines stages, inner
                list contains strings defining block configuration(s)
        Return:
             List of block stacks (each stack wrapped in nn.Sequential)
        """
        self.in_chs = in_chs
        total_block_count = sum([len(x) for x in model_block_args])
        total_block_idx = 0
        current_stride = 2
        current_dilation = 1
        feature_idx = 0
        stages = []
        # outer list of block_args defines the stacks ('stages' by some conventions)
        for stage_idx, stage_block_args in enumerate(model_block_args):
            last_stack = stage_idx == (len(model_block_args) - 1)
            assert isinstance(stage_block_args, list)

            blocks = []
            # each stack (stage) contains a list of block arguments
            for block_idx, block_args in enumerate(stage_block_args):
                last_block = block_idx == (len(stage_block_args) - 1)
                extract_features = ''  # No features extracted

                # Sort out stride, dilation, and feature extraction details
                assert block_args['stride'] in (1, 2)
                if block_idx >= 1:
                    # only the first block in any stack can have a stride > 1
                    block_args['stride'] = 1

                do_extract = False
                if self.feature_location == 'pre_pwl':
                    if last_block:
                        next_stage_idx = stage_idx + 1
                        if next_stage_idx >= len(model_block_args):
                            do_extract = True
                        else:
                            do_extract = model_block_args[next_stage_idx][0]['stride'] > 1
                elif self.feature_location == 'post_exp':
                    if block_args['stride'] > 1 or (last_stack and last_block):
                        do_extract = True
                if do_extract:
                    extract_features = self.feature_location

                next_dilation = current_dilation
                if block_args['stride'] > 1:
                    next_output_stride = current_stride * block_args['stride']
                    if next_output_stride > self.output_stride:
                        next_dilation = current_dilation * block_args['stride']
                        block_args['stride'] = 1
                    else:
                        current_stride = next_output_stride
                block_args['dilation'] = current_dilation
                if next_dilation != current_dilation:
                    current_dilation = next_dilation

                # create the block
                block = self._make_block(block_args, total_block_idx, total_block_count)
                blocks.append(block)

                # stash feature module name and channel info for model feature extraction
                if extract_features:
                    feature_module = block.feature_module(extract_features)
                    if feature_module:
                        feature_module = 'blocks.{}.{}.'.format(stage_idx, block_idx) + feature_module
                    feature_channels = block.feature_channels(extract_features)
                    self.features[feature_idx] = dict(
                        name=feature_module,
                        num_chs=feature_channels
                    )
                    feature_idx += 1

                total_block_idx += 1  # incr global block idx (across all stacks)
            stages.append(nn.Sequential(*blocks))
        return stages


class LightBackbone(nn.Module):
    def __init__(self, block_args, in_chans=3, stem_size=16,
                 channel_multiplier=1.0, pad_type='', drop_path_rate=0.,
                 norm_layer=nn.BatchNorm2d, zero_gamma=False):
        super(LightBackbone, self).__init__()

        act_layer = Swish
        norm_kwargs = {}
        se_kwargs = {'act_layer': nn.ReLU, 'divisor': 8, 'reduce_mid': True, 'gate_fn': hard_sigmoid}

        self._in_chs = in_chans
        # Stem
        stem_size = round_channels(stem_size, channel_multiplier)
        self.conv_stem = create_conv2d(self._in_chs, stem_size, 3, stride=2, padding=pad_type)
        self.bn1 = norm_layer(stem_size, **norm_kwargs)
        self.act1 = act_layer(inplace=True)
        self._in_chs = stem_size

        # Middle stages (IR/ER/DS Blocks)
        builder = ChildNetBuilder(
            channel_multiplier, 8, None, 32, pad_type, act_layer, se_kwargs,
            norm_layer, norm_kwargs, drop_path_rate, verbose=False)
        self.blocks = nn.Sequential(*builder(self._in_chs, block_args))
        self._in_chs = builder.in_chs

        efficientnet_init_weights(self, zero_gamma=zero_gamma)
        self.strides = [2, 4, 8, 16, 16, 32, 32]

    def forward(self, x):
        # architecture = [[0], [], [], [], [], [], [0]]
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.blocks(x)
        return x


def light_backbone(setting_path):
    block_args = np.load(setting_path, allow_pickle=True)
    model = LightBackbone(block_args)
    return model


if __name__ == '__main__':
    from pysot.utils.model_load import load_pretrain

    block_args = np.load('pretrained_models/settings.npy', allow_pickle=True)
    net = LightBackbone(block_args).eval().cuda()
    load_pretrain(net, 'light_backbone.pth')

    a = torch.ones((1, 3, 128, 128)).cuda()
    b = net(a)
    c = 0
    pass

# class BN_adj(nn.Module):
#     def __init__(self, num_channel):
#         super(BN_adj, self).__init__()
#         self.BN_z = nn.BatchNorm2d(num_channel)
#         self.BN_x = nn.BatchNorm2d(num_channel)
#
#     def forward(self, zf, xf):
#         return self.BN_z(zf), self.BN_x(xf)
#
#
# def build_subnet_BN(path_ops, model_cfg):
#     inc_idx = model_cfg.stage_idx.index(path_ops[0])
#     num_channel = model_cfg.in_c[inc_idx]
#     return BN_adj(num_channel)
#
#
# def pixel_corr(z, x):
#     """Pixel-wise correlation (implementation by for-loop and convolution)
#     The speed is slower because the for-loop"""
#     size = z.size()  # (bs, c, hz, wz)
#     CORR = []
#     for i in range(len(x)):
#         ker = z[i:i + 1]  # (1, c, hz, wz)
#         fea = x[i:i + 1]  # (1, c, hx, wx)
#         ker = ker.view(size[1], size[2] * size[3]).transpose(0, 1)  # (hz * wz, c)
#         ker = ker.unsqueeze(2).unsqueeze(3)  # (hz * wz, c, 1, 1)
#         co = F.conv2d(fea, ker.contiguous())  # (1, hz * wz, hx, wx)
#         CORR.append(co)
#     corr = torch.cat(CORR, 0)  # (bs, hz * wz, hx, wx)
#     return corr
#
#
# def pixel_corr_mat(z, x):
#     """Pixel-wise correlation (implementation by matrix multiplication)
#     The speed is faster because the computation is vectorized"""
#     b, c, h, w = x.size()
#     z_mat = z.view((b, c, -1)).transpose(1, 2)  # (b, hz * wz, c)
#     x_mat = x.view((b, c, -1))  # (b, c, hx * wx)
#     return torch.matmul(z_mat, x_mat).view((b, -1, h, w))  # (b, hz * wz, hx * wx) --> (b, hz * wz, hx, wx)
#
#
# class CAModule(nn.Module):
#     """Channel attention module"""
#
#     def __init__(self, channels=64, reduction=1):
#         super(CAModule, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,
#                              padding=0)
#         self.relu = nn.ReLU(inplace=True)
#         self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,
#                              padding=0)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         module_input = x
#         x = self.avg_pool(x)
#         x = self.fc1(x)
#         x = self.relu(x)
#         x = self.fc2(x)
#         x = self.sigmoid(x)
#         return module_input * x
#
#
# class PWCA(nn.Module):
#     """
#     Pointwise Correlation & Channel Attention
#     """
#
#     def __init__(self, num_channel, cat=False, CA=True, matrix=False):
#         super(PWCA, self).__init__()
#         self.cat = cat
#         self.CA = CA
#         self.matrix = matrix
#         if self.CA:
#             self.CA_layer = CAModule(channels=num_channel)
#
#     def forward(self, z, x):
#         z11 = z[0]
#         x11 = x[0]
#         # pixel-wise correlation
#         if self.matrix:
#             corr = pixel_corr_mat(z11, x11)
#         else:
#             corr = pixel_corr(z11, x11)
#         if self.CA:
#             # channel attention
#             opt = self.CA_layer(corr)
#             if self.cat:
#                 return torch.cat([opt, x11], dim=1)
#             else:
#                 return opt
#         else:
#             return corr
#
#
# class PW_Corr_adj(nn.Module):
#     def __init__(self, num_kernel=64, cat=False, matrix=True, adj_channel=128):
#         super(PW_Corr_adj, self).__init__()
#         self.pw_corr = PWCA(num_kernel, cat=cat, CA=True, matrix=matrix)
#         self.adj_layer = nn.Conv2d(num_kernel, adj_channel, 1)
#
#     def forward(self, kernel, search):
#         '''stride_idx: 0 or 1. 0 represents stride 8. 1 represents stride 16'''
#         oup = {}
#         corr_feat = self.pw_corr([kernel], [search])
#         corr_feat = self.adj_layer(corr_feat)
#         oup['cls'], oup['reg'] = corr_feat, corr_feat
#         return oup
#
#
# def build_subnet_feat_fusor(path_ops, model_cfg, cat=False, matrix=True, adj_channel=128):
#     stride = model_cfg.strides[path_ops[0]]
#     stride_idx = model_cfg.strides_use_new.index(stride)
#     num_kernel = model_cfg.num_kernel_corr[stride_idx]
#     return PW_Corr_adj(num_kernel=num_kernel, cat=cat, matrix=matrix, adj_channel=adj_channel)
#
#
# class head_subnet(nn.Module):
#     def __init__(self, module_dict):
#         super(head_subnet, self).__init__()
#         self.cls_tower = module_dict['cls_tower']
#         self.reg_tower = module_dict['reg_tower']
#         self.cls_perd = module_dict['cls_pred']
#         self.reg_pred = module_dict['reg_pred']
#
#     def forward(self, inp):
#         oup = {}
#         # cls
#         cls_feat = self.cls_tower(inp['cls'])
#         oup['cls'] = self.cls_perd(cls_feat)
#         # reg
#         reg_feat = self.reg_tower(inp['reg'])
#         oup['reg'] = self.reg_pred(reg_feat)
#         return oup
#
#
# class cls_pred_head(nn.Module):
#     def __init__(self, inchannels=256):
#         super(cls_pred_head, self).__init__()
#         self.cls_pred = nn.Conv2d(inchannels, 1, kernel_size=3, stride=1, padding=1)
#
#     def forward(self, x):
#         """mode should be in ['all', 'cls', 'reg']"""
#         x = 0.1 * self.cls_pred(x)
#         return x
#
#
# class reg_pred_head(nn.Module):
#     def __init__(self, inchannels=256, linear_reg=False, stride=16):
#         super(reg_pred_head, self).__init__()
#         self.linear_reg = linear_reg
#         self.stride = stride
#         # reg head
#         self.bbox_pred = nn.Conv2d(inchannels, 4, kernel_size=3, stride=1, padding=1)
#         # adjust scale
#         if not self.linear_reg:
#             self.adjust = nn.Parameter(0.1 * torch.ones(1))
#             self.bias = nn.Parameter(torch.Tensor(1.0 * torch.ones(1, 4, 1, 1)).cuda())
#
#     def forward(self, x):
#         if self.linear_reg:
#             x = nn.functional.relu(self.bbox_pred(x)) * self.stride
#         else:
#             x = self.adjust * self.bbox_pred(x) + self.bias
#             x = torch.exp(x)
#         return x
#
#
# class SeparableConv2d_BNReLU(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
#         super(SeparableConv2d_BNReLU, self).__init__()
#
#         self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels,
#                                bias=bias)
#         self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)
#         self.BN = nn.BatchNorm2d(out_channels)
#         self.ReLU = nn.ReLU()
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.pointwise(x)
#         x = self.ReLU(self.BN(x))
#         return x
#
#
# def get_towers(module_list: torch.nn.ModuleList, path_head, inchannels, outchannels, towernum=8, kernel_list=[3, 5, 0]):
#     num_choice_kernel = len(kernel_list)
#     for tower_idx in range(towernum):
#         block_idx = path_head[1][tower_idx]
#         kernel_sz = kernel_list[block_idx]
#         if tower_idx == 0:
#             assert (kernel_sz != 0)
#             padding = (kernel_sz - 1) // 2
#             module_list.append(SeparableConv2d_BNReLU(inchannels, outchannels, kernel_size=kernel_sz,
#                                                       stride=1, padding=padding, dilation=1))
#         else:
#             if block_idx != num_choice_kernel - 1:  # else skip
#                 assert (kernel_sz != 0)
#                 padding = (kernel_sz - 1) // 2
#                 module_list.append(SeparableConv2d_BNReLU(outchannels, outchannels, kernel_size=kernel_sz,
#                                                           stride=1, padding=padding, dilation=1))
#     return module_list
#
#
# def build_subnet_head(path_head, channel_list=[128, 192, 256], kernel_list=[3, 5, 0], inchannels=64, towernum=8,
#                       linear_reg=False):
#     channel_idx_cls, channel_idx_reg = path_head['cls'][0], path_head['reg'][0]
#     num_channel_cls, num_channel_reg = channel_list[channel_idx_cls], channel_list[channel_idx_reg]
#     tower_cls_list = nn.ModuleList()
#     tower_reg_list = nn.ModuleList()
#     # add operations
#     tower_cls = nn.Sequential(
#         *get_towers(tower_cls_list, path_head['cls'], inchannels, num_channel_cls, towernum=towernum,
#                     kernel_list=kernel_list))
#     tower_reg = nn.Sequential(
#         *get_towers(tower_reg_list, path_head['reg'], inchannels, num_channel_reg, towernum=towernum,
#                     kernel_list=kernel_list))
#     # add prediction head
#     cls_pred = cls_pred_head(inchannels=num_channel_cls)
#     reg_pred = reg_pred_head(inchannels=num_channel_reg, linear_reg=linear_reg)
#
#     module_dict = {'cls_tower': tower_cls, 'reg_tower': tower_reg, 'cls_pred': cls_pred, 'reg_pred': reg_pred}
#     return head_subnet(module_dict)

