

import torch
import torch.nn as nn
import torch.nn.functional as F

from pysot.models.init_weight import xavier_fill, gauss_fill
from pysot.models.head.rpn_head import DepthwiseXCorr
from pysot.models.init_weight import kaiming_init, constant_init
# from pysot.models.head.dcn.deform_conv import DeformConvPack, ModulatedDeformConvPack
# from pysot.models.head.dcn.deform_pool import DeformRoIPoolingPack, ModulatedDeformRoIPoolingPack
from mmcv.ops.deform_conv import DeformConv2dPack
from mmcv.ops.deform_roi_pool import DeformRoIPoolPack

import warnings


class ConvModule(nn.Module):
    """Conv-Norm-Activation block.
    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int or tuple[int]): Same as nn.Conv2d.
        padding (int or tuple[int]): Same as nn.Conv2d.
        dilation (int or tuple[int]): Same as nn.Conv2d.
        groups (int): Same as nn.Conv2d.
        bias (bool or str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
            False.
        conv_cfg (dict): Config dict for convolution layer.
        norm_cfg (dict): Config dict for normalization layer.
        activation (str or None): Activation type, "ReLU" by default.
        inplace (bool): Whether to use inplace mode for activation.
        activate_last (bool): Whether to apply the activation layer in the
            last. (Do not use this flag since the behavior and api may be
            changed in the future.)
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias='auto',
                 norm=True,
                 activation='relu',
                 inplace=True,
                 activate_last=True):
        super(ConvModule, self).__init__()
        self.activation = activation
        self.inplace = inplace
        self.activate_last = activate_last

        self.with_norm = norm
        self.with_activatation = activation is not None
        # if the conv layer is before a norm layer, bias is unnecessary.
        if bias == 'auto':
            bias = False if self.with_norm else True
        self.with_bias = bias

        if self.with_norm and self.with_bias:
            warnings.warn('ConvModule has norm and bias at the same time')

        # build convolution layer
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias)
        # export the attributes of self.conv to a higher level for convenience
        self.in_channels = self.conv.in_channels
        self.out_channels = self.conv.out_channels
        self.kernel_size = self.conv.kernel_size
        self.stride = self.conv.stride
        self.padding = self.conv.padding
        self.dilation = self.conv.dilation
        self.transposed = self.conv.transposed
        self.output_padding = self.conv.output_padding
        self.groups = self.conv.groups

        # build normalization layers
        if self.with_norm:
            norm_channels = out_channels if self.activate_last else in_channels
            self.norm_name = "BN" + str(norm_channels)
            norm = nn.BatchNorm2d(norm_channels)
            self.add_module(self.norm_name, norm)

        # build activation layer
        if self.with_activatation:
            if self.activation not in ['relu']:
                raise ValueError('{} is currently not supported.'.format(
                    self.activation))
            if self.activation == 'relu':
                self.activate = nn.ReLU(inplace=inplace)

        # Use msra init by default
        self.init_weights()

    @property
    def norm(self):
        return getattr(self, self.norm_name)

    def init_weights(self):
        nonlinearity = 'relu' if self.activation is None else self.activation
        kaiming_init(self.conv, nonlinearity=nonlinearity)
        if self.with_norm:
            constant_init(self.norm, 1, bias=0)

    def forward(self, x, activate=True, norm=True):
        if self.activate_last:
            x = self.conv(x)
            if norm and self.with_norm:
                x = self.norm(x)
            if activate and self.with_activatation:
                x = self.activate(x)
        else:
            # WARN: this may be removed or modified
            if norm and self.with_norm:
                x = self.norm(x)
            if activate and self.with_activatation:
                x = self.activate(x)
            x = self.conv(x)
        return x


class FeatureEnhance(nn.Module):
    def __init__(self,
                 in_channels=256,
                 out_channels=256):
        super(FeatureEnhance, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.pams = nn.ModuleList()
        self.cam_cals = nn.ModuleList()
        self.cam_uses = nn.ModuleList()
        self.deform_convs = nn.ModuleList()
        for i in range(3):
            # self.deform_convs.append(DeformConvPack(in_channels=self.in_channels,
            #                                         out_channels=self.out_channels,
            #                                         kernel_size=3,
            #                                         padding=1))
            self.deform_convs.append(DeformConv2dPack(in_channels=self.in_channels,
                                                      out_channels=self.out_channels,
                                                      kernel_size=3,
                                                      padding=1))

            # self.deform_convs.append(nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, padding=1))
            self.pams.append(PAM_Module(self.in_channels))
            self.cam_cals.append(CAM_Calculate(self.in_channels))
            self.cam_uses.append(CAM_Use(self.in_channels))

    def forward(self, z_fs, x_fs):
        z_out = []
        x_out = []

        for idx, (z_f, x_f) in enumerate(zip(z_fs, x_fs)):
            z_sp_feat = self.pams[idx](z_f)
            x_sp_feat = self.pams[idx](x_f)

            z_attention = self.cam_cals[idx](z_f)
            x_attention = self.cam_cals[idx](x_f)

            zz_sc_feat = self.cam_uses[idx](z_f, z_attention)
            xx_sc_feat = self.cam_uses[idx](x_f, x_attention)

            zx_sc_feat = self.cam_uses[idx](z_f, x_attention)
            xz_sc_feat = self.cam_uses[idx](x_f, z_attention)

            z_f = z_sp_feat + zz_sc_feat + zx_sc_feat
            x_f = x_sp_feat + xx_sc_feat + xz_sc_feat

            z_f = self.deform_convs[idx](z_f)
            x_f = self.deform_convs[idx](x_f)

            z_out.append(z_f)
            x_out.append(x_f)

        return z_out, x_out


class PAM_Module(nn.Module):
    """ Position attention module"""

    # Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)
        self.init_weights()

    def init_weights(self):
        kaiming_init(self.query_conv)
        kaiming_init(self.key_conv)
        kaiming_init(self.value_conv)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out


class CAM_Calculate(nn.Module):
    """ Channel attention module"""

    def __init__(self, in_dim):
        super(CAM_Calculate, self).__init__()
        self.chanel_in = in_dim
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.contiguous().view(m_batchsize, C, -1)
        proj_key = x.contiguous().view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)

        return attention


class CAM_Use(nn.Module):
    """ Channel attention module"""

    def __init__(self, in_dim):
        super(CAM_Use, self).__init__()
        self.chanel_in = in_dim
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x, attention):
        """
            inputs :
                x : input feature maps( B X C X H X W)
                attention: B X C X C
            returns :
                out : attention value + input feature
        """
        m_batchsize, C, height, width = x.size()
        proj_value = x.contiguous().view(m_batchsize, C, -1)
        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)
        out = self.gamma * out + x
        return out


class FeatureFusionNeck(nn.Module):
    def __init__(self,
                 num_ins,
                 fusion_level,
                 in_channels=[64, 256, 256, 256, 256],
                 conv_out_channels=256):
        super(FeatureFusionNeck, self).__init__()
        self.num_ins = num_ins
        self.fusion_level = fusion_level
        self.in_channels = in_channels
        self.conv_out_channels = conv_out_channels

        assert num_ins == len(in_channels), "num_ins must equal to length of in_channel."

        self.lateral_convs = nn.ModuleList()
        for i in range(self.num_ins):
            self.lateral_convs.append(
                ConvModule(
                    self.in_channels[i],
                    self.conv_out_channels,
                    1,
                    inplace=False))

        self.xcorr = nn.ModuleList()
        for i in range(len(self.in_channels[2:])):
            self.xcorr.append(DepthwiseXCorr(self.in_channels[i+2],
                                             self.in_channels[i+2],
                                             self.conv_out_channels))

        self.convs = nn.ModuleList()
        for i in range(len(self.in_channels[2:])):
            self.convs.append(
                ConvModule(
                    self.in_channels[i+2],
                    self.conv_out_channels,
                    1,
                    inplace=False))

    def forward(self, z_fs, x_fs):
        m_feats = x_fs
        b_feats = []
        for idx, (z_f, x_f) in enumerate(zip(z_fs[2:], x_fs[2:])):
            f = self.xcorr[idx](z_f, x_f)
            b_feats.append(f)

        x = self.lateral_convs[self.fusion_level](m_feats[self.fusion_level])
        fused_size = tuple(x.shape[-2:])
        for i, feat in enumerate(m_feats):
            if i != self.fusion_level:
                feat = F.interpolate(
                    feat, size=fused_size, mode='bilinear', align_corners=True)
                x += self.lateral_convs[i](feat)

        y = self.convs[0](b_feats[0])
        for i, feat in enumerate(b_feats[1:], start=1):
            y += self.convs[i](feat)
        return y, x


class FCx2DetHead(nn.Module):
    def __init__(self, cfg, pooling_func, in_channels, fc_channels=512):
        super(FCx2DetHead, self).__init__()
        self.in_chns = in_channels
        self.tail_det = nn.Sequential(
            nn.Linear(in_channels, fc_channels),
            nn.ReLU(inplace=True),
            nn.Linear(fc_channels, fc_channels),
            nn.ReLU(inplace=True)
        )
        self.tail_det_box = nn.Linear(fc_channels, 4)
        self.pooling_func = pooling_func

        # from torchvision.ops import RoIPool, RoIAlign
        # self.roi_pool_b = DeformRoIPoolingPack(spatial_scale=1 / (cfg.TRAIN.SEARCH_SIZE / 25),
        #                                        out_size=cfg.TRAIN.ROIPOOL_OUTSIZE // 4,
        #                                        out_channels=256,
        #                                        no_trans=False,
        #                                        trans_std=0.1)
        self.roi_pool_b = DeformRoIPoolPack(spatial_scale=1 / (cfg.TRAIN.SEARCH_SIZE / 25),
                                            output_size=cfg.TRAIN.ROIPOOL_OUTSIZE // 4,
                                            output_channels=256,
                                            gamma=0.1)
        '''
        self.roi_pool_b = ModulatedDeformRoIPoolingPack(spatial_scale=1 / (cfg.TRAIN.SEARCH_SIZE / 25),
                                               out_size=cfg.TRAIN.ROIPOOL_OUTSIZE // 4,
                                               out_channels=256,
                                               no_trans=False,
                                               trans_std=0.1,
                                               deform_fc_channels=512)

        self.roi_pool_b = ops.RoIAlign(output_size=(cfg.TRAIN.ROIPOOL_OUTSIZE // 4, cfg.TRAIN.ROIPOOL_OUTSIZE // 4),
                                     spatial_scale=1 / (cfg.TRAIN.SEARCH_SIZE / 25),
                                     sampling_ratio=-1)
        '''

        # init parameters
        xavier_fill(self.tail_det)
        gauss_fill(self.tail_det_box, std=0.001)

    def forward(self, x, roi_list):
        x = self.roi_pool_b(x, roi_list)
        x = x.view(-1, self.in_chns)
        x = self.tail_det(x)
        bbox_pred = self.tail_det_box(x)
        return bbox_pred
