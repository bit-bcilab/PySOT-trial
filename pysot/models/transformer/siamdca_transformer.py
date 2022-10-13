

import numpy as np
import torch
import torch.nn as nn
from pysot.models.transformer.utils import _get_clones


def get_angles(pos, i, model_dims):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(model_dims))
    return pos * angle_rates


def positionalencoding(seq_length, model_dims):
    angle_rads = get_angles(np.arange(seq_length)[:, None], np.arange(model_dims)[None, :], model_dims)
    # 第2i项使用sin
    sines = np.sin(angle_rads[:, 0::2])
    # 第2i+1项使用cos
    cones = np.cos(angle_rads[:, 1::2])
    pos_encoding = np.stack([sines, cones], axis=-1)
    pos_encoding = np.reshape(pos_encoding, (seq_length, -1)).astype(np.float32)[:, None, :]
    pos_encoding = torch.from_numpy(pos_encoding).cuda()
    return pos_encoding


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, act=nn.ReLU()):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.act = act
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class ConvUnit(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size, bias=True, stride=1, padding='same', act=None):
        super().__init__()
        if act is None:
            self.conv_uint = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                          stride=stride, padding=padding, bias=bias),
                nn.BatchNorm2d(out_channels))
        else:
            self.conv_uint = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                          stride=stride, padding=padding, bias=bias),
                nn.BatchNorm2d(out_channels),
                act)

    def forward(self, x):
        return self.conv_uint(x)
    
    
class FeatureFusionLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, act=nn.GELU()):
        super().__init__()
        self.act = act

        self.self_attn1 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn1 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.linear11 = nn.Linear(d_model, dim_feedforward)
        self.dropout1 = nn.Dropout(dropout)
        self.linear12 = nn.Linear(dim_feedforward, d_model)

        self.norm11 = nn.LayerNorm(d_model)
        self.norm12 = nn.LayerNorm(d_model)
        self.norm13 = nn.LayerNorm(d_model)
        self.dropout11 = nn.Dropout(dropout)
        self.dropout12 = nn.Dropout(dropout)
        self.dropout13 = nn.Dropout(dropout)

        self.self_attn2 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn2 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear21 = nn.Linear(d_model, dim_feedforward)
        self.dropout2 = nn.Dropout(dropout)
        self.linear22 = nn.Linear(dim_feedforward, d_model)
        self.norm21 = nn.LayerNorm(d_model)
        self.norm22 = nn.LayerNorm(d_model)
        self.norm23 = nn.LayerNorm(d_model)
        self.dropout21 = nn.Dropout(dropout)
        self.dropout22 = nn.Dropout(dropout)
        self.dropout23 = nn.Dropout(dropout)

    def with_pos_embed(self, tensor, pos=None):
        return tensor if pos is None else tensor + pos

    def forward(self, src1, src2, pos_src1, pos_src2):
        q1 = k1 = self.with_pos_embed(src1, pos_src1)
        src12 = self.self_attn1(q1, k1, value=src1)[0]
        src1 = src1 + self.dropout11(src12)
        src1 = self.norm11(src1)

        q2 = k2 = self.with_pos_embed(src2, pos_src2)
        src22 = self.self_attn2(q2, k2, value=src2)[0]
        src2 = src2 + self.dropout21(src22)
        src2 = self.norm21(src2)

        src12 = self.multihead_attn1(query=self.with_pos_embed(src1, pos_src1),
                                     key=self.with_pos_embed(src2, pos_src2),
                                     value=src2)[0]
        src1 = src1 + self.dropout12(src12)
        src1 = self.norm12(src1)
        src12 = self.linear12(self.dropout1(self.act(self.linear11(src1))))
        src1 = src1 + self.dropout13(src12)
        src1 = self.norm13(src1)

        src22 = self.multihead_attn2(query=self.with_pos_embed(src2, pos_src2),
                                     key=self.with_pos_embed(src1, pos_src1),
                                     value=src1)[0]
        src2 = src2 + self.dropout22(src22)
        src2 = self.norm22(src2)
        src22 = self.linear22(self.dropout2(self.act(self.linear21(src2))))
        src2 = src2 + self.dropout23(src22)
        src2 = self.norm23(src2)
        return src1, src2


class Encoder(nn.Module):
    def __init__(self, featurefusion_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(featurefusion_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, src1, src2, pos_src1=None, pos_src2=None):
        output1 = src1
        output2 = src2

        for layer in self.layers:
            output1, output2 = layer(output1, output2, pos_src1=pos_src1, pos_src2=pos_src2)
        return output1, output2


class Decoder(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, act=nn.GELU()):
        super().__init__()

        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.act = act

    def with_pos_embed(self, tensor, pos=None):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory, pos_enc=None, pos_dec=None):
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, pos_dec),
                                   key=self.with_pos_embed(memory, pos_enc),
                                   value=memory)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.linear2(self.dropout(self.act(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        return tgt


class Transformer(nn.Module):
    def __init__(self, cfg, d_model=512, nhead=8, num_featurefusion_layers=4,
                 dim_feedforward=2048, dropout=0.1, act=nn.GELU()):
        super().__init__()
        self.s_z = int(cfg.TRAIN.EXEMPLAR_SIZE / cfg.POINT.STRIDE)
        self.s_x = int(cfg.TRAIN.SEARCH_SIZE / cfg.POINT.STRIDE)

        self.input_proj = nn.Conv2d(d_model, d_model, kernel_size=(1, 1))

        featurefusion_layer = FeatureFusionLayer(d_model, nhead, dim_feedforward, dropout, act)
        self.encoder = Encoder(featurefusion_layer, num_featurefusion_layers)
        self.decoder = Decoder(d_model, nhead, dim_feedforward, dropout, act)

        self.class_embed = MLP(d_model, d_model, 2, 3)
        # self.class_embed = MLP(d_model * 2, d_model, 2, 3)
        self.bbox_embed = MLP(d_model * 2, d_model, 4, 3)
        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

        self.pos_temp = positionalencoding(self.s_z ** 2, model_dims=self.d_model)
        self.pos_search = positionalencoding(self.s_x ** 2, model_dims=self.d_model)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, xf, zf):
        src_temp = self.input_proj(zf)
        src_search = self.input_proj(xf)
        src_temp = src_temp.flatten(2).permute(2, 0, 1).contiguous()
        src_search = src_search.flatten(2).permute(2, 0, 1).contiguous()
        xf_ = src_search.transpose(1, 0).contiguous()

        memory_temp, memory_search = self.encoder(src1=src_temp, src2=src_search,
                                                  pos_src1=self.pos_temp, pos_src2=self.pos_search)
        hs = self.decoder(memory_search, memory_temp,
                          pos_enc=self.pos_temp, pos_dec=self.pos_search)
        hs = hs.transpose(1, 0).contiguous()

        cls = self.class_embed(hs)
        loc = self.bbox_embed(torch.cat((xf_, hs), dim=-1))
        # cls = self.class_embed(torch.cat((xf_, hs), dim=-1))
        # loc = self.bbox_embed(torch.cat((xf_, hs), dim=-1)).sigmoid()

        cls = cls.transpose(2, 1).contiguous().view((-1, 2, self.s_x, self.s_x)).contiguous()
        loc = loc.transpose(2, 1).contiguous().view((-1, 4, self.s_x, self.s_x)).contiguous()
        return cls, loc


# class DCAEncoderLayer(nn.Module):
#     def __init__(self, num, d_model, nhead, dim_feedforward=2048, dropout=0.1, act=nn.GELU()):
#         super().__init__()
#         self.act = act
#
#         self.self_attn1 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
#         self.norm11 = nn.LayerNorm(d_model)
#         self.norm13 = nn.LayerNorm(d_model)
#         self.dropout11 = nn.Dropout(dropout)
#         self.dropout13 = nn.Dropout(dropout)
#         self.linear11 = nn.Linear(d_model, dim_feedforward)
#         self.dropout1 = nn.Dropout(dropout)
#         self.linear12 = nn.Linear(dim_feedforward, d_model)
#
#         self.self_attn2 = nn.MultiheadAttention(num, nhead, dropout=dropout)
#         self.norm21 = nn.LayerNorm(num)
#         self.norm23 = nn.LayerNorm(num)
#         self.dropout21 = nn.Dropout(dropout)
#         self.dropout23 = nn.Dropout(dropout)
#         self.linear21 = nn.Linear(num, dim_feedforward)
#         self.dropout2 = nn.Dropout(dropout)
#         self.linear22 = nn.Linear(dim_feedforward, num)
#
#     def with_pos_embed(self, tensor, pos=None):
#         return tensor if pos is None else tensor + pos
#
#     def forward(self, src1, src2, pos_src1, pos_src2):
#         q1 = k1 = self.with_pos_embed(src1, pos_src1)
#         src12 = self.self_attn1(q1, k1, value=src1)[0]
#         src1 = src1 + self.dropout11(src12)
#         src1 = self.norm11(src1)
#         src12 = self.linear12(self.dropout1(self.act(self.linear11(src1))))
#         src1 = src1 + self.dropout13(src12)
#         src1 = self.norm13(src1)
#
#         q2 = k2 = self.with_pos_embed(src2, pos_src2)
#         src22 = self.self_attn2(q2, k2, value=src2)[0]
#         src2 = src2 + self.dropout21(src22)
#         src2 = self.norm21(src2)
#         src22 = self.linear22(self.dropout2(self.act(self.linear21(src2))))
#         src2 = src2 + self.dropout23(src22)
#         src2 = self.norm23(src2)
#         return src1, src2
class DCAEncoderLayer(nn.Module):
    def __init__(self, num, d_model, nhead, dim_feedforward=2048, dropout=0.1, act=nn.GELU()):
        super().__init__()
        self.act = act

        self.self_attn1 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn1 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.linear11 = nn.Linear(d_model, dim_feedforward)
        self.dropout1 = nn.Dropout(dropout)
        self.linear12 = nn.Linear(dim_feedforward, d_model)

        self.norm11 = nn.LayerNorm(d_model)
        self.norm12 = nn.LayerNorm(d_model)
        self.norm13 = nn.LayerNorm(d_model)
        self.dropout11 = nn.Dropout(dropout)
        self.dropout12 = nn.Dropout(dropout)
        self.dropout13 = nn.Dropout(dropout)

        self.self_attn2 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn2 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear21 = nn.Linear(d_model, dim_feedforward)
        self.dropout2 = nn.Dropout(dropout)
        self.linear22 = nn.Linear(dim_feedforward, d_model)
        self.norm21 = nn.LayerNorm(d_model)
        self.norm22 = nn.LayerNorm(d_model)
        self.norm23 = nn.LayerNorm(d_model)
        self.dropout21 = nn.Dropout(dropout)
        self.dropout22 = nn.Dropout(dropout)
        self.dropout23 = nn.Dropout(dropout)

        self.self_attn3 = nn.MultiheadAttention(num, nhead, dropout=dropout)
        self.linear31 = nn.Linear(num, dim_feedforward)
        self.dropout3 = nn.Dropout(dropout)
        self.linear32 = nn.Linear(dim_feedforward, num)
        self.norm31 = nn.LayerNorm(num)
        self.norm33 = nn.LayerNorm(num)
        self.dropout31 = nn.Dropout(dropout)
        self.dropout33 = nn.Dropout(dropout)

    def with_pos_embed(self, tensor, pos=None):
        return tensor if pos is None else tensor + pos

    def forward(self, src1, src2, src3, pos_src1, pos_src2, pos_src3):
        q1 = k1 = self.with_pos_embed(src1, pos_src1)
        src12 = self.self_attn1(q1, k1, value=src1)[0]
        src1 = src1 + self.dropout11(src12)
        src1 = self.norm11(src1)

        q2 = k2 = self.with_pos_embed(src2, pos_src2)
        src22 = self.self_attn2(q2, k2, value=src2)[0]
        src2 = src2 + self.dropout21(src22)
        src2 = self.norm21(src2)

        src12 = self.multihead_attn1(query=self.with_pos_embed(src1, pos_src1),
                                     key=self.with_pos_embed(src2, pos_src2),
                                     value=src2)[0]
        src1 = src1 + self.dropout12(src12)
        src1 = self.norm12(src1)
        src12 = self.linear12(self.dropout1(self.act(self.linear11(src1))))
        src1 = src1 + self.dropout13(src12)
        src1 = self.norm13(src1)

        src22 = self.multihead_attn2(query=self.with_pos_embed(src2, pos_src2),
                                     key=self.with_pos_embed(src1, pos_src1),
                                     value=src1)[0]
        src2 = src2 + self.dropout22(src22)
        src2 = self.norm22(src2)
        src22 = self.linear22(self.dropout2(self.act(self.linear21(src2))))
        src2 = src2 + self.dropout23(src22)
        src2 = self.norm23(src2)

        q3 = k3 = self.with_pos_embed(src3, pos_src3)
        src32 = self.self_attn3(q3, k3, value=src3)[0]
        src3 = src3 + self.dropout31(src32)
        src3 = self.norm31(src3)
        src32 = self.linear32(self.dropout3(self.act(self.linear31(src3))))
        src3 = src3 + self.dropout33(src32)
        src3 = self.norm33(src3)
        return src1, src2, src3


class DCAEncoder(nn.Module):
    def __init__(self, featurefusion_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(featurefusion_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, src1, src2, src3, pos_src1=None, pos_src2=None, pos_src3=None):
        output1 = src1
        output2 = src2
        output3 = src3

        for layer in self.layers:
            output1, output2, output3 = layer(output1, output2, output3,
                                              pos_src1=pos_src1, pos_src2=pos_src2, pos_src3=pos_src3)
        return output1, output2, output3


class DCADecoder(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, act=nn.GELU()):
        super().__init__()

        self.attn_ch = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm1_ch = nn.LayerNorm(d_model)
        self.dropout1_ch = nn.Dropout(dropout)

        self.attn_sp = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1_sp = nn.Linear(d_model, dim_feedforward)
        self.dropout_sp = nn.Dropout(dropout)
        self.linear2_sp = nn.Linear(dim_feedforward, d_model)
        self.norm1_sp = nn.LayerNorm(d_model)
        self.norm2_sp = nn.LayerNorm(d_model)
        self.dropout1_sp = nn.Dropout(dropout)
        self.dropout2_sp = nn.Dropout(dropout)

        self.act = act

    def with_pos_embed(self, tensor, pos=None):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory_sp, memory_ch, pos_dec=None, pos_enc_sp=None, pos_enc_ch=None):
        tgt2 = self.attn_ch(query=self.with_pos_embed(tgt, pos_dec),
                            key=self.with_pos_embed(memory_ch, pos_enc_ch),
                            value=memory_ch)[0]
        tgt = tgt + self.dropout1_ch(tgt2)
        tgt = self.norm1_ch(tgt)

        tgt2 = self.attn_sp(query=self.with_pos_embed(tgt, pos_dec),
                            key=self.with_pos_embed(memory_sp, pos_enc_sp),
                            value=memory_sp)[0]
        tgt = tgt + self.dropout1_sp(tgt2)
        tgt = self.norm1_sp(tgt)
        tgt2 = self.linear2_sp(self.dropout_sp(self.act(self.linear1_sp(tgt))))
        tgt = tgt + self.dropout2_sp(tgt2)
        tgt = self.norm2_sp(tgt)
        return tgt


class DCATransformer(nn.Module):
    def __init__(self, cfg, d_model=512, nhead=8, num_featurefusion_layers=4,
                 dim_feedforward=2048, dropout=0.1, act=nn.GELU()):
        super().__init__()
        self.s_z = int(cfg.TRAIN.EXEMPLAR_SIZE / cfg.POINT.STRIDE)
        self.s_x = int(cfg.TRAIN.SEARCH_SIZE / cfg.POINT.STRIDE)
        self.d_model = d_model
        self.nhead = nhead
        self.input_proj = nn.Conv2d(d_model, d_model, kernel_size=(1, 1))

        self.pos_temp_sp = positionalencoding(self.s_z ** 2, model_dims=self.d_model)
        self.pos_temp_ch = positionalencoding(self.d_model, model_dims=self.s_z ** 2)
        self.pos_search = positionalencoding(self.s_x ** 2, model_dims=self.d_model)

        featurefusion_layer = DCAEncoderLayer(self.s_z ** 2, d_model, nhead, dim_feedforward, dropout, act)
        self.encoder = DCAEncoder(featurefusion_layer, num_featurefusion_layers)
        self.decoder = DCADecoder(d_model, nhead, dim_feedforward, dropout, act)

        self.class_embed = MLP(d_model * 2, d_model, 2, 3)
        self.bbox_embed = MLP(d_model * 2, d_model, 4, 3)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, xf, zf):
        src_search = self.input_proj(xf)
        src_search = src_search.flatten(2).permute(2, 0, 1).contiguous()
        xf_ = src_search.transpose(1, 0).contiguous()

        src_temp = self.input_proj(zf)
        src_temp_sp = src_temp.flatten(2).permute(2, 0, 1).contiguous()
        src_temp_ch = src_temp.flatten(2).permute(1, 0, 2).contiguous()
        src_temp_sp, src_search, src_temp_ch = self.encoder(src1=src_temp_sp, src2=src_search, src3=src_temp_ch,
                                                            pos_src1=self.pos_temp_sp, pos_src2=self.pos_search,
                                                            pos_src3=self.pos_temp_ch)
        src_temp_ch = src_temp_ch.permute(2, 1, 0).contiguous()

        hs = self.decoder(src_search, src_temp_sp, src_temp_ch,
                          pos_dec=self.pos_search, pos_enc_sp=self.pos_temp_sp, pos_enc_ch=self.pos_temp_ch)
        hs = hs.transpose(1, 0).contiguous()

        cls = self.class_embed(torch.cat((xf_, hs), dim=-1))
        cls = cls.transpose(2, 1).contiguous().view((-1, 2, self.s_x, self.s_x)).contiguous()
        loc = self.bbox_embed(torch.cat((xf_, hs), dim=-1)).sigmoid()
        loc = loc.transpose(2, 1).contiguous().view((-1, 4, self.s_x, self.s_x)).contiguous()
        return cls, loc
