# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import (ConvModule, build_activation_layer, build_conv_layer,
                      build_norm_layer)
from mmengine.model import BaseModule, ModuleList, Sequential
from torch import Tensor

from mmdet.models.layers.transformer.detr_layers import DetrTransformerEncoder
from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType



class RepVGGBlock(BaseModule):
    """RepVGG block is modified to skip branch_norm."""

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: int = 1,
                 padding: int = 1,
                 dilation: int = 1,
                 groups: int = 1,
                 padding_mode: str = 'zeros',
                 conv_cfg: OptConfigType = None,
                 norm_cfg: OptConfigType = dict(type='BN'),
                 act_cfg: OptConfigType = dict(type='ReLU'),
                 without_branch_norm: bool = True,
                 init_cfg: OptConfigType = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode
        self.without_branch_norm = without_branch_norm

        self.conv = build_conv_layer(
            conv_cfg,
            in_channels,
            out_channels,
            3,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=True,
            padding_mode=padding_mode)

        self.norm = build_norm_layer(norm_cfg, out_channels)[1]
        self.act = build_activation_layer(act_cfg)

    def forward(self, x: Tensor) -> Tensor:
        """Forward function."""
        return self.act(self.norm(self.conv(x)))


class CSPRepLayer(BaseModule):
    """CSPRepLayer."""

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 num_blocks: int,
                 expansion: float = 1.0,
                 act_cfg: OptConfigType = dict(type='SiLU', inplace=True),
                 init_cfg: OptConfigType = None) -> None:
        super().__init__(init_cfg=init_cfg)
        mid_channels = int(out_channels // 2)
        self.conv1 = ConvModule(
            in_channels,
            mid_channels,
            1,
            act_cfg=act_cfg)
        self.conv2 = ConvModule(
            in_channels,
            mid_channels,
            1,
            act_cfg=act_cfg)
        self.conv3 = ConvModule(
            mid_channels,
            out_channels,
            1,
            act_cfg=act_cfg)

        self.bottlenecks = Sequential(
            *[RepVGGBlock(mid_channels, mid_channels, act_cfg=act_cfg)
              for _ in range(num_blocks)])

    def forward(self, x: Tensor) -> Tensor:
        """Forward function."""
        x_1 = self.conv1(x)
        x_1 = self.bottlenecks(x_1)
        x_2 = self.conv2(x)
        return self.conv3(x_1 + x_2)


@MODELS.register_module()
class DFRHybridEncoder(BaseModule):
    """DFR-Net Hybrid Encoder with domain decoupling.
    
    This encoder combines the original RT-DETR HybridEncoder with DFR-Net's
    domain decoupling mechanism. It includes:
    1. Content encoder (original backbone features)
    2. Attribute encoder (lightweight domain attribute extractor)
    3. Feature reorganization module (AdaIN-based recombination)
    4. Original HybridEncoder components
    """

    def __init__(self,
                 layer_cfg: ConfigType,
                 projector: OptConfigType = None,
                 num_encoder_layers: int = 1,
                 in_channels: List[int] = [512, 1024, 2048],
                 feat_strides: List[int] = [8, 16, 32],
                 hidden_dim: int = 256,
                 use_encoder_idx: List[int] = [2],
                 pe_temperature: int = 10000,
                 expansion: float = 1.0,
                 depth_mult: float = 1.0,
                 norm_cfg: OptConfigType = dict(type='BN', requires_grad=True),
                 act_cfg: OptConfigType = dict(type='SiLU', inplace=True),
                 eval_size=None,
                 ):
        super(DFRHybridEncoder, self).__init__()
        
        # Original HybridEncoder parameters
        self.in_channels = in_channels
        self.feat_strides = feat_strides
        self.hidden_dim = hidden_dim
        self.use_encoder_idx = use_encoder_idx
        self.num_encoder_layers = num_encoder_layers
        self.pe_temperature = pe_temperature
        self.eval_size = eval_size


        # Channel projection (original)
        self.input_proj = ModuleList()
        for in_channel in in_channels:
            self.input_proj.append(
                ConvModule(
                    in_channel,
                    hidden_dim,
                    kernel_size=1,
                    padding=0,
                    norm_cfg=norm_cfg,
                    act_cfg=None))

        # Encoder transformer (original)
        self.encoder = ModuleList([
            DetrTransformerEncoder(num_encoder_layers, layer_cfg)
            for _ in range(len(use_encoder_idx))
        ])

        # Top-down FPN (original)
        lateral_convs = list()
        fpn_blocks = list()
        for idx in range(len(in_channels) - 1, 0, -1):
            lateral_convs.append(
                ConvModule(
                    hidden_dim,
                    hidden_dim,
                    1,
                    1,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
            fpn_blocks.append(
                CSPRepLayer(
                    hidden_dim * 2,
                    hidden_dim,
                    round(3 * depth_mult),
                    act_cfg=act_cfg,
                    expansion=expansion))
        self.lateral_convs = ModuleList(lateral_convs)
        self.fpn_blocks = ModuleList(fpn_blocks)

        # Bottom-up PAN (original)
        downsample_convs = list()
        pan_blocks = list()
        for idx in range(len(in_channels) - 1):
            downsample_convs.append(
                ConvModule(
                    hidden_dim,
                    hidden_dim,
                    3,
                    stride=2,
                    padding=1,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
            pan_blocks.append(
                CSPRepLayer(
                    hidden_dim * 2,
                    hidden_dim,
                    round(3 * depth_mult),
                    act_cfg=act_cfg,
                    expansion=expansion))
        self.downsample_convs = ModuleList(downsample_convs)
        self.pan_blocks = ModuleList(pan_blocks)

        # Projector (original)
        if projector is not None:
            self.projector = MODELS.build(projector)
        else:
            self.projector = None


        self._reset_parameters()

    def _reset_parameters(self):
        """Reset parameters for positional encoding."""
        if self.eval_size:
            for idx in self.use_encoder_idx:
                stride = self.feat_strides[idx]
                pos_embed = self.build_2d_sincos_position_embedding(
                    self.eval_size[1] // stride, self.eval_size[0] // stride,
                    self.hidden_dim, self.pe_temperature)
                setattr(self, f'pos_embed{idx}', pos_embed)

    def forward(self, 
                inputs: Tuple[Tensor], 
                original_input: Optional[Tensor] = None,
                return_dfr_features: bool = False) -> Tuple[Tensor]:
        """Forward function with DFR-Net support."""
        assert len(inputs) == len(self.in_channels)

        # Original HybridEncoder forward pass
        proj_feats = [
            self.input_proj[i](inputs[i]) for i in range(len(inputs))
        ]
        
        # Encoder (original)
        if self.num_encoder_layers > 0:
            for i, enc_ind in enumerate(self.use_encoder_idx):
                h, w = proj_feats[enc_ind].shape[2:]
                # flatten [B, C, H, W] to [B, HxW, C]
                src_flatten = proj_feats[enc_ind].flatten(2).permute(
                    0, 2, 1).contiguous()
                if self.training or self.eval_size is None:
                    pos_embed = self.build_2d_sincos_position_embedding(
                        w, h, self.hidden_dim, self.pe_temperature)
                else:
                    pos_embed = getattr(self, f'pos_embed{enc_ind}', None)
                memory = self.encoder[i](
                    src_flatten,
                    query_pos=pos_embed.to(src_flatten.device),
                    key_padding_mask=None)
                proj_feats[enc_ind] = memory.permute(
                    0, 2, 1).contiguous().view([-1, self.hidden_dim, h, w])

        # Top-down FPN (original)
        inner_outs = [proj_feats[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_high = inner_outs[0]
            feat_low = proj_feats[idx - 1]
            feat_high = self.lateral_convs[len(self.in_channels) - 1 - idx](
                feat_high)
            inner_outs[0] = feat_high

            upsample_feat = F.interpolate(
                feat_high, scale_factor=2., mode='nearest')
            inner_out = self.fpn_blocks[len(self.in_channels) - 1 - idx](
                torch.cat([upsample_feat, feat_low], axis=1))
            inner_outs.insert(0, inner_out)

        # Bottom-up PAN (original)
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_high = inner_outs[idx + 1]
            downsample_feat = self.downsample_convs[idx](feat_low)
            out = self.pan_blocks[idx](
                torch.cat([downsample_feat, feat_high], axis=1))
            outs.append(out)

        pre_projector_outs = outs

        # Projector (original)
        if self.projector is not None:
            outs = self.projector(outs)

        # 在阶段一（self.enable_dfr=True）的训练中，我们需要 pre_projector_outs
        # 在阶段二（self.enable_dfr=False）的训练中，我们其实也需要它来计算检测loss
        # 因此，统一在训练时返回两个，在推理时返回一个，是最干净的
        if self.training:
            return tuple(pre_projector_outs), tuple(outs)
        else:  # 推理模式
            return tuple(outs)

    @staticmethod
    def build_2d_sincos_position_embedding(w: int,
                                           h: int,
                                           embed_dim=256,
                                           temperature=10000.) -> Tensor:
        """Build 2D sin-cos position embedding."""
        grid_w = torch.arange(int(w), dtype=torch.float32)
        grid_h = torch.arange(int(h), dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h)
        assert embed_dim % 4 == 0, \
            ('Embed dimension must be divisible by 4 for '
             '2D sin-cos position embedding')
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature**omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]

        return torch.cat([
            torch.sin(out_w),
            torch.cos(out_w),
            torch.sin(out_h),
            torch.cos(out_h)
        ],
                         axis=1)[None, :, :] 