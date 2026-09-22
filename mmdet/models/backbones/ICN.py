import warnings

import torch
from torch import Tensor
import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import build_conv_layer, build_norm_layer, build_plugin_layer, ConvModule
from mmengine.model import BaseModule
from torch.nn.modules.batchnorm import _BatchNorm

from mmdet.registry import MODELS


class MPConv(BaseModule):
    def __init__(self,
                 inc,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        # 第一层卷积：输入通道 inc，输出通道 inc，分组默认为1
        self.conv1 = ConvModule(
            in_channels=inc,
            out_channels=inc,
            kernel_size=3,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg
        )

        # 第二层卷积：输入 inc//2，输出 inc//2，分组为 inc//2
        self.conv2 = ConvModule(
            in_channels=inc // 2,
            out_channels=inc // 2,
            kernel_size=5,
            padding=2,
            groups=inc // 2,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg
        )

        # 第三层卷积：输入 inc//4，输出 inc//4，分组为 inc//4
        self.conv3 = ConvModule(
            in_channels=inc // 4,
            out_channels=inc // 4,
            kernel_size=7,
            padding=3,
            groups=inc // 4,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg
        )

        # 最终融合卷积
        self.conv4 = ConvModule(
            in_channels=inc,
            out_channels=inc,
            kernel_size=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None
        )

    def forward(self, x):
        conv1_out = self.conv1(x)
        conv1_out_1, conv1_out_2 = conv1_out.chunk(2, dim=1)

        conv2_out = self.conv2(conv1_out_1)
        conv2_out_1, conv2_out_2 = conv2_out.chunk(2, dim=1)

        conv3_out = self.conv3(conv2_out_1)

        out = torch.cat([conv3_out, conv2_out_2, conv1_out_2], dim=1)
        out = self.conv4(out) + x
        return out


class ICNet(BaseModule):
    """集成 MPConv 的 ICNet 结构"""

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 expand_ratio: float = 0.5,
                 num_blocks: int = 1,
                 conv_cfg = None,
                 norm_cfg = dict(type='BN'),
                 act_cfg = dict(type='SiLU'),
                 init_cfg = None):
        super().__init__(init_cfg=init_cfg)


        mid_channels = int(out_channels * expand_ratio)

        self.main_conv_module = ConvModule(
            in_channels=in_channels,
            out_channels=mid_channels*2,
            kernel_size=1,
            stride=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self.short_conv_module = ConvModule(
            in_channels=mid_channels*2,
            out_channels=out_channels,
            kernel_size=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)


        self.blocks = nn.Sequential(*[
            MPConv(
                inc=mid_channels,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg) for _ in range(num_blocks)
        ])


    def forward(self, x: Tensor) -> Tensor:
        x_1, x_2 = self.short_conv_module(x).chunk(2, 1)
        x_out = self.main_conv_module(torch.cat((self.blocks(x_1), x_2), 1))
        return x_out


@MODELS.register_module()
class ICN(BaseModule):
    """基于 ICNet 的主干网络"""

    def __init__(self,
                 in_channels=3,
                 arch_cfg=[
                     # (stage_channels, num_blocks, stride)
                     (128, 1, 2),  # stage1
                     (256, 1, 2),  # stage2
                     (384, 1, 2),  # stage3
                     (384, 3, 2),  # stage4 (最后阶段可关闭MPConv)
                 ],
                 expand_ratio=0.5,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg=dict(type='SiLU'),
                 out_indices=(2, 3, 4),  # 默认输出后三个阶段的特征图
                 init_cfg=None,
                 norm_eval=True,
                 pretrained=None,):
        super().__init__(init_cfg=init_cfg)


        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be specified at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is None:
            if init_cfg is None:
                self.init_cfg = [
                    dict(type='Kaiming', layer='Conv2d'),
                    dict(
                        type='Constant',
                        val=1,
                        layer=['_BatchNorm', 'GroupNorm'])
                ]
        else:
            raise TypeError('pretrained must be a str or None')

        self.norm_eval = norm_eval

        self.out_indices = out_indices
        self.stages = nn.ModuleList()

        # 初始下采样层
        self.stem = ConvModule(in_channels, 64, 3, stride=2, padding=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)

        # 构建各阶段
        in_ch = 64
        for i, (stage_ch, num_blocks, stride) in enumerate(arch_cfg):
            stage = nn.Sequential(
                # 下采样卷积
                ConvModule(in_ch, stage_ch, 3, stride=stride, padding=1,
                           conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg),
                # ICNet模块堆叠
                ICNet(
                    in_channels=stage_ch,
                    out_channels=stage_ch,
                    num_blocks=num_blocks,
                    expand_ratio=expand_ratio,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg
                )
            )
            self.stages.append(stage)
            in_ch = stage_ch


    def forward(self, x):
        x = self.stem(x)
        outs = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)

    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        freezed."""
        super(ICN, self).train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()
