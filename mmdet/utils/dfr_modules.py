from typing import List, Optional
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import (ConvModule, build_activation_layer,build_norm_layer)
from mmengine.model import BaseModule, ModuleList, Sequential
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType
from torch.utils.checkpoint import checkpoint


class GradientScaler(torch.autograd.Function):
    """Custom autograd function to scale gradients without changing dtype.

    This prevents gradient overflow in backward pass while maintaining FP32 output
    in forward pass for numerical stability.
    """
    @staticmethod
    def forward(ctx, input_tensor, scale=0.5):
        """Forward: pass through without dtype change"""
        ctx.scale = scale
        return input_tensor

    @staticmethod
    def backward(ctx, grad_output):
        """Backward: scale gradient to prevent overflow"""
        # Scale down gradient to prevent overflow
        scaled_grad = grad_output * ctx.scale
        return scaled_grad, None

class ZeroInitConv2d(nn.Conv2d):
    """
    A zero-initialized convolution that preserves the residual path at startup.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight.data.zero_()
        if self.bias is not None:
            self.bias.data.zero_()

@MODELS.register_module()
class AdaIN(nn.Module):
    def __init__(self, content_dim: int, style_dim: int, num_scales: int = 3, zero_init: bool = True, gamma_gate_init: float = 0.0):
        super().__init__()
        self.content_dim = content_dim
        self.style_dim = style_dim
        self.num_scales = num_scales

        self.low_freq_dim = content_dim // num_scales
        self.concat_dim = self.low_freq_dim * num_scales

        self.content_proj = nn.Conv2d(content_dim, self.concat_dim, 1)


        self.fc_gamma = nn.Linear(style_dim, self.low_freq_dim, dtype = torch.float32)
        self.fc_beta = nn.Linear(style_dim, self.low_freq_dim, dtype = torch.float32)

        if zero_init:
            nn.init.zeros_(self.fc_gamma.weight)
            nn.init.ones_(self.fc_gamma.bias)
            nn.init.zeros_(self.fc_beta.weight)
            nn.init.zeros_(self.fc_beta.bias)




        self.gamma_gate = nn.Parameter(torch.ones(1) * gamma_gate_init)

        self.aggregate = nn.Conv2d(self.concat_dim, content_dim, 1)

        if not zero_init:
            self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu', a=0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, content: Tensor, style: Tensor) -> Tensor:
        B, C, H, W = content.shape

        if torch.isnan(content).any():
            content = torch.nan_to_num(content, nan=0.0)
        if torch.isnan(style).any():
            style = torch.nan_to_num(style, nan=0.0)

        decomposed = self.content_proj(content).chunk(self.num_scales, dim=1)
        low_freq = decomposed[0]
        high_freqs = decomposed[1:]


        with torch.cuda.amp.autocast(enabled=False):
            low_freq_fp32 = low_freq.float()


            mu_orig = low_freq_fp32.mean([2, 3], keepdim=True)
            std_orig = low_freq_fp32.std([2, 3], keepdim=True) + 1e-6


            gamma_attr = F.linear(
                style.float(),
                self.fc_gamma.weight.float(),
                self.fc_gamma.bias.float()
            ).view(B, -1, 1, 1)

            beta_attr = F.linear(
                style.float(),
                self.fc_beta.weight.float(),
                self.fc_beta.bias.float()
            ).view(B, -1, 1, 1)


            gamma = std_orig + self.gamma_gate * (gamma_attr - std_orig)
            beta = mu_orig + self.gamma_gate * (beta_attr - mu_orig)

            # AdaIN
            normalized = (low_freq_fp32 - mu_orig) / std_orig
            low_enhanced = normalized * gamma + beta


            low_enhanced = low_enhanced.to(content.dtype)

        enhanced = torch.cat([low_enhanced] + list(high_freqs), dim=1)
        return self.aggregate(enhanced)

@MODELS.register_module()
class AttributeEncoder(BaseModule):
    """Lightweight attribute encoder for domain-specific features.

    Args:
        in_channels: Input image channels (default: 3)
        hidden_dims: Hidden dimensions of feature extractor (default: [64, 128, 256])
        out_dim: Output attribute vector dimension (default: 256)
        norm_cfg: Normalization config for feature extractor
        act_cfg: Activation config
        normalize_output: If True, L2-normalize output to unit sphere (direction-only).
            This forces the encoder to learn direction information rather than
            relying on magnitude differences (e.g., brightness/scale).
            Default: False for backward compatibility.
        norm_scale: Scale factor applied after L2 normalization. Only used when
            normalize_output=True. Default: 5.0 (matches typical pre-normalization norms).
    """

    def __init__(self,
                 in_channels: int = 3,
                 hidden_dims: List[int] = [64, 128, 256],
                 out_dim: int = 256,
                 norm_cfg: OptConfigType = dict(type='GN', num_groups=32),
                 act_cfg: OptConfigType = dict(type='ReLU'),
                 normalize_output: bool = False,
                 norm_scale: float = 5.0):
        super().__init__()

        layers = []
        prev_channels = in_channels

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Conv2d(prev_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                build_norm_layer(norm_cfg, hidden_dim)[1],
                build_activation_layer(act_cfg),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
                build_norm_layer(norm_cfg, hidden_dim)[1],
                build_activation_layer(act_cfg)
            ])
            prev_channels = hidden_dim
        self.out_dim = out_dim
        self.normalize_output = normalize_output
        self.norm_scale = norm_scale
        self.feature_extractor = nn.Sequential(*layers)

        # Global average pooling + MLP for final attribute vector
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.attribute_mlp = nn.Sequential(
            nn.Linear(prev_channels, out_dim),
            build_activation_layer(act_cfg),
            nn.Linear(out_dim, out_dim)
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input image (B, 3, H, W)

        Returns:
            Attribute vector (B, out_dim)
        """
        features = self.feature_extractor(x)
        pooled = self.global_pool(features).squeeze(-1).squeeze(-1)
        attribute_vector = self.attribute_mlp(pooled)

        if self.normalize_output:
            # L2 normalize to unit sphere, then scale
            # This forces the model to learn DIRECTIONAL information
            # rather than relying on magnitude (e.g., brightness)
            attribute_vector = F.normalize(attribute_vector, dim=1, eps=1e-6)
            attribute_vector = attribute_vector * self.norm_scale

        return attribute_vector

@MODELS.register_module()
class FeatureReorganizationModule(BaseModule):
    """Feature reorganization module using AdaIN for domain feature recombination."""

    def __init__(self,
                 content_dim: list = [512, 256, 128],
                 style_dim: int = 256,
                 num_levels: int = 3,
                 gamma_gate_init: float = 0.0):
        super().__init__()

        self.num_levels = num_levels
        self.adain_modules = ModuleList([
            AdaIN(c_dim, style_dim, gamma_gate_init=gamma_gate_init) for c_dim in content_dim
        ])

    def forward(self, content_features: List[Tensor], style_vector: Tensor) -> List[Tensor]:
        """
        Args:
            content_features: List of content features [(B, C, H, W), ...]
            style_vector: Style vector (B, style_dim)

        Returns:
            Reorganized features [(B, C, H, W), ...]
        """
        reorganized_features = []
        for i, content_feat in enumerate(content_features):
            if i < len(self.adain_modules):
                reorganized_feat = self.adain_modules[i](content_feat, style_vector)
                reorganized_features.append(reorganized_feat)
            else:
                reorganized_features.append(content_feat)
        return reorganized_features


class UpBlock(nn.Module):
    """Upsampling block with PixelShuffle"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels * 4, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.norm(x)
        x = self.relu(x)
        return x

class UpBlockWithSkip(nn.Module):
    """Upsampling block that concatenates with a skip connection."""
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        # Upsample + Conv for the main path
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        # The input channels for the merge convolution will be the sum
        self.conv = ConvModule(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            act_cfg=dict(type='ReLU')
        )

    def forward(self, x, skip_feature):
        x = self.upsample(x)
        # Concatenate along the channel dimension
        x = torch.cat([x, skip_feature], dim=1)
        x = self.conv(x)
        return x


@MODELS.register_module()
class ImageDecoder(BaseModule):
    def __init__(self,
                 neck_in_channels: List[int],
                 backbone_in_channels: List[int],
                 base_channels: int = 256,
                 final_upsample_stages: int = 3,
                 out_channels: int = 3,
                 with_cp: bool = False,
                 init_cfg: Optional[dict] = None):
        super().__init__(init_cfg=init_cfg)
        self.with_cp = with_cp


        # backbone_in_channels: [c1(large), c2, c3(small)] e.g., [256, 512, 1024]

        # skip_channels: [c3(small), c2, c1(large)]
        skip_channels = backbone_in_channels[::-1]



        self.fusion_conv = ConvModule(sum(neck_in_channels), base_channels, 1, act_cfg=dict(type='ReLU'))


        self.up_blocks = nn.ModuleList()
        current_channels = base_channels



        num_up_with_skip = len(skip_channels) - 1

        for i in range(num_up_with_skip):


            skip_ch = skip_channels[i + 1]
            out_ch = current_channels // 2

            self.up_blocks.append(
                UpBlockWithSkip(current_channels, skip_ch, out_ch)
            )
            current_channels = out_ch


        self.final_upsample = nn.Sequential()
        for i in range(final_upsample_stages):
            out_ch = current_channels // 2 if current_channels > out_channels * 4 else current_channels
            self.final_upsample.add_module(f'up_{i}', UpBlock(current_channels, out_ch))
            current_channels = out_ch


        self.final_conv = nn.Conv2d(current_channels, out_channels, kernel_size=3, padding=1)
        self.final_act = nn.Sigmoid()

    def forward(self, neck_features: List[Tensor], backbone_features: List[Tensor]) -> Tensor:
        """
        Args:
            neck_features (List[Tensor]): [c1(large), c2, c3(small)]
            backbone_features (List[Tensor]): [c1(large), c2, c3(small)]
        """

        def _checkpoint_wrapper(module, *args):

            if self.training and self.with_cp:
                return checkpoint(module, *args)
            else:
                return module(*args)

        h, w = neck_features[-1].shape[2:]
        x = torch.cat([F.interpolate(f, size=(h, w), mode='bilinear', align_corners=False)
                       for f in neck_features], dim=1)
        x = _checkpoint_wrapper(self.fusion_conv, x)



        for i, up_block in enumerate(self.up_blocks):
            # i=0: x(20) -> upsample to 40, fuse with backbone_features[1](40)
            # i=1: x(40) -> upsample to 80, fuse with backbone_features[0](80)



            skip_feature = backbone_features[len(backbone_features) - 2 - i]
            x = _checkpoint_wrapper(up_block, x, skip_feature)


        x = _checkpoint_wrapper(self.final_upsample, x)


        output = self.final_conv(x)
        return self.final_act(output)


@MODELS.register_module()
class FiLM(nn.Module):
    """Feature-wise Linear Modulation.

    Uses a small MLP to predict per-channel gamma and bias from a style vector.
    Zero-initialized to be identity mapping at start (gamma=1, bias=0),
    ensuring frozen bbox_head sees unchanged features initially.
    """

    def __init__(self, style_dim: int, num_channels: int, zero_init: bool = True):
        super().__init__()
        self.num_channels = num_channels
        hidden_dim = 128
        self.mlp = nn.Sequential(
            nn.Linear(style_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, num_channels * 2)
        )
        if zero_init:
            nn.init.zeros_(self.mlp[-1].weight)
            nn.init.constant_(self.mlp[-1].bias[:num_channels], 1.0)  # gamma = 1
            nn.init.constant_(self.mlp[-1].bias[num_channels:], 0.0)   # bias = 0

    def forward(self, feat: Tensor, style_vec: Tensor) -> Tensor:
        B, C, H, W = feat.shape
        params = self.mlp(style_vec)  # [B, C*2]
        gamma = params[:, :C].view(B, C, 1, 1)
        bias = params[:, C:].view(B, C, 1, 1)
        # Clamp gamma to prevent extreme scaling that causes NaN in feat
        # Range [0.1, 10] allows strong modulation but prevents explosion/collapse
        gamma = torch.clamp(gamma, min=0.1, max=10.0)
        return feat * gamma + bias
