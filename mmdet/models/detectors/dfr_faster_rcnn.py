# ============================================================
# DFR-FasterRCNN (Fixed Version)
# ============================================================
# Fixes applied compared to original:
#   1. Added FrozenBatchNorm2d + freeze_bn_stats for generation stage
#   2. Added self/cross reconstruction losses (feature_reorganization usage)
#   3. Added curriculum phase control (Phase A/B/C)
#   4. Added domain EMA statistics (mu/sigma buffers)
#   5. Added AdaIN gamma_gate scheduling
#   6. Added attribute monitoring
#   7. Added NaN checks
#   8. Added forward() ZeRO-3 compatibility
#   9. Added BN restore for joint stage
#   10. Added init_weights() with custom initialization
# ============================================================

from typing import Dict, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F
import numpy as np
from torch.utils.checkpoint import checkpoint
from mmengine import MessageHub

from mmdet.registry import MODELS
from mmdet.models.detectors.faster_rcnn import FasterRCNN
from mmdet.structures import OptSampleList
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig

from mmdet.utils.dfr_modules import FiLM
from mmdet.utils.physical_degradation import (
    PhysicalLowlightDegradation,
    PairedDegradationSampler,
)


# ------------------------------------------------------------------
# FrozenBatchNorm2d: freeze BN statistics while keeping weight/bias trainable
# ------------------------------------------------------------------
class FrozenBatchNorm2d(nn.Module):
    """FrozenBN: use fixed running_mean/var, no parameter updates."""
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.register_buffer("weight", torch.ones(num_features))
        self.register_buffer("bias", torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x):
        scale = self.weight * (self.running_var + self.eps).rsqrt()
        bias = self.bias - self.running_mean * scale
        scale = scale.reshape(1, -1, 1, 1)
        bias = bias.reshape(1, -1, 1, 1)
        return x * scale + bias

    @classmethod
    def from_bn(cls, bn_module):
        frozen = cls(bn_module.num_features, bn_module.eps)
        with torch.no_grad():
            frozen.weight.copy_(bn_module.weight)
            frozen.bias.copy_(bn_module.bias)
            frozen.running_mean.copy_(bn_module.running_mean)
            frozen.running_var.copy_(bn_module.running_var)
        return frozen


def freeze_bn_stats(module):
    """Freeze BN statistics (running_mean/var) but keep weight/bias trainable."""
    for m in module.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
            m.eval()
            m.running_mean.requires_grad = False
            m.running_var.requires_grad = False


@MODELS.register_module()
class DFRFasterRCNN(FasterRCNN):

    def __init__(
        self,
        backbone: ConfigType,
        rpn_head: ConfigType,
        roi_head: ConfigType,
        train_cfg: OptConfigType = None,
        test_cfg: OptConfigType = None,
        neck: OptConfigType = None,
        data_preprocessor: OptConfigType = None,
        init_cfg: OptMultiConfig = None,

        attribute_encoder: OptConfigType = None,
        reorganization: OptConfigType = None,
        feature_consistency_loss: OptConfigType = None,

        training_stage: str = 'joint',
        enable_dfr: bool = True,
        enable_reconstruction: bool = True,
        enable_head_film: bool = True,
        use_physical_degradation: bool = True,

        dfr_recon_loss_weight: float = 0.1,
        dfr_cross_loss_weight: float = 0.5,
        dfr_content_loss_weight: float = 1.0,
        dfr_ortho_loss_weight: float = 0.1,
        dfr_attr_sep_loss_weight: float = 0.15,
        dfr_equivariant_loss_weight: float = 1.0,
        dfr_content_fnr_threshold: float = 0.5,
        dfr_content_use_domain_aware: bool = True,
        dfr_content_domain_temperature: float = 0.3,

        attr_monitor_interval: int = 100,
        freeze_backbone_bn_layers: int = 2,  # ResNet: freeze first 2 stages
        freeze_neck_bn: bool = True,

        fpn_channels=(256, 256, 256, 256, 256),
    ):
        super().__init__(
            backbone=backbone,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            neck=neck,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg,
        )

        # ---- Stage and feature flags ----
        self.training_stage = training_stage
        self.enable_dfr = enable_dfr
        self.enable_reconstruction = enable_reconstruction
        self.enable_head_film = enable_head_film
        self.use_physical_degradation = use_physical_degradation

        # ---- Loss weights ----
        self.dfr_recon_loss_weight = dfr_recon_loss_weight
        self.dfr_cross_loss_weight = dfr_cross_loss_weight
        self.dfr_content_loss_weight = dfr_content_loss_weight
        self.dfr_ortho_loss_weight = dfr_ortho_loss_weight
        self.dfr_attr_sep_loss_weight = dfr_attr_sep_loss_weight
        self.dfr_equivariant_loss_weight = dfr_equivariant_loss_weight
        self.dfr_content_fnr_threshold = dfr_content_fnr_threshold
        self.dfr_content_use_domain_aware = dfr_content_use_domain_aware
        self.dfr_content_domain_temperature = dfr_content_domain_temperature

        self.attr_monitor_interval = attr_monitor_interval
        self.freeze_backbone_bn_layers = freeze_backbone_bn_layers
        self.freeze_neck_bn = freeze_neck_bn

        # ---- DFR modules ----
        self.attribute_encoder = MODELS.build(attribute_encoder)
        self.feature_reorganization = MODELS.build(reorganization)
        self.feature_consistency_loss = MODELS.build(feature_consistency_loss)

        style_dim = getattr(self.attribute_encoder, 'out_dim', 256)

        # ---- DFR losses ----
        self.loss_ortho = MODELS.build(
            dict(type='DFROrthogonalityLoss', loss_weight=dfr_ortho_loss_weight)
        )
        self.loss_equivariant = MODELS.build(
            dict(
                type='DFRDegradationRegressionLoss',
                loss_weight=dfr_equivariant_loss_weight,
                attr_dim=style_dim,
            )
        )
        self.loss_self_recon = MODELS.build(
            dict(type='DFRReconstructionLoss', loss_weight=0.5)
        )
        # Cross-reconstruction perceptual loss uses feature_consistency_loss directly

        # ---- Physical degradation ----
        self.physical_degrade = PhysicalLowlightDegradation(return_params=True)
        self.paired_degradation = PairedDegradationSampler(self.physical_degrade)

        # ---- FPN feature dimensions ----
        self.fpn_channels = list(fpn_channels)

        # ---- Head FiLM modules ----
        if self.enable_head_film:
            self.head_film_modules = nn.ModuleList([
                FiLM(style_dim=style_dim, num_channels=ch)
                for ch in self.fpn_channels
            ])
        else:
            self.head_film_modules = None

        # ---- Fixed orthogonal projection for content vector ----
        content_dim = sum(self.fpn_channels)
        proj = torch.empty(style_dim, content_dim)
        nn.init.orthogonal_(proj)
        self.register_buffer('content_proj_fixed', proj)

        # ---- Domain EMA mean vectors ----
        self.register_buffer('ema_sup_mean', torch.zeros(style_dim))
        self.register_buffer('ema_unsup_mean', torch.zeros(style_dim))
        self.register_buffer(
            'domain_ema_initialized',
            torch.tensor(False, dtype=torch.bool)
        )
        self.domain_stat_ema_momentum = 0.99

        # ---- Domain-level statistics EMA (for cross-reconstruction style target) ----
        self.register_buffer('voc_stat_mu', torch.zeros(self.fpn_channels[0]))
        self.register_buffer('voc_stat_sigma', torch.ones(self.fpn_channels[0]))
        self.register_buffer('exdark_stat_mu', torch.zeros(self.fpn_channels[0]))
        self.register_buffer('exdark_stat_sigma', torch.ones(self.fpn_channels[0]))
        self.ema_momentum = 0.99

        # ---- Initialize weights ----
        self.init_weights()

        # ---- BN freeze/restore by stage ----
        if training_stage == 'generation':
            self._freeze_bn_layers()
        else:
            self._restore_bn_layers()

        self._freeze_modules_by_stage()

    # =================================================================
    # BN Management (ResNet50 adapted)
    # =================================================================
    def _freeze_bn_layers(self):
        """Layer-wise BN freeze for ResNet50:
        - Freeze conv1 + layer1 + layer2 (first 2 stages) -> FrozenBN
        - Adaptive layer3 + layer4 (last 2 stages) -> freeze stats only
        - FPN: freeze all BN stats
        """
        frozen_count = 0
        adaptive_count = 0

        for name, module in self.backbone.named_modules():
            if isinstance(module, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                layer_idx = self._parse_resnet_layer_idx(name)

                if layer_idx < self.freeze_backbone_bn_layers:
                    # Freeze: convert to FrozenBN
                    parent_name = '.'.join(name.split('.')[:-1])
                    child_name = name.split('.')[-1]
                    parent = self.backbone.get_submodule(parent_name) if parent_name else self.backbone
                    frozen = FrozenBatchNorm2d.from_bn(module)
                    setattr(parent, child_name, frozen)
                    frozen_count += 1
                else:
                    # Adaptive: freeze running stats only
                    freeze_bn_stats(module)
                    adaptive_count += 1

        # Freeze FPN BN stats
        if self.freeze_neck_bn and hasattr(self, 'neck'):
            for name, module in self.neck.named_modules():
                if isinstance(module, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                    freeze_bn_stats(module)

        print(f"[BN Freeze] generation stage: {frozen_count} FrozenBN, "
              f"{adaptive_count} adaptive BN")

    def _restore_bn_layers(self):
        """Restore FrozenBatchNorm2d to BatchNorm2d for joint fine-tuning."""
        def _replace_frozenbn_recursive(module):
            restored_count = 0
            for child_name, child in module.named_children():
                if isinstance(child, FrozenBatchNorm2d):
                    new_bn = nn.BatchNorm2d(
                        num_features=child.num_features,
                        eps=child.eps,
                        affine=True,
                        track_running_stats=True,
                    )
                    with torch.no_grad():
                        new_bn.weight.copy_(child.weight)
                        new_bn.bias.copy_(child.bias)
                        new_bn.running_mean.copy_(child.running_mean)
                        new_bn.running_var.copy_(child.running_var)
                    setattr(module, child_name, new_bn)
                    restored_count += 1
                else:
                    restored_count += _replace_frozenbn_recursive(child)
            return restored_count

        total = _replace_frozenbn_recursive(self)
        if total > 0:
            print(f"[BN Restore] {total} FrozenBN -> BatchNorm2d")

    @staticmethod
    def _parse_resnet_layer_idx(module_name):
        """Parse ResNet50 layer index from module name.
        e.g., 'layer1.0.conv1.bn' -> 1, 'layer3.2.conv2.bn' -> 3
        """
        parts = module_name.split('.')
        for part in parts:
            if part.startswith('layer'):
                try:
                    return int(part[5:])
                except ValueError:
                    pass
            elif part == 'conv1':
                return 0
        return 0

    # =================================================================
    # Initialization
    # =================================================================
    def init_weights(self):
        """Custom initialization for DFR modules."""
        super().init_weights()
        self._load_ema_weights_if_available()

        self.attribute_encoder.apply(self._kaiming_init_with_bn)
        self.feature_reorganization.apply(self._xavier_init_for_linear)

        # Break symmetry in attribute_encoder
        for m in self.attribute_encoder.modules():
            if isinstance(m, nn.Linear) and m.weight is not None:
                m.weight.data += torch.randn_like(m.weight.data) * 0.05
            if isinstance(m, nn.Conv2d):
                m.weight.data += torch.randn_like(m.weight.data) * 0.02

        print("--- DFR-FasterRCNN custom initializations applied ---")

    def _load_ema_weights_if_available(self):
        """Load EMA weights from checkpoint if available."""
        import os
        init_cfg = getattr(self, 'init_cfg', None)
        if not init_cfg or not isinstance(init_cfg, dict):
            return
        checkpoint_path = init_cfg.get('checkpoint')
        if not checkpoint_path or not os.path.exists(checkpoint_path):
            return
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            if 'ema_state_dict' not in checkpoint:
                return
            ema_sd = checkpoint['ema_state_dict']
            processed = {
                k.replace('module.', ''): v
                for k, v in ema_sd.items() if k != 'steps'
            }
            missing, unexpected = self.load_state_dict(processed, strict=False)
            print(f"Loaded EMA weights: {len(processed)} keys, "
                  f"{len(missing)} missing, {len(unexpected)} unexpected")
        except Exception as e:
            print(f"Failed to load EMA weights: {e}")

    @staticmethod
    def _kaiming_init_with_bn(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    @staticmethod
    def _xavier_init_for_linear(m):
        if isinstance(m, nn.Linear) and m.weight is not None:
            if m.weight.dim() >= 2:
                nn.init.xavier_uniform_(m.weight)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    # =================================================================
    # Stage-based module freeze
    # =================================================================
    def _freeze_modules_by_stage(self):
        if self.training_stage == 'detection':
            self._freeze_generation_modules()
            print("[Stage 1] Frozen generation modules")
        elif self.training_stage == 'generation':
            self._freeze_detection_modules()
            print("[Stage 2] Frozen detection modules (backbone, neck, rpn, roi)")
        elif self.training_stage == 'joint':
            print("[Stage 3] Joint training: all detection modules unfrozen")
            if hasattr(self, 'attribute_encoder'):
                for p in self.attribute_encoder.parameters():
                    p.requires_grad = False
                self.attribute_encoder.eval()
                print("[Stage 3] Frozen attribute_encoder")
        else:
            raise ValueError(f"Unknown training_stage: {self.training_stage}")

    def _freeze_generation_modules(self):
        for module_name in [
            'attribute_encoder', 'feature_reorganization',
            'feature_consistency_loss', 'loss_ortho',
            'loss_equivariant', 'loss_self_recon',
        ]:
            if hasattr(self, module_name):
                module = getattr(self, module_name)
                for p in module.parameters():
                    p.requires_grad = False
                module.eval()

    def _freeze_detection_modules(self):
        for module_name in ['backbone', 'neck', 'rpn_head', 'roi_head']:
            if hasattr(self, module_name):
                module = getattr(self, module_name)
                for p in module.parameters():
                    p.requires_grad = False
                module.eval()

    # =================================================================
    # Attribute encoding helpers
    # =================================================================
    def _get_attr_vecs(self, batch_inputs: Tensor):
        attr_input_fp32 = F.interpolate(
            batch_inputs.float(),
            size=(224, 224),
            mode='bilinear',
            align_corners=False,
        )
        if batch_inputs.dtype == torch.bfloat16:
            attr_input = attr_input_fp32.bfloat16()
        else:
            attr_input = attr_input_fp32
        attr_vecs = self.attribute_encoder(attr_input)
        return attr_vecs, attr_input_fp32, attr_input

    def _apply_film(self, feats, attr_vecs):
        if not self.enable_head_film or self.head_film_modules is None:
            return feats
        return tuple(
            film(feat, attr_vecs)
            for film, feat in zip(self.head_film_modules, feats)
        )

    # =================================================================
    # NaN checker
    # =================================================================
    @staticmethod
    def _check_nan(tensor, name):
        if tensor is not None and torch.isnan(tensor).any():
            print(f"[CRITICAL] {name} contains NaN! "
                  f"shape={tensor.shape}, count={torch.isnan(tensor).sum().item()}")
            return True
        return False

    # =================================================================
    # Sample grouping
    # =================================================================
    def _group_samples(self, batch_data_samples: OptSampleList) -> Tuple[list, list]:
        sup_idx, unsup_idx = [], []
        for i, sample in enumerate(batch_data_samples):
            meta = getattr(sample, 'metainfo', {})
            if 'domain_id' in meta:
                (sup_idx if meta['domain_id'] == 0 else unsup_idx).append(i)
                continue
            if (hasattr(sample, 'gt_instances')
                    and getattr(sample.gt_instances, 'bboxes', None) is not None
                    and len(sample.gt_instances.bboxes) > 0):
                sup_idx.append(i)
            else:
                unsup_idx.append(i)
        return sup_idx, unsup_idx

    def _get_supervised_samples(self, batch_data_samples: OptSampleList) -> Tuple[list, list]:
        sup_idx, sup_samples = [], []
        for i, sample in enumerate(batch_data_samples):
            if (hasattr(sample, 'gt_instances')
                    and getattr(sample.gt_instances, 'bboxes', None) is not None
                    and len(sample.gt_instances.bboxes) > 0):
                sup_idx.append(i)
                sup_samples.append(sample)
        return sup_idx, sup_samples

    # =================================================================
    # Core loss method
    # =================================================================
    def loss(self, batch_inputs: Tensor, batch_data_samples: OptSampleList) -> Dict[str, Tensor]:
        losses = {}

        message_hub = MessageHub.get_current_instance()
        iter_num = message_hub.get_info('iter') or 0
        epoch = message_hub.get_info('epoch') or 0
        epoch_1idx = epoch + 1

        # ---- NaN check on input ----
        if self._check_nan(batch_inputs, 'batch_inputs'):
            batch_inputs = torch.nan_to_num(batch_inputs, nan=0.0)

        # =============================================================
        # 1. Curriculum phase control (A/B/C) - NEW
        # =============================================================
        enable_deg_reg = True
        enable_ortho = True
        enable_attr_sep = False
        enable_self_recon = False
        enable_cross = False
        deg_reg_weight = 1.0
        ortho_weight = 0.1
        attr_sep_weight = 0.0
        self_recon_weight = 0.0
        effective_recon_weight = 0.0
        effective_cross_content = 0.0

        if self.training_stage == 'generation':
            if epoch_1idx <= 7:
                # Phase A: build attribute space
                enable_attr_sep = True
                enable_self_recon = True
                enable_cross = False
                deg_reg_weight = 1.0
                ortho_weight = 0.1
                attr_sep_weight = 0.10
                self_recon_weight = 0.5
                effective_recon_weight = self.dfr_recon_loss_weight * 0.5
                effective_cross_content = 0.0
            elif epoch_1idx <= 16:
                # Phase B: add cross-recon content
                enable_attr_sep = True
                enable_self_recon = True
                enable_cross = True
                deg_reg_weight = 0.5
                ortho_weight = 0.1
                attr_sep_weight = 0.15
                self_recon_weight = 0.5
                effective_recon_weight = self.dfr_recon_loss_weight * 0.75
                effective_cross_content = 0.025
            else:
                # Phase C: full
                enable_attr_sep = True
                enable_self_recon = True
                enable_cross = True
                deg_reg_weight = 0.5
                ortho_weight = 0.1
                attr_sep_weight = 0.15
                self_recon_weight = 0.5
                effective_recon_weight = self.dfr_recon_loss_weight
                effective_cross_content = 0.05

        # Apply dynamic weights
        self.loss_equivariant.loss_weight = deg_reg_weight
        self.loss_ortho.loss_weight = ortho_weight
        self.loss_self_recon.loss_weight = self_recon_weight

        # =============================================================
        # 2. Extract features
        # =============================================================
        feats = self.extract_feat(batch_inputs)
        for i, f in enumerate(feats):
            self._check_nan(f, f'feats[{i}]')

        # =============================================================
        # 3. Attribute encoding + physical degradation
        # =============================================================
        attr_vecs, attr_input_fp32, attr_input = self._get_attr_vecs(batch_inputs)

        sup_idx, unsup_idx = self._group_samples(batch_data_samples)
        B = attr_vecs.size(0)
        domain_labels = torch.zeros(B, dtype=torch.long, device=attr_vecs.device)
        if unsup_idx:
            domain_labels[unsup_idx] = 1

        # Physical degradation views
        attr_vecs_deg1 = attr_vecs_deg2 = deg_params_1 = deg_params_2 = None
        if self.training and self.enable_dfr and self.use_physical_degradation:
            attr_input_deg1, attr_input_deg2, deg_params_1, deg_params_2 = \
                self.paired_degradation.sample_pair_asymmetric(
                    attr_input_fp32, domain_labels,
                    voc_alpha_range=(0.5, 1.0), exdark_alpha_range=(0.1, 0.5),
                )
            if batch_inputs.dtype == torch.bfloat16:
                attr_input_deg1 = attr_input_deg1.bfloat16()
                attr_input_deg2 = attr_input_deg2.bfloat16()
            attr_vecs_deg1 = self.attribute_encoder(attr_input_deg1)
            attr_vecs_deg2 = self.attribute_encoder(attr_input_deg2)

        # =============================================================
        # 4. Apply FiLM to FPN features
        # =============================================================
        feats = self._apply_film(feats, attr_vecs)

        # ---- AdaIN gamma_gate scheduling (NEW) ----
        if self.training_stage == 'generation':
            gate = min(1.0, iter_num / 8000.0)
            if hasattr(self.feature_reorganization, 'adain_modules'):
                for m in self.feature_reorganization.adain_modules:
                    if hasattr(m, 'gamma_gate'):
                        m.gamma_gate.data.fill_(gate)
                        m.gamma_gate.requires_grad_(False)

        # =============================================================
        # 5. Detection losses (RPN + ROI) on supervised samples
        # =============================================================
        sup_idx_det, sup_samples = self._get_supervised_samples(batch_data_samples)

        if sup_idx_det:
            sup_feats = tuple(f[sup_idx_det].contiguous() for f in feats)

            rpn_losses, rpn_results_list = self.rpn_head.loss_and_predict(
                sup_feats, sup_samples,
                proposal_cfg=self.train_cfg.get('rpn_proposal', self.test_cfg.rpn),
            )
            losses.update({f'rpn_{k}': v for k, v in rpn_losses.items()})

            roi_losses = self.roi_head.loss(sup_feats, rpn_results_list, sup_samples)
            losses.update({f'roi_{k}': v for k, v in roi_losses.items()})
        else:
            losses['det_dummy'] = sum(f.mean() for f in feats) * 0.0

        # =============================================================
        # 6. Self/Cross Reconstruction losses (NEW - critical fix)
        # =============================================================
        if self.enable_reconstruction and self.training:
            if enable_self_recon:
                # ---- Self-reconstruction ----
                recon_self = checkpoint(
                    self.feature_reorganization, feats, attr_vecs, use_reentrant=False
                )
                loss_self_recon = self.loss_self_recon(
                    [f.detach() for f in feats], list(recon_self)
                )
                losses.update(loss_self_recon)

                # ---- Perceptual consistency ----
                loss_recon_feat = self.feature_consistency_loss(
                    reorganized_features=recon_self[0],
                    target_image=attr_input,
                )
                dummy_loss = sum(f.mean() for f in recon_self[1:]) * 0.0
                losses['loss_feat_recon'] = loss_recon_feat * effective_recon_weight + dummy_loss

                # ---- Update domain EMA statistics (NEW) ----
                with torch.no_grad():
                    level0 = feats[0]  # (B, C, H, W)
                    if sup_idx:
                        voc_feats = level0[sup_idx]
                        self.voc_stat_mu = (
                            self.ema_momentum * self.voc_stat_mu
                            + (1 - self.ema_momentum) * voc_feats.mean(dim=[0, 2, 3])
                        )
                        self.voc_stat_sigma = (
                            self.ema_momentum * self.voc_stat_sigma
                            + (1 - self.ema_momentum) * voc_feats.std(dim=[0, 2, 3])
                        )
                    if unsup_idx:
                        exd_feats = level0[unsup_idx]
                        self.exdark_stat_mu = (
                            self.ema_momentum * self.exdark_stat_mu
                            + (1 - self.ema_momentum) * exd_feats.mean(dim=[0, 2, 3])
                        )
                        self.exdark_stat_sigma = (
                            self.ema_momentum * self.exdark_stat_sigma
                            + (1 - self.ema_momentum) * exd_feats.std(dim=[0, 2, 3])
                        )

                # ---- Cross-reconstruction (Phase B/C only) ----
                if enable_cross and unsup_idx and sup_idx:
                    sup_feats_cross = tuple(f[sup_idx].contiguous() for f in feats)
                    exd_attr = attr_vecs[unsup_idx].contiguous()
                    n_sup = len(sup_idx)
                    if exd_attr.size(0) >= n_sup:
                        sampled_exd_attr = exd_attr[torch.randperm(exd_attr.size(0))[:n_sup]]
                    else:
                        sampled_exd_attr = exd_attr.repeat(
                            (n_sup + exd_attr.size(0) - 1) // exd_attr.size(0), 1
                        )[:n_sup]

                    reorganized_cross = checkpoint(
                        self.feature_reorganization,
                        sup_feats_cross, sampled_exd_attr,
                        use_reentrant=False,
                    )
                    sup_inputs = attr_input[sup_idx].contiguous()

                    # Content term: perceptual consistency
                    loss_cross_content = self.feature_consistency_loss(
                        reorganized_features=reorganized_cross[0],
                        target_image=sup_inputs,
                    )
                    # Style term: match target domain statistics
                    mu_c = reorganized_cross[0].mean(dim=[2, 3]).clamp(-10, 10)
                    sd_c = reorganized_cross[0].std(dim=[2, 3]).clamp(0, 10)
                    loss_style = (
                        F.mse_loss(mu_c.mean(dim=0), self.exdark_stat_mu)
                        + F.mse_loss(sd_c.mean(dim=0), self.exdark_stat_sigma)
                    )

                    dummy_cross = sum(f.mean() for f in reorganized_cross[1:]) * 0.0
                    losses['loss_feat_cross_content'] = (
                        loss_cross_content * effective_cross_content + dummy_cross
                    )
                    # Style loss disabled by default (same as YOLO)
                    # losses['loss_feat_cross_style'] = loss_style * 0.0
            else:
                # Phase A: dummy forward to keep weights in graph
                recon_dummy = checkpoint(
                    self.feature_reorganization, feats, attr_vecs, use_reentrant=False
                )
                losses['dummy_recon'] = sum(f.mean() for f in recon_dummy) * 0.0

        # =============================================================
        # 7. DFR auxiliary losses
        # =============================================================
        if self.enable_dfr and self.training:
            losses.update(
                self._dfr_losses(
                    feats=feats,
                    attr_vecs=attr_vecs,
                    attr_vecs_deg1=attr_vecs_deg1,
                    attr_vecs_deg2=attr_vecs_deg2,
                    deg_params_1=deg_params_1,
                    deg_params_2=deg_params_2,
                    sup_idx=sup_idx,
                    unsup_idx=unsup_idx,
                    iter_num=iter_num,
                    enable_ortho=enable_ortho,
                    enable_deg_reg=enable_deg_reg,
                    enable_attr_sep=enable_attr_sep,
                    attr_sep_weight=attr_sep_weight,
                    epoch_1idx=epoch_1idx,
                    effective_cross_content=effective_cross_content,
                )
            )

        return losses

    # =================================================================
    # DFR losses (refactored to accept phase flags)
    # =================================================================
    def _dfr_losses(
        self,
        feats,
        attr_vecs,
        attr_vecs_deg1,
        attr_vecs_deg2,
        deg_params_1,
        deg_params_2,
        sup_idx,
        unsup_idx,
        iter_num,
        enable_ortho=True,
        enable_deg_reg=True,
        enable_attr_sep=False,
        attr_sep_weight=0.0,
        epoch_1idx=0,
        effective_cross_content=0.0,
    ):
        losses = {}

        # ---- Orthogonality loss ----
        if enable_ortho:
            pooled_feats = [
                F.adaptive_avg_pool2d(f, (1, 1)).flatten(1)
                for f in feats
            ]
            content_vec_fused = torch.cat(pooled_feats, dim=1)
            content_vec = F.linear(content_vec_fused, self.content_proj_fixed)
            loss_ortho = self.loss_ortho([content_vec], [attr_vecs])
            losses.update(loss_ortho)

        # ---- Degradation regression loss ----
        if (
            self.use_physical_degradation
            and attr_vecs_deg1 is not None
            and attr_vecs_deg2 is not None
            and enable_deg_reg
        ):
            loss_deg = self.loss_equivariant(
                attr_vecs_deg1, attr_vecs_deg2, deg_params_1, deg_params_2,
            )
            losses.update(loss_deg)

        # ---- Domain stat separation loss ----
        losses['loss_domain_stat'] = attr_vecs.sum() * 0.0
        has_both_domains = bool(sup_idx) and bool(unsup_idx)

        if self.training_stage == 'generation' and not has_both_domains:
            import torch.distributed as dist
            if dist.is_initialized() and dist.get_world_size() > 1:
                raise RuntimeError(
                    f"Rank {dist.get_rank()}: single-domain batch detected. "
                    f"RatioSampler must guarantee dual-domain on EVERY GPU."
                )

        if enable_attr_sep and has_both_domains:
            sup_vecs = attr_vecs[sup_idx]
            unsup_vecs = attr_vecs[unsup_idx]

            sup_mean_live = F.normalize(sup_vecs.mean(0), dim=0, eps=1e-6)
            unsup_mean_live = F.normalize(unsup_vecs.mean(0), dim=0, eps=1e-6)

            cos_live = torch.dot(sup_mean_live, unsup_mean_live)
            loss_live = F.relu(cos_live)

            if bool(self.domain_ema_initialized.item()):
                loss_ema_anchor = 0.5 * (
                    F.relu(torch.dot(sup_mean_live, self.ema_unsup_mean.detach()))
                    + F.relu(torch.dot(unsup_mean_live, self.ema_sup_mean.detach()))
                )
            else:
                loss_ema_anchor = attr_vecs.sum() * 0.0

            losses['loss_domain_stat'] = attr_sep_weight * (
                0.7 * loss_live + 0.3 * loss_ema_anchor
            )

            # Update EMA centers
            with torch.no_grad():
                m = self.domain_stat_ema_momentum
                if not bool(self.domain_ema_initialized.item()):
                    self.ema_sup_mean.copy_(sup_mean_live.detach())
                    self.ema_unsup_mean.copy_(unsup_mean_live.detach())
                    self.domain_ema_initialized.fill_(True)
                else:
                    self.ema_sup_mean.mul_(m).add_(sup_mean_live.detach(), alpha=1.0 - m)
                    self.ema_unsup_mean.mul_(m).add_(unsup_mean_live.detach(), alpha=1.0 - m)
                    self.ema_sup_mean.copy_(
                        F.normalize(self.ema_sup_mean, dim=0, eps=1e-6))
                    self.ema_unsup_mean.copy_(
                        F.normalize(self.ema_unsup_mean, dim=0, eps=1e-6))

        # ---- Attribute monitoring (NEW) ----
        if iter_num % self.attr_monitor_interval == 0:
            with torch.no_grad():
                attr_normed = F.normalize(attr_vecs, dim=1, eps=1e-6)
                phase_label = 'A' if effective_cross_content == 0.0 else (
                    'B' if effective_cross_content <= 0.03 else 'C')

                if sup_idx and unsup_idx:
                    z_sup = F.normalize(attr_normed[sup_idx].mean(0), dim=0)
                    z_unsup = F.normalize(attr_normed[unsup_idx].mean(0), dim=0)
                    cos_current = torch.dot(z_sup, z_unsup).item()
                    angle_deg = np.degrees(np.arccos(np.clip(cos_current, -1, 1)))
                    sim_mat = torch.mm(attr_normed, attr_normed.t())
                    mask = 1.0 - torch.eye(attr_vecs.size(0), device=attr_vecs.device)
                    off_diag_cos = (sim_mat * mask).sum() / mask.sum()
                    mean_angle = torch.acos(torch.clamp(off_diag_cos, -1, 1)).item()
                    r2_val = losses.get('deg_reg_r2', torch.tensor(0.0)).item()
                    r2_a = losses.get('deg_reg_r2_alpha', torch.tensor(0.0)).item()
                    print(
                        f"[AttrMonitor] iter={iter_num} phase={phase_label} "
                        f"domain_cos={cos_current:.4f} domain_angle={angle_deg:.1f} deg "
                        f"mean_pairwise_angle={mean_angle:.3f} "
                        f"deg_reg_r2={r2_val:.3f} r2_alpha={r2_a:.3f}"
                    )
                else:
                    r2_val = losses.get('deg_reg_r2', torch.tensor(0.0)).item()
                    r2_a = losses.get('deg_reg_r2_alpha', torch.tensor(0.0)).item()
                    print(
                        f"[AttrMonitor] iter={iter_num} phase={phase_label} "
                        f"no_dual_domain deg_reg_r2={r2_val:.3f} r2_alpha={r2_a:.3f}"
                    )

        return losses

    # =================================================================
    # Inference
    # =================================================================
    def predict(self, batch_inputs, batch_data_samples, rescale=True):
        feats = self.extract_feat(batch_inputs)

        if self.enable_head_film and self.head_film_modules is not None:
            with torch.no_grad():
                attr_vecs, _, _ = self._get_attr_vecs(batch_inputs)
            feats = self._apply_film(feats, attr_vecs)

        if batch_data_samples[0].get('proposals', None) is None:
            rpn_results_list = self.rpn_head.predict(
                feats, batch_data_samples, rescale=False,
            )
        else:
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]

        results_list = self.roi_head.predict(
            feats, rpn_results_list, batch_data_samples, rescale=rescale,
        )
        return self.add_pred_to_datasample(batch_data_samples, results_list)

    # =================================================================
    # ZeRO-3 compatibility (NEW)
    # =================================================================
    def forward(self, inputs, data_samples=None, mode='tensor'):
        if isinstance(inputs, list):
            inputs = torch.stack(inputs)
        return super().forward(inputs, data_samples, mode)
