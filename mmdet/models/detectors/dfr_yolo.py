from typing import Dict, Tuple

from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import OptSampleList, SampleList
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
#from mmdet.core import bbox2result
from .base import BaseDetector

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from torch.utils.checkpoint import checkpoint
from mmengine import MessageHub
from mmdet.utils.dfr_modules import FiLM
from mmdet.utils.physical_degradation import PhysicalLowlightDegradation, PairedDegradationSampler


class FrozenBatchNorm2d(nn.Module):
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
    for m in module.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
            m.eval()

            m.running_mean.requires_grad = False
            m.running_var.requires_grad = False


@MODELS.register_module()
class DFRYOLO(BaseDetector):
    def __init__(self,
                 backbone: ConfigType,
                 neck: OptConfigType = None,
                 bbox_head: OptConfigType = None,
                 attribute_encoder: OptConfigType = None,
                 reorganization: OptConfigType = None,
                 feature_consistency_loss: OptConfigType = None,
                 enable_dfr: bool = True,
                 enable_reconstruction: bool = True,
                 enable_attribute_mixup: bool = True,
                 enable_head_film: bool = True,
                 dfr_loss_weight=1.0,
                 dfr_recon_loss_weight=0.1,
                 dfr_cross_loss_weight=0.5,
                 dfr_content_loss_weight=1.0,
                 dfr_ortho_loss_weight=5.0,
                 dfr_attr_sep_loss_weight=2.0,
                 dfr_content_fnr_threshold: float = 0.5,
                 dfr_content_use_domain_aware: bool = True,
                 dfr_content_domain_temperature: float = 0.3,
                 # Physics-based degradation config
                 use_physical_degradation: bool = True,
                 dfr_equivariant_loss_weight: float = 1.0,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None,
                 training_stage: str = 'joint',
                 freeze_backbone_bn_layers: int = 3,
                 freeze_neck_bn: bool = True,
                 attr_monitor_interval: int = 100, 
                 stage2_phase_iters: Tuple[int, int] = (5600, 12800),
                film_identity_weight: float = 1e-4,
                use_progressive_asymmetric_degradation: bool = True) -> None:
        super(DFRYOLO, self).__init__(
            data_preprocessor = data_preprocessor, init_cfg = init_cfg)

        self.enable_dfr = enable_dfr
        self.enable_reconstruction = enable_reconstruction
        self.enable_attribute_mixup = enable_attribute_mixup
        self.enable_head_film = enable_head_film
        self.dfr_loss_weight = dfr_loss_weight
        self.dfr_recon_loss_weight = dfr_recon_loss_weight
        self.dfr_cross_loss_weight = dfr_cross_loss_weight
        self.dfr_content_loss_weight = dfr_content_loss_weight
        self.dfr_ortho_loss_weight = dfr_ortho_loss_weight
        self.dfr_attr_sep_loss_weight = dfr_attr_sep_loss_weight
        self.dfr_content_fnr_threshold = dfr_content_fnr_threshold
        self.dfr_content_use_domain_aware = dfr_content_use_domain_aware
        self.dfr_content_domain_temperature = dfr_content_domain_temperature
        self.use_physical_degradation = use_physical_degradation
        self.dfr_equivariant_loss_weight = dfr_equivariant_loss_weight
        self.training_stage = training_stage
        self.freeze_backbone_bn_layers = freeze_backbone_bn_layers
        self.freeze_neck_bn = freeze_neck_bn
        self.attr_monitor_interval = attr_monitor_interval

        self.stage2_phase_iters = tuple(stage2_phase_iters)
        self.film_identity_weight = float(film_identity_weight)
        self.use_progressive_asymmetric_degradation = (
            bool(use_progressive_asymmetric_degradation)
        )

        if len(self.stage2_phase_iters) != 2:
            raise ValueError(
                'stage2_phase_iters must contain exactly two boundaries: '
                '(phase_a_end_iter, phase_b_end_iter).'
            )

        if self.stage2_phase_iters[0] <= 0:
            raise ValueError('The Phase-A iteration boundary must be positive.')

        if self.stage2_phase_iters[1] <= self.stage2_phase_iters[0]:
            raise ValueError(
                'Phase-B end iteration must be larger than Phase-A end iteration.'
            )

        # init model layers
        self.backbone = MODELS.build(backbone)
        if neck is not None:
            self.neck = MODELS.build(neck)
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = MODELS.build(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.original_input = None
        self.attribute_encoder = MODELS.build(attribute_encoder)
        self.feature_reorganization = MODELS.build(reorganization)
        self.feature_consistency_loss = MODELS.build(feature_consistency_loss)
        # self.image_decoder = MODELS.build(self.image_decoder_cfg)
        # self.loss_lpips = MODELS.build(dict(type='LPIPSLoss', net=self.lpips_net_type, loss_weight=self.lpips_loss_weight))
        self.loss_ortho = MODELS.build(dict(type='DFROrthogonalityLoss', loss_weight=self.dfr_ortho_loss_weight))
        self.loss_attr_sep = MODELS.build(dict(
            type='DFRAttributeSeparationLoss',
            loss_weight=self.dfr_attr_sep_loss_weight))
        # NOTE: DFRContentInvarianceLoss (InfoNCE) removed per P0-4.
        # It conflicts with degradation parameter regression objective.
        # self.loss_content = ...
        # Degradation parameter regression loss (P0-2): g(z_a) -> normalized params
        style_dim = getattr(self.attribute_encoder, 'out_dim', 256)
        self.loss_equivariant = MODELS.build(dict(
            type='DFRDegradationRegressionLoss',
            loss_weight=self.dfr_equivariant_loss_weight,
            attr_dim=style_dim,
        ))
        # Physics-based low-light degradation module (replaces ColorJitter)
        self.physical_degrade = PhysicalLowlightDegradation(return_params=True)
        self.paired_degradation = PairedDegradationSampler(self.physical_degrade)
        # P1-2: Fixed orthogonal projection for content vector (prevents trivial solution)
        proj = torch.empty(256, 896)
        nn.init.orthogonal_(proj)
        self.register_buffer('content_proj_fixed', proj)
        # P0-6: Self-reconstruction loss (feature-level MSE to anchor statistics)
        self.loss_self_recon = MODELS.build(dict(
            type='DFRReconstructionLoss',
            loss_weight=0.5,
        ))
        # P0-5: Domain-level statistics EMA buffers for cross-reconstruction style target
        self.register_buffer('coco_stat_mu', torch.zeros(512))
        self.register_buffer('coco_stat_sigma', torch.ones(512))
        self.register_buffer('exdark_stat_mu', torch.zeros(512))
        self.register_buffer('exdark_stat_sigma', torch.ones(512))
        self.ema_momentum = 0.99

        # DomainStat EMA buffers: smooth domain mean directions for stable separation
        style_dim = getattr(self.attribute_encoder, 'out_dim', 256)
        self.register_buffer('ema_sup_mean', torch.zeros(style_dim))
        self.register_buffer('ema_unsup_mean', torch.zeros(style_dim))
        self.register_buffer('domain_ema_initialized', torch.tensor(False, dtype=torch.bool))
        self.domain_stat_ema_momentum = 0.99



        if self.enable_head_film:
            style_dim = getattr(self.attribute_encoder, 'out_dim', 256)
            head_film_channels = [512, 256, 128]  # YOLOV3Neck 3-scale outputs
            self.head_film_modules = nn.ModuleList([
                FiLM(style_dim=style_dim, num_channels=ch)
                for ch in head_film_channels
            ])
        else:
            self.head_film_modules = None

        self.init_weights()


        if training_stage == 'generation':
            self._freeze_bn_layers()
        else:
            # Stage 3 (joint): restore FrozenBatchNorm2d to normal BatchNorm2d
            # so that BN statistics can adapt to the target domain (ExDark).
            # This is a no-op if no FrozenBN exists (e.g. Stage 1 -> joint).
            self._restore_bn_layers()


        self._freeze_modules_by_stage()

    def _freeze_bn_layers(self):
        



        for name, module in self.backbone.named_modules():
            if isinstance(module, (nn.BatchNorm2d, nn.SyncBatchNorm)):

                layer_idx = self._parse_backbone_layer_idx(name)
                
                if layer_idx < self.freeze_backbone_bn_layers:

                    parent_name = '.'.join(name.split('.')[:-1])
                    child_name = name.split('.')[-1]
                    if parent_name:
                        parent = self.backbone.get_submodule(parent_name)
                    else:
                        parent = self.backbone
                    frozen = FrozenBatchNorm2d.from_bn(module)
                    setattr(parent, child_name, frozen)
                else:

                    freeze_bn_stats(module)
        

        if self.freeze_neck_bn and hasattr(self, 'neck'):
            for name, module in self.neck.named_modules():
                if isinstance(module, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                    parent_name = '.'.join(name.split('.')[:-1])
                    child_name = name.split('.')[-1]
                    if parent_name:
                        parent = self.neck.get_submodule(parent_name)
                    else:
                        parent = self.neck
                    frozen = FrozenBatchNorm2d.from_bn(module)
                    setattr(parent, child_name, frozen)

    def _restore_bn_layers(self):
        """Restore FrozenBatchNorm2d to BatchNorm2d for Stage 3 fine-tuning.

        Stage 2 replaces some BatchNorm2d with FrozenBatchNorm2d (irreversible
        module swap). When switching to Stage 3, these must be restored so that
        BN statistics can adapt to the target domain (ExDark).

        This method traverses all modules and replaces FrozenBN with normal BN,
        preserving weight/bias/running_mean/running_var.
        """
        import torch.nn as nn

        def _replace_frozenbn_recursive(module, parent=None, name=''):
            """Recursively replace FrozenBatchNorm2d with BatchNorm2d."""
            restored_count = 0
            for child_name, child in module.named_children():
                if isinstance(child, FrozenBatchNorm2d):
                    # Create new BatchNorm2d with same params
                    new_bn = nn.BatchNorm2d(
                        num_features=child.num_features,
                        eps=child.eps,
                        affine=True,
                        track_running_stats=True
                    )
                    # Copy state from FrozenBN
                    with torch.no_grad():
                        new_bn.weight.copy_(child.weight)
                        new_bn.bias.copy_(child.bias)
                        new_bn.running_mean.copy_(child.running_mean)
                        new_bn.running_var.copy_(child.running_var)
                    # Replace in parent
                    setattr(module, child_name, new_bn)
                    restored_count += 1
                else:
                    # Recurse into children
                    restored_count += _replace_frozenbn_recursive(child, module, child_name)
            return restored_count

        _replace_frozenbn_recursive(self)

    def _parse_backbone_layer_idx(self, module_name):

        parts = module_name.split('.')
        for part in parts:
            if part.startswith('layer'):
                try:
                    return int(part[5:])  # 'layer3' -> 3
                except ValueError:
                    pass
            elif part.startswith('conv_res_block'):
                try:
                    return int(part[14:])  # 'conv_res_block3' -> 3
                except ValueError:
                    pass
            elif part == 'conv1':
                return 0
            elif part.isdigit():
                return int(part)
        return 0

    def init_weights(self) -> None:
        super(DFRYOLO, self).init_weights()

        self._load_ema_weights_if_available()
        self.attribute_encoder.apply(self._kaiming_init_with_bn)
        self.feature_reorganization.apply(self._xavier_init_for_linear)
        #self.image_decoder.apply(self._kaiming_init_with_bn)

        # Break symmetry in attribute_encoder: stronger perturbation to prevent
        # all outputs from collapsing to the same direction on the unit sphere.
        for m in self.attribute_encoder.modules():
            if isinstance(m, nn.Linear) and m.weight is not None:
                m.weight.data += torch.randn_like(m.weight.data) * 0.05
            if isinstance(m, nn.Conv2d):
                m.weight.data += torch.randn_like(m.weight.data) * 0.02
    def _load_ema_weights_if_available(self):
        """Load EMA weights from the configured checkpoint when available."""
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
            self.load_state_dict(processed, strict=False)
        except Exception:
            return

    def _kaiming_init_with_bn(self, m):
        """Initialize Conv2d with Kaiming and BatchNorm2d with 1,0."""
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def _xavier_init_for_linear(self, m):
        """Initialize Linear layers with Xavier."""
        if isinstance(m, nn.Linear):
            if hasattr(m, 'weight') and m.weight is not None:

                if m.weight.dim() >= 2:
                    nn.init.xavier_uniform_(m.weight)


            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _freeze_modules_by_stage(self):
        if self.training_stage == 'detection':
            self._freeze_generation_modules()

        elif self.training_stage == 'generation':
            self._freeze_detection_modules()

        elif self.training_stage == 'joint':
            if hasattr(self, 'attribute_encoder'):
                for param in self.attribute_encoder.parameters():
                    param.requires_grad = False
                self.attribute_encoder.eval()
        else:
            raise ValueError(f"Unknown training_stage: {self.training_stage}")

    def _freeze_generation_modules(self):

        if hasattr(self, 'attribute_encoder'):
            for param in self.attribute_encoder.parameters():
                param.requires_grad = False
            self.attribute_encoder.eval()


        if hasattr(self, 'feature_reorganization'):
            for param in self.feature_reorganization.parameters():
                param.requires_grad = False
            self.feature_reorganization.eval()


        if hasattr(self, 'feature_consistency_loss'):
            for param in self.feature_consistency_loss.parameters():
                param.requires_grad = False
            self.feature_consistency_loss.eval()

    def _freeze_detection_modules(self):



        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()


        for param in self.neck.parameters():
            param.requires_grad = False
        self.neck.eval()


        for param in self.bbox_head.parameters():
            param.requires_grad = False
        self.bbox_head.eval()


    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
        """Extract features with DFR-Net support.

        Args:
            batch_inputs (Tensor): Image tensor, has shape (bs, dim, H, W).

        Returns:
            tuple[Tensor]: Tuple of feature maps from neck. Each feature map
            has shape (bs, dim, H, W).
        """




        if self.training_stage == 'generation' and self.training:


            backbone_feats = self.backbone(batch_inputs)
            if self.with_neck:
                with torch.no_grad():
                    neck_feats = self.neck(backbone_feats)

                neck_feats = tuple(f.detach() for f in neck_feats)
            else:
                neck_feats = backbone_feats
        else:
            backbone_feats = self.backbone(batch_inputs)
            if self.with_neck:
                neck_feats = self.neck(backbone_feats)
            else:
                neck_feats = backbone_feats

        return neck_feats


    def loss(self, batch_inputs: Tensor, batch_data_samples: OptSampleList) -> Dict[str, Tensor]:
        losses = dict()

        message_hub = MessageHub.get_current_instance()
        iter_num = message_hub.get_info('iter')
        if iter_num is None:
            iter_num = 0

        # ---- Phase Training Strategy (Stage 2, v2.0 curriculum subtraction) ----
        # P0-2: Phase A first builds the attribute space with minimal losses,
        # then progressively adds back reconstruction and cross-domain losses.
        '''
        epoch = message_hub.get_info('epoch')
        if epoch is None:
            epoch = 0
        # message_hub epoch is 0-indexed (starts at 0), but we think in 1-indexed
        epoch_1idx = epoch + 1
        '''

        phase_a_end_iter, phase_b_end_iter = self.stage2_phase_iters

        # Phase flags (default: generation stage)
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
            if iter_num < phase_a_end_iter:
                # -----------------------------------------------------
                # Phase A: stable attribute-space construction.
                # -----------------------------------------------------
                current_phase = 'A'
                enable_attr_sep = True
                enable_self_recon = True
                enable_cross = False
                deg_reg_weight = 1.0
                ortho_weight = 0.1
                attr_sep_weight = 0.10
                self_recon_weight = 0.5
                effective_recon_weight = self.dfr_recon_loss_weight * 0.5
                effective_cross_content = 0.0

            elif iter_num < phase_b_end_iter:
                # -----------------------------------------------------
                # Phase B: add mild cross-domain content preservation.
                # -----------------------------------------------------
                current_phase = 'B'
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
                # -----------------------------------------------------
                # Phase C: complete Stage-2 objective.
                # -----------------------------------------------------
                current_phase = 'C'
                enable_attr_sep = True
                enable_self_recon = True
                enable_cross = True
                deg_reg_weight = 0.5
                ortho_weight = 0.1
                attr_sep_weight = 0.15
                self_recon_weight = 0.5
                effective_recon_weight = self.dfr_recon_loss_weight
                effective_cross_content = 0.05

        else:
            current_phase = 'N/A'

        # Apply dynamic weights (phase variables -> module weights)
        self.loss_equivariant.loss_weight = deg_reg_weight
        self.loss_ortho.loss_weight = ortho_weight
        self.loss_attr_sep.loss_weight = attr_sep_weight
        self.loss_self_recon.loss_weight = self_recon_weight

        # P1-1: AdaIN gamma_gate scheduling (slower ramp, reduce early AdaIN shock)
        gate = min(1.0, iter_num / 8000.0)
        for m in self.feature_reorganization.adain_modules:
            m.gamma_gate.data.fill_(gate)
            m.gamma_gate.requires_grad_(False)



        neck_feats = self.extract_feat(batch_inputs)


        if self.training_stage == 'generation' and self.training:
            backbone_feats = self.backbone(batch_inputs)
        else:
            backbone_feats = None


        attr_input_fp32 = F.interpolate(batch_inputs.float(), size=(224, 224), mode='bilinear')
        if batch_inputs.dtype == torch.bfloat16:
            attr_input = attr_input_fp32.bfloat16()
        else:
            attr_input = attr_input_fp32
        attr_vecs = self.attribute_encoder(attr_input)

        # --- Pre-compute domain labels for asymmetric degradation ---
        sup_idx, unsup_idx = self._group_samples(batch_data_samples)
        B = attr_vecs.size(0)
        domain_labels = torch.zeros(B, dtype=torch.long, device=attr_vecs.device)
        if unsup_idx:
            domain_labels[unsup_idx] = 1

        # Physics-based degradation: only when DFR losses are active
        if self.training and self.enable_dfr:
            # P0-CRITICAL FIX: attr_input_fp32 is ALREADY in [0,1] because
            # data_preprocessor uses mean=[0,0,0], std=[255,255,255].
            # Do NOT divide by 255 again — that would crush values to [0, 0.0039],
            # making the degraded image pure noise / all-black, destroying
            # the degradation parameter signal and causing deg_reg_r2 to stay negative.
            attr_input_01 = attr_input_fp32

            if self.use_progressive_asymmetric_degradation:
                if current_phase == 'A':
                    coco_alpha_range = (0.4, 1.0)
                    exdark_alpha_range = (0.4, 1.0)
                elif current_phase == 'B':
                    coco_alpha_range = (0.5, 1.0)
                    exdark_alpha_range = (0.25, 0.7)
                else:
                    coco_alpha_range = (0.5, 1.0)
                    exdark_alpha_range = (0.15, 0.5)
            else:
                coco_alpha_range = (0.5, 1.0)
                exdark_alpha_range = (0.1, 0.5)


            # Generate paired degradations with different parameters
            # Asymmetric: coco lighter, ExDark heavier — forces non-overlapping
            # degradation priors so the encoder learns domain-specific ranges.
            attr_input_deg1, attr_input_deg2, deg_params_1, deg_params_2 = \
                self.paired_degradation.sample_pair_asymmetric(
                    attr_input_01, domain_labels,
                    coco_alpha_range=coco_alpha_range, exdark_alpha_range=exdark_alpha_range
                )

            # Keep degraded images in same [0,1] scale as clean path.
            # Do NOT multiply by 255 — that would mismatch the scale.
            if batch_inputs.dtype == torch.bfloat16:
                attr_input_deg1 = attr_input_deg1.bfloat16()
                attr_input_deg2 = attr_input_deg2.bfloat16()

            # Encode both degraded views
            attr_vecs_deg1 = self.attribute_encoder(attr_input_deg1)
            attr_vecs_deg2 = self.attribute_encoder(attr_input_deg2)



        if self.enable_head_film and self.training and self.head_film_modules is not None:
            neck_feats = tuple(
                film(feat, attr_vecs)
                for film, feat in zip(self.head_film_modules, neck_feats)
            )


        sup_idx, sup_samples = self._get_supervised_samples(batch_data_samples)
        if sup_idx:
            sup_neck_feats = tuple(f[sup_idx].contiguous() for f in neck_feats)
            detection_losses = self.bbox_head.loss(sup_neck_feats, batch_data_samples=sup_samples)
            losses.update({f'det_{k}': v for k, v in detection_losses.items()})
        else:

            dummy_preds = self.bbox_head(neck_feats)

            dummy_det_loss = 0.0
            for level_preds in dummy_preds:
                if isinstance(level_preds, (tuple, list)):
                    dummy_det_loss += sum(p.mean() for p in level_preds)
                else:
                    dummy_det_loss += level_preds.mean()
            losses['det_dummy'] = dummy_det_loss * 0.0


        if self.enable_reconstruction and self.training:
            sup_idx, unsup_idx = self._group_samples(batch_data_samples)

            # P0-6: Self-reconstruction (feature-level MSE) — Phase B+ only
            if enable_self_recon:
                recon_self = checkpoint(self.feature_reorganization, neck_feats, attr_vecs, use_reentrant=False)
                loss_self_recon = self.loss_self_recon(
                    [f.detach() for f in neck_feats], list(recon_self)
                )
                losses.update(loss_self_recon)

                # Perceptual consistency loss (using first scale)
                loss_recon_feat = self.feature_consistency_loss(
                    reorganized_features=recon_self[0],
                    target_image=attr_input
                )
                dummy_loss = sum(f.mean() for f in recon_self[1:]) * 0.0
                losses['loss_feat_recon'] = loss_recon_feat * effective_recon_weight + dummy_loss

                # P0-5: Update domain-level statistics EMA (needed for cross-reconstruction style target)
                with torch.no_grad():
                    level0 = neck_feats[0]  # (B, 512, H, W)
                    if sup_idx:
                        coco_feats = level0[sup_idx]
                        self.coco_stat_mu = self.ema_momentum * self.coco_stat_mu + (1 - self.ema_momentum) * coco_feats.mean(dim=[0, 2, 3])
                        self.coco_stat_sigma = self.ema_momentum * self.coco_stat_sigma + (1 - self.ema_momentum) * coco_feats.std(dim=[0, 2, 3])
                    if unsup_idx:
                        exd_feats = level0[unsup_idx]
                        self.exdark_stat_mu = self.ema_momentum * self.exdark_stat_mu + (1 - self.ema_momentum) * exd_feats.mean(dim=[0, 2, 3])
                        self.exdark_stat_sigma = self.ema_momentum * self.exdark_stat_sigma + (1 - self.ema_momentum) * exd_feats.std(dim=[0, 2, 3])
            else:
                # Phase A: skip self-reconstruction, but run a dummy forward to keep weights in graph
                recon_self = checkpoint(self.feature_reorganization, neck_feats, attr_vecs, use_reentrant=False)
                dummy_loss = sum(f.mean() for f in recon_self) * 0.0
                losses['dummy_recon'] = dummy_loss

            # Cross-reconstruction with directional attribute swap — Phase C+ only
            if enable_cross and unsup_idx and sup_idx:
                sup_neck_feats = tuple(f[sup_idx].contiguous() for f in neck_feats)
                exd_attr = attr_vecs[unsup_idx].contiguous()
                n_sup = len(sup_idx)
                if exd_attr.size(0) >= n_sup:
                    sampled_exd_attr = exd_attr[torch.randperm(exd_attr.size(0))[:n_sup]]
                else:
                    sampled_exd_attr = exd_attr.repeat((n_sup + exd_attr.size(0) - 1) // exd_attr.size(0), 1)[:n_sup]

                reorganized_cross = checkpoint(self.feature_reorganization, sup_neck_feats, sampled_exd_attr, use_reentrant=False)
                sup_inputs = attr_input[sup_idx].contiguous()

                # (a) Content term: perceptual consistency with source image
                loss_cross_content = self.feature_consistency_loss(
                    reorganized_features=reorganized_cross[0],
                    target_image=sup_inputs
                )
                dummy_cross = sum(f.mean() for f in reorganized_cross[1:]) * 0.0
                losses['loss_feat_cross_content'] = loss_cross_content * effective_cross_content + dummy_cross
            elif not enable_cross:
                # Phase A/B: skip cross-reconstruction, dummy forward for graph
                dummy_reorganized = checkpoint(self.feature_reorganization, neck_feats, attr_vecs, use_reentrant=False)
                dummy_loss = sum(f.mean() for f in dummy_reorganized) * 0.0
                losses['dummy_reorg'] = dummy_loss


        if self.enable_dfr and self.training:
            # sup_idx, unsup_idx, domain_labels already computed above for asymmetric degradation

            # P1-2: Orthogonality loss with fixed projection — always active
            if enable_ortho:
                pooled_feats = [F.adaptive_avg_pool2d(f, (1, 1)).squeeze(-1).squeeze(-1) for f in neck_feats]
                content_vec_fused = torch.cat(pooled_feats, dim=1)
                content_vec = F.linear(content_vec_fused, self.content_proj_fixed)
                loss_ortho = self.loss_ortho([content_vec], [attr_vecs])
                losses.update(loss_ortho)

            # P0-2: Paired differential degradation parameter regression
            # Main signal: content cancels, well-posed.
            if self.training and self.use_physical_degradation and enable_deg_reg:
                loss_deg = self.loss_equivariant(
                    attr_vecs_deg1, attr_vecs_deg2, deg_params_1, deg_params_2
                )
                losses.update(loss_deg)

            # 1.3 fix: Domain statistic separation with EMA smoothing.
            # Always register loss_domain_stat for DDP safety (zero if no dual-domain).
            losses['loss_domain_stat'] = attr_vecs.sum() * 0.0

            has_both_domains = bool(sup_idx) and bool(unsup_idx)

            # DDP safety: assert every rank sees both domains
            if self.training_stage == 'generation' and not has_both_domains:
                import torch.distributed as dist
                if dist.is_initialized() and dist.get_world_size() > 1:
                    rank = dist.get_rank()
                    raise RuntimeError(
                        f'Rank {rank} received single-domain batch: '
                        f'supervised={len(sup_idx)}, unsupervised={len(unsup_idx)}. '
                        f'RatioSampler must guarantee dual-domain on EVERY GPU.'
                    )

            if enable_attr_sep and has_both_domains:
                sup_vecs = attr_vecs[sup_idx]      # [N_s, D]
                unsup_vecs = attr_vecs[unsup_idx]  # [N_t, D]

                sup_mean_live = F.normalize(sup_vecs.mean(0), dim=0, eps=1e-6)
                unsup_mean_live = F.normalize(unsup_vecs.mean(0), dim=0, eps=1e-6)

                # Live separation loss (has gradient)
                cos_live = torch.dot(sup_mean_live, unsup_mean_live)
                loss_live = F.relu(cos_live)  # stop at orthogonality, not antipodal

                # EMA anchor loss: live vector vs detached EMA center (gradient flows through live)
                if bool(self.domain_ema_initialized.item()):
                    loss_ema_anchor = 0.5 * (
                        F.relu(torch.dot(sup_mean_live, self.ema_unsup_mean.detach())) +
                        F.relu(torch.dot(unsup_mean_live, self.ema_sup_mean.detach()))
                    )
                else:
                    loss_ema_anchor = attr_vecs.sum() * 0.0

                # Weighted mix + configurable attr_sep_weight (replaces hardcoded 0.3)
                losses['loss_domain_stat'] = attr_sep_weight * (
                    0.7 * loss_live + 0.3 * loss_ema_anchor
                )

                # Update EMA centers AFTER loss computation
                with torch.no_grad():
                    m = self.domain_stat_ema_momentum
                    if not bool(self.domain_ema_initialized.item()):
                        self.ema_sup_mean.data.copy_(sup_mean_live.detach())
                        self.ema_unsup_mean.data.copy_(unsup_mean_live.detach())
                        self.domain_ema_initialized.data.fill_(True)
                    else:
                        self.ema_sup_mean.data.mul_(m).add_(sup_mean_live.detach(), alpha=1.0 - m)
                        self.ema_unsup_mean.data.mul_(m).add_(unsup_mean_live.detach(), alpha=1.0 - m)
                        self.ema_sup_mean.data.copy_(
                            F.normalize(self.ema_sup_mean, dim=0, eps=1e-6))
                        self.ema_unsup_mean.data.copy_(
                            F.normalize(self.ema_unsup_mean, dim=0, eps=1e-6))


        # Keep losses in FP32 for numerical stability in mixed precision training
        # DeepSpeed will handle loss scaling and dtype conversion automatically
        # DO NOT convert losses to FP16 as it can cause overflow

        return losses

    def _group_samples(self, batch_data_samples: OptSampleList) -> Tuple[list, list]:
        """
        Group batch samples by domain.

        Priority:
        1. Explicit domain_id in metainfo (most reliable)
        2. Fallback: presence of gt bboxes (coco has labels, ExDark does not)
        """
        sup_idx, unsup_idx = [], []
        for i, sample in enumerate(batch_data_samples):
            # Method 1: explicit domain_id (preferred)
            meta = getattr(sample, 'metainfo', {})
            if 'domain_id' in meta:
                if meta['domain_id'] == 0:
                    sup_idx.append(i)
                else:
                    unsup_idx.append(i)
                continue

            # Method 2: bbox presence (fallback — may misclassify after heavy aug)
            if hasattr(sample, 'gt_instances') and getattr(sample.gt_instances, 'bboxes', None) is not None and len(
                    sample.gt_instances.bboxes) > 0:
                sup_idx.append(i)
            else:
                unsup_idx.append(i)
        return sup_idx, unsup_idx

    def _get_supervised_samples(self, batch_data_samples: OptSampleList) -> Tuple[list, list]:
        sup_idx = []
        sup_samples = []
        for i, sample in enumerate(batch_data_samples):
            if hasattr(sample, 'gt_instances') and getattr(sample.gt_instances, 'bboxes', None) is not None and len(
                    sample.gt_instances.bboxes) > 0:
                sup_idx.append(i)
                sup_samples.append(sample)
        return sup_idx, sup_samples

    def _mixup_attributes(self, attr_vecs: Tensor) -> Tensor:
        B, D = attr_vecs.shape
        dtype = next(self.parameters()).dtype
        idx1 = torch.randperm(B, device=attr_vecs.device)
        idx2 = torch.randperm(B, device=attr_vecs.device)
        lam = torch.distributions.Beta(0.4, 0.4).sample((B,)).to(dtype=dtype, device=attr_vecs.device).view(-1, 1)
        mixed_attr_vecs = lam * attr_vecs[idx1] + (1 - lam) * attr_vecs[idx2]
        noise = torch.randn_like(mixed_attr_vecs) * 0.05
        return mixed_attr_vecs + noise

    def predict(self, batch_inputs: Tensor,
                batch_data_samples: OptSampleList,
                rescale: bool = True) -> OptSampleList:
        """Predict results from a batch of inputs.

        During inference, DFR-Net components are disabled for real-time performance.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.
            rescale (bool): Whether to rescale the results.
                Defaults to True.

        Returns:
            list[:obj:`DetDataSample`]: Detection results of the
            input images. Each DetDataSample usually contain
            'pred_instances'. And the ``pred_instances`` usually
            contains following keys.

                - scores (Tensor): Classification scores, has a
                  shape (num_instance, )
                - labels (Tensor): Labels of bboxes, has a
                  shape (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        # Stage 3: FiLM-assisted inference.
        # When enable_head_film=True, attribute vectors modulate neck features
        # per-image, providing domain-adaptive detection without changing backbone.
        img_feats = self.extract_feat(batch_inputs)

        if self.enable_head_film and self.head_film_modules is not None:
            with torch.no_grad():
                attr_input = F.interpolate(batch_inputs.float(), size=(224, 224), mode='bilinear')
                if batch_inputs.dtype == torch.bfloat16:
                    attr_input = attr_input.bfloat16()
                attr_vecs = self.attribute_encoder(attr_input)
            img_feats = tuple(
                film(feat, attr_vecs)
                for film, feat in zip(self.head_film_modules, img_feats)
            )

        try:


            bbox_list = self.bbox_head.predict(
                img_feats, batch_data_samples, rescale=rescale)


            for bbox_data in bbox_list:
                if hasattr(bbox_data, 'bboxes') and isinstance(bbox_data.bboxes, torch.Tensor):
                    if bbox_data.bboxes.dtype != torch.float32:
                        bbox_data.bboxes = bbox_data.bboxes.float()
                if hasattr(bbox_data, 'scores') and isinstance(bbox_data.scores, torch.Tensor):
                    if bbox_data.scores.dtype != torch.float32:
                        bbox_data.scores = bbox_data.scores.float()
                if hasattr(bbox_data, 'labels') and isinstance(bbox_data.labels, torch.Tensor):
                    if bbox_data.labels.dtype in (torch.float16, torch.bfloat16, torch.float32):
                        bbox_data.labels = bbox_data.labels.long()

            results = self.add_pred_to_datasample(
                batch_data_samples, bbox_list)
        finally:
            pass

        return results

    def forward(self, inputs, data_samples=None, mode='tensor'):

        if isinstance(inputs, list):
            inputs = torch.stack(inputs)
        # ------------------------------------

        return super().forward(inputs, data_samples, mode)

    def _forward(self, batch_inputs: Tensor,
                 batch_data_samples: OptSampleList) -> tuple:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (list[:obj:`DetDataSample`]): Each item contains
                the meta information of each image and corresponding
                annotations.

        Returns:
            tuple: A tuple of features from ``bbox_head`` forward.
        """
        # Stage 3: FiLM-assisted forward (for export / tracing).
        img_feats = self.extract_feat(batch_inputs)

        if self.enable_head_film and self.head_film_modules is not None:
            attr_input = F.interpolate(batch_inputs.float(), size=(224, 224), mode='bilinear')
            if batch_inputs.dtype == torch.bfloat16:
                attr_input = attr_input.bfloat16()
            attr_vecs = self.attribute_encoder(attr_input)
            img_feats = tuple(
                film(feat, attr_vecs)
                for film, feat in zip(self.head_film_modules, img_feats)
            )

        results = self.bbox_head(img_feats)
        return results


    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/get_flops.py`
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        return outs

    '''def simple_test(self, img, img_metas, rescale=False):
        """Test function without test time augmentation.

        Args:
            imgs (list[torch.Tensor]): List of multiple images
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        bbox_list = self.bbox_head.get_bboxes(
            *outs, img_metas, rescale=rescale)
        # skip post-processing when exporting to ONNX
        if torch.onnx.is_in_onnx_export():
            return bbox_list

        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in bbox_list
        ]
        return bbox_results'''

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test function with test time augmentation.

        Args:
            imgs (list[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch. each dict has image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        assert hasattr(self.bbox_head, 'aug_test'), \
            f'{self.bbox_head.__class__.__name__}' \
            ' does not support test-time augmentation'

        feats = self.extract_feats(imgs)
        return [self.bbox_head.aug_test(feats, img_metas, rescale=rescale)]
