# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.utils import ConfigType
from torchvision import transforms


@MODELS.register_module()
class DFRReconstructionLoss(nn.Module):
    """Reconstruction loss for DFR-Net feature reorganization.
    
    This loss ensures that the reorganized features can be reconstructed
    back to the original features, promoting feature decoupling.
    """
    
    def __init__(self, 
                 loss_weight: float = 1.0,
                 reduction: str = 'mean'):
        super().__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction
        
    def forward(self, 
                original_features: List[Tensor],
                reconstructed_features: List[Tensor]) -> Dict[str, Tensor]:
        """
        Args:
            original_features: List of original features [(B, C, H, W), ...]
            reconstructed_features: List of reconstructed features [(B, C, H, W), ...]
            
        Returns:
            Dictionary containing reconstruction loss
        """
        total_loss = 0.0
        num_levels = min(len(original_features), len(reconstructed_features))
        
        for i in range(num_levels):
            orig_feat = original_features[i]
            recon_feat = reconstructed_features[i]
            
            # L2 reconstruction loss
            level_loss = F.mse_loss(orig_feat, recon_feat, reduction=self.reduction)
            total_loss += level_loss
            
        avg_loss = total_loss / num_levels
        
        return {
            'dfr_reconstruction_loss': avg_loss * self.loss_weight
        }


@MODELS.register_module()
class DFRContentInvarianceLoss(nn.Module):
    """
    Content-invariance loss with a domain-aware memory bank.

    Samples from the same domain are weak positives and are excluded from
    memory-bank negatives.
    """
    def __init__(self,
                 loss_weight: float = 1.0,
                 temperature: float = 0.07,
                 queue_size: int = 4096,
                 feat_dim: int = 256,
                 warmup_iters: int = 512,
                 rampup_iters: int = 512,
                 fnr_threshold: float = 0.5,
                 use_domain_aware: bool = True,
                 domain_temperature: float = 0.3):
        super().__init__()
        self.loss_weight = loss_weight
        self.temperature = temperature
        self.queue_size = queue_size
        self.warmup_iters = warmup_iters
        self.rampup_iters = rampup_iters
        self.fnr_threshold = fnr_threshold
        self.use_domain_aware = use_domain_aware
        self.domain_temperature = domain_temperature


        self.register_buffer("queue", torch.randn(feat_dim, queue_size))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.register_buffer("queue_domain_labels", torch.full((queue_size,), -1, dtype=torch.long))

        self.register_buffer("iter_count", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys: torch.Tensor, domain_labels: Optional[torch.Tensor] = None):
        """Update the memory bank and its domain labels."""

        if torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            keys_list = [torch.zeros_like(keys) for _ in range(world_size)]
            torch.distributed.all_gather(keys_list, keys)
            keys = torch.cat(keys_list, dim=0)

            if domain_labels is not None:
                domain_list = [torch.zeros_like(domain_labels) for _ in range(world_size)]
                torch.distributed.all_gather(domain_list, domain_labels)
                domain_labels = torch.cat(domain_list, dim=0)

        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)

        if ptr + batch_size > self.queue_size:
            first_part = self.queue_size - ptr
            self.queue[:, ptr:] = keys[:first_part].T
            self.queue[:, :batch_size - first_part] = keys[first_part:].T
            if domain_labels is not None:
                self.queue_domain_labels[ptr:] = domain_labels[:first_part]
                self.queue_domain_labels[:batch_size - first_part] = domain_labels[first_part:]
            ptr = batch_size - first_part
        else:
            self.queue[:, ptr:ptr + batch_size] = keys.T
            if domain_labels is not None:
                self.queue_domain_labels[ptr:ptr + batch_size] = domain_labels
            ptr = (ptr + batch_size) % self.queue_size

        self.queue_ptr[0] = ptr

    def forward(self, attr_vecs_v1: torch.Tensor, attr_vecs_v2: torch.Tensor,
                domain_labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            attr_vecs_v1: Attribute vectors for the primary view, shape (B, D).
            attr_vecs_v2: Attribute vectors for the augmented view, shape (B, D).
            domain_labels: Domain labels, shape (B,), with 0 for source and 1 for target.
        """
        self.iter_count += 1
        iter_num = int(self.iter_count.item())

        batch_size = attr_vecs_v1.size(0)
        if torch.distributed.is_initialized():
            batch_size_tensor = torch.tensor(batch_size, device=attr_vecs_v1.device)
            torch.distributed.all_reduce(batch_size_tensor, op=torch.distributed.ReduceOp.MIN)
            min_batch_size = int(batch_size_tensor.item())
        else:
            min_batch_size = batch_size

        v1 = F.normalize(attr_vecs_v1, dim=1)
        v2 = F.normalize(attr_vecs_v2, dim=1)


        if iter_num <= self.warmup_iters:
            with torch.no_grad():
                self._dequeue_and_enqueue(v2, domain_labels)
            dummy = v1.sum() * 0.0
            return {'dfr_content_invariance_loss': dummy}

        if min_batch_size < 2:
            with torch.no_grad():
                self._dequeue_and_enqueue(v2, domain_labels)
            return {'dfr_content_invariance_loss': torch.tensor(0.0, device=attr_vecs_v1.device)}


        pos_logits = torch.matmul(v1, v2.T) / self.temperature  # (B, B)


        if self.use_domain_aware and domain_labels is not None:

            same_domain_mask = (domain_labels.unsqueeze(0) == domain_labels.unsqueeze(1)).float()


            off_diag = 1.0 - torch.eye(batch_size, device=v1.device)
            weak_positive = same_domain_mask * off_diag * self.domain_temperature


            soft_targets = torch.zeros(batch_size, batch_size, device=v1.device)
            soft_targets[range(batch_size), range(batch_size)] = 1.0
            soft_targets += weak_positive
            soft_targets = soft_targets / soft_targets.sum(dim=1, keepdim=True).clamp(min=1e-8)
            use_soft_targets = True
        else:
            soft_targets = None
            use_soft_targets = False


        queue_snapshot = self.queue.to(v1.device).detach().clone()
        neg_logits = torch.matmul(v1, queue_snapshot) / self.temperature  # (B, K)


        if self.use_domain_aware and domain_labels is not None:
            # queue_domain_labels: (K,)
            qdl = self.queue_domain_labels.to(v1.device).unsqueeze(0)  # (1, K)
            dl = domain_labels.unsqueeze(1)  # (B, 1)

            same_domain_neg = (qdl == dl) & (qdl != -1)
            if same_domain_neg.any():

                neg_logits = neg_logits.masked_fill(same_domain_neg, -1e9)


        with torch.no_grad():
            raw_sim = torch.matmul(v1, queue_snapshot)
            fn_mask = raw_sim > self.fnr_threshold
            if fn_mask.any():

                neg_logits = neg_logits.masked_fill(fn_mask, -1e9)


        logits = torch.cat([pos_logits, neg_logits], dim=1)  # (B, B + K)


        if torch.isnan(logits).any():
            return {'dfr_content_invariance_loss': torch.tensor(0.0, device=v1.device)}

        if use_soft_targets:

            full_targets = torch.zeros(batch_size, batch_size + self.queue_size, device=v1.device)
            full_targets[:, :batch_size] = soft_targets

            log_probs = F.log_softmax(logits, dim=1)

            if torch.isnan(log_probs).any():
                labels = torch.arange(batch_size, device=v1.device)
                loss = F.cross_entropy(logits, labels)
            else:
                loss = -(full_targets * log_probs).sum(dim=1).mean()
        else:
            labels = torch.arange(batch_size, device=v1.device)
            loss = F.cross_entropy(logits, labels)


        if torch.isnan(loss):
            loss = torch.tensor(0.0, device=v1.device, requires_grad=True)

        # Ramp-up
        if iter_num <= self.warmup_iters + self.rampup_iters:
            alpha = float(iter_num - self.warmup_iters) / float(self.rampup_iters)
            loss = loss * alpha


        with torch.no_grad():
            self._dequeue_and_enqueue(v2, domain_labels)

        return {'dfr_content_invariance_loss': loss * self.loss_weight}


@MODELS.register_module()
class DFROrthogonalityLoss(nn.Module):
    """Lightweight hybrid orthogonality loss.

    Combines:
    1. Linear HSIC for global feature independence (avoid RBF overflow in BF16)
    2. Instance-level orthogonality (aligned with detection task)
    """

    def __init__(self,
                 loss_weight: float = 1.0,
                 lambda_hsic: float = 0.02,
                 lambda_instance: float = 0.1):
        super().__init__()
        self.loss_weight = loss_weight
        self.lambda_hsic = lambda_hsic
        self.lambda_instance = lambda_instance

    def _linear_hsic(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Linear kernel HSIC (avoid RBF exponential for BF16 stability).
        Formula: HSIC_linear = || mean(x^T y) - mean(x)^T mean(y) ||^2

        Args:
            x: (B, D1) content features
            y: (B, D2) attribute features
        Returns:
            HSIC loss value
        """

        if x.size(0) < 2:
            return torch.tensor(0.0, device=x.device, dtype=x.dtype)

        # Center the features (in FP32 for stability)
        x_fp32 = x.float()
        y_fp32 = y.float()

        x_centered = x_fp32 - x_fp32.mean(0, keepdim=True)
        y_centered = y_fp32 - y_fp32.mean(0, keepdim=True)

        # Linear kernel inner product
        cov = torch.matmul(x_centered.T, y_centered) / (x_fp32.size(0) - 1)

        # HSIC = ||cov||^2_F
        hsic = cov.pow(2).sum()

        return hsic
        
    def forward(self,
                content_features: List[Tensor],
                domain_attributes: List[Tensor]) -> Dict[str, Tensor]:
        """
        Args:
            content_features: List of content features [(B, C), ...]
            domain_attributes: List of domain attribute vectors [(B, D), ...]

        Returns:
            Dictionary containing orthogonality losses
        """
        if len(content_features) == 0 or len(domain_attributes) == 0:
            return {'dfr_orthogonality_loss': torch.tensor(0.0, device=content_features[0].device)}

        device = content_features[0].device
        losses = {}

        # 1. Global HSIC (batch-level, O(B^2) complexity)
        content_vec = content_features[0]  # (B, C)
        domain_vec = domain_attributes[0]   # (B, D)

        # Normalize features
        content_vec = F.normalize(content_vec.float(), dim=1)
        domain_vec = F.normalize(domain_vec.float(), dim=1)

        loss_hsic = self._linear_hsic(content_vec, domain_vec)
        losses['loss_hsic'] = loss_hsic * self.lambda_hsic

        # 2. Instance-level orthogonality (cosine similarity)
        # Compute cosine similarity and minimize its absolute value
        sim = F.cosine_similarity(content_vec, domain_vec, dim=1).abs().mean()
        losses['loss_inst_ortho'] = sim * self.lambda_instance

        # Total orthogonality loss
        total_loss = sum(losses.values())
        losses['dfr_orthogonality_loss'] = total_loss * self.loss_weight

        return losses


# Cross-reconstruction is now handled inline in dfr_yolo.py with style+content separation.


@MODELS.register_module()
class DFRDegradationRegressionLoss(nn.Module):
    """Paired differential degradation parameter regression loss (4-dim, no gamma).

    Gamma dimension removed from regression target because it is visually
    coupled with alpha in x_deg = clip(alpha * x^gamma + noise). The encoder
    reliably learns alpha (exposure scaling) but cannot stably predict gamma
    (power-law contrast change). Gamma is still sampled in degradation for
    diversity, just not regressed.

    Core insight: differential regression on paired views cancels content,
    leaving only param differences — well-posed.
    """

    def __init__(self, loss_weight=1.0, attr_dim=256, abs_weight=0.02):
        super().__init__()
        self.loss_weight = loss_weight
        self.abs_weight = abs_weight
        # 4-dim: alpha, sigma_g, sigma_p, saturation (gamma excluded)
        self.param_keys = ('alpha', 'sigma_g', 'sigma_p', 'saturation')
        param_dims = 4
        self.reg_head = nn.Sequential(
            nn.Linear(attr_dim, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 128), nn.ReLU(inplace=True),
            nn.Linear(128, param_dims))
        self.register_buffer('p_low',  torch.tensor([0.1, 0.005, 0.01, 0.4]))
        self.register_buffer('p_high', torch.tensor([0.8, 0.05,  0.08, 1.0]))

    def _norm(self, params):
        """Convert parameter dict to normalized vector (B, 4) in [0,1]."""
        vals = [params[k].view(-1) for k in self.param_keys]
        theta = torch.stack(vals, dim=1)
        return (theta - self.p_low) / (self.p_high - self.p_low)

    def forward(self, attr_deg1, attr_deg2, params1, params2):
        if attr_deg1.size(0) < 1:
            return {'dfr_deg_reg_loss': torch.tensor(0.0, device=attr_deg1.device),
                    'deg_reg_r2': torch.tensor(0.0, device=attr_deg1.device)}

        t1, t2 = self._norm(params1), self._norm(params2)
        p1, p2 = self.reg_head(attr_deg1), self.reg_head(attr_deg2)

        # Main signal: differential regression (content cancels)
        loss_diff = F.smooth_l1_loss(p1 - p2, t1 - t2)
        # Weak absolute anchor (ill-posed, low weight, only prevents drift)
        loss_abs = 0.5 * (F.smooth_l1_loss(p1, t1) + F.smooth_l1_loss(p2, t2))
        loss = loss_diff + self.abs_weight * loss_abs

        with torch.no_grad():
            pd, td = p1 - p2, t1 - t2
            ss_res = ((pd - td) ** 2).sum(0)
            ss_tot = ((td - td.mean(0)) ** 2).sum(0) + 1e-8

            ss_tot = torch.clamp(ss_tot, min=1e-3)
            r2_per_dim = 1 - ss_res / ss_tot  # (4,)
            r2_alpha = r2_per_dim[0]
            r2_mean = r2_per_dim.mean()

        return {
            'dfr_deg_reg_loss': loss * self.loss_weight,
            'deg_reg_r2': r2_mean,
            'deg_reg_r2_alpha': r2_alpha,
            'loss_diff': loss_diff.detach(),
            'loss_abs': loss_abs.detach(),
        }


# NOTE: Old DFREquivariantAttributeLoss removed per P0-2 engineering fix.
# Replaced by DFRDegradationRegressionLoss above. 
