"""Physical low-light degradation model for DFR-Net.

Implements a physics-aware low-light image degradation pipeline:
    x_low = clip(alpha * x^gamma + shot_noise + read_noise)

This replaces naive ColorJitter with an interpretable degradation model
that supports equivariant attribute learning.

Reference: proposal.md Section P0-2
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional


class PhysicalLowlightDegradation(nn.Module):
    """Physics-based low-light degradation module.

    Degradation pipeline:
        1. Gamma darkening: x^gamma (gamma > 1)
        2. Exposure scaling: alpha * x^gamma (alpha < 1)
        3. Poisson shot noise: N(0, sigma_p * sqrt(x))
        4. Gaussian read noise: N(0, sigma_g^2)
        5. Saturation/desaturation scale
        6. Clip to [0, 1]

    Args:
        gamma_range: Range for gamma darkening. Default (1.5, 3.5).
        alpha_range: Range for exposure scaling. Default (0.1, 0.8).
        sigma_g_range: Range for Gaussian read noise std. Default (0.005, 0.05).
        sigma_p_range: Range for Poisson shot noise multiplier. Default (0.01, 0.08).
        saturation_range: Range for saturation scaling. Default (0.4, 1.0).
        return_params: If True, forward() returns (degraded_img, params_dict).
    """

    def __init__(
        self,
        gamma_range: Tuple[float, float] = (1.5, 3.5),
        alpha_range: Tuple[float, float] = (0.1, 0.8),
        sigma_g_range: Tuple[float, float] = (0.005, 0.05),
        sigma_p_range: Tuple[float, float] = (0.01, 0.08),
        saturation_range: Tuple[float, float] = (0.4, 1.0),
        return_params: bool = False,
    ):
        super().__init__()
        self.gamma_range = gamma_range
        self.alpha_range = alpha_range
        self.sigma_g_range = sigma_g_range
        self.sigma_p_range = sigma_p_range
        self.saturation_range = saturation_range
        self.return_params = return_params

    def sample_params(self, device: torch.device, dtype: torch.dtype, batch_size: int = 1,
                      alpha_range: Optional[Tuple[float, float]] = None) -> dict:
        """Sample random degradation parameters for a batch.

        Args:
            device: torch device
            dtype: torch dtype
            batch_size: Number of samples in the batch. Default 1.
            alpha_range: Optional override for alpha range (for asymmetric degradation).

        Returns a dict of tensors with keys: gamma, alpha, sigma_g, sigma_p, saturation.
        Each value has shape (batch_size, 1, 1, 1) for broadcasting over (B, C, H, W).
        """
        def u(lo, hi):
            return torch.empty(batch_size, 1, 1, 1, device=device, dtype=dtype).uniform_(lo, hi)
        alpha_lo, alpha_hi = alpha_range if alpha_range is not None else self.alpha_range
        return {
            'gamma': u(*self.gamma_range),
            'alpha': u(alpha_lo, alpha_hi),
            'sigma_g': u(*self.sigma_g_range),
            'sigma_p': u(*self.sigma_p_range),
            'saturation': u(*self.saturation_range),
        }

    def forward(self, x: torch.Tensor, params: Optional[dict] = None) -> torch.Tensor:
        """Apply physical low-light degradation.

        Args:
            x: Input image tensor, shape (B, C, H, W), values in [0, 1].
            params: Optional pre-sampled degradation parameters. If None, sample randomly.

        Returns:
            If return_params=False: degraded image (B, C, H, W) in [0, 1].
            If return_params=True: (degraded_image, params_dict).
        """
        if params is None:
            params = self.sample_params(x.device, x.dtype)

        gamma = params["gamma"]
        alpha = params["alpha"]
        sigma_g = params["sigma_g"]
        sigma_p = params["sigma_p"]
        saturation = params["saturation"]

        # 1. Gamma darkening + exposure scaling
        x_dark = alpha * (x.clamp(min=1e-6) ** gamma)

        # 2. Poisson shot noise: approximated as Gaussian with std = sigma_p * sqrt(x_dark)
        #    This is valid for moderate photon counts.
        shot_std = sigma_p * x_dark.sqrt().clamp(min=1e-8)
        shot_noise = torch.randn_like(x_dark) * shot_std

        # 3. Gaussian read noise
        read_noise = torch.randn_like(x_dark) * sigma_g

        # 4. Combine
        x_degraded = x_dark + shot_noise + read_noise

        # 5. Saturation/desaturation: convert to grayscale, blend
        #    Simple approach: scale chrominance (reduce saturation)
        gray = x_degraded.mean(dim=1, keepdim=True)
        x_degraded = gray + saturation * (x_degraded - gray)

        # 6. Clip to valid range
        x_degraded = torch.clamp(x_degraded, 0.0, 1.0)

        if self.return_params:
            return x_degraded, params
        return x_degraded

    def apply_with_strength(
        self, x: torch.Tensor, strength: float
    ) -> torch.Tensor:
        """Apply degradation with a controllable strength parameter.

        Useful for generating degradation curves (vis6) where strength varies
        from 0 (no degradation) to 1 (full degradation).

        Args:
            x: Input image, (B, C, H, W) in [0, 1].
            strength: Scalar in [0, 1]. 0 = identity, 1 = full degradation.

        Returns:
            Degraded image with interpolated parameters.
        """
        # Interpolate parameters between identity (no degradation) and full degradation
        gamma = 1.0 + strength * (self.gamma_range[1] - 1.0)
        alpha = 1.0 - strength * (1.0 - self.alpha_range[0])
        sigma_g = strength * self.sigma_g_range[1]
        sigma_p = strength * self.sigma_p_range[1]
        saturation = 1.0 - strength * (1.0 - self.saturation_range[0])

        params = {
            "gamma": torch.tensor(gamma, device=x.device, dtype=x.dtype),
            "alpha": torch.tensor(alpha, device=x.device, dtype=x.dtype),
            "sigma_g": torch.tensor(sigma_g, device=x.device, dtype=x.dtype),
            "sigma_p": torch.tensor(sigma_p, device=x.device, dtype=x.dtype),
            "saturation": torch.tensor(saturation, device=x.device, dtype=x.dtype),
        }
        return self.forward(x, params)


class PairedDegradationSampler:
    """Sample paired degradations for equivariant attribute learning.

    Generates two degraded versions of the same image with different
    degradation parameters, enabling the equivariance loss L_eq.
    """

    def __init__(self, degradation_module: PhysicalLowlightDegradation):
        self.degradation = degradation_module

    def sample_pair_asymmetric(
        self, x: torch.Tensor, domain_labels: torch.Tensor,
        coco_alpha_range: Tuple[float, float] = (0.5, 1.0),
        exdark_alpha_range: Tuple[float, float] = (0.1, 0.5)
    ) -> Tuple[torch.Tensor, torch.Tensor, dict, dict]:
        """Domain-aware asymmetric degradation: coco light, ExDark heavy.

        Args:
            x: Input image, (B, C, H, W) in [0, 1].
            domain_labels: (B,), 0=coco, 1=ExDark.
            coco_alpha_range: Alpha range for coco (lighter degradation).
            exdark_alpha_range: Alpha range for ExDark (heavier degradation).

        Returns:
            (x_deg1, x_deg2, params1, params2)
        """
        B = x.size(0)
        device, dtype = x.device, x.dtype

        # Build per-sample alpha ranges based on domain
        alpha_mins = torch.empty(B, device=device, dtype=dtype)
        alpha_maxs = torch.empty(B, device=device, dtype=dtype)

        mask_coco = domain_labels == 0
        mask_exd = domain_labels == 1

        if mask_coco.any():
            alpha_mins[mask_coco] = coco_alpha_range[0]
            alpha_maxs[mask_coco] = coco_alpha_range[1]
        if mask_exd.any():
            alpha_mins[mask_exd] = exdark_alpha_range[0]
            alpha_maxs[mask_exd] = exdark_alpha_range[1]

        # Sample with per-sample alpha bounds
        def sample_with_alpha_bounds(alpha_m, alpha_M):
            params = self.degradation.sample_params(
                device, dtype, batch_size=B,
                alpha_range=None  # We override alpha below
            )
            # Override alpha with per-sample range
            alpha = alpha_m.view(B, 1, 1, 1) + torch.rand(B, 1, 1, 1, device=device, dtype=dtype) * \
                    (alpha_M.view(B, 1, 1, 1) - alpha_m.view(B, 1, 1, 1))
            params['alpha'] = alpha
            return params

        params1 = sample_with_alpha_bounds(alpha_mins, alpha_maxs)
        params2 = sample_with_alpha_bounds(alpha_mins, alpha_maxs)

        out1 = self.degradation(x, params1)
        out2 = self.degradation(x, params2)
        if self.degradation.return_params:
            x_deg1, _ = out1
            x_deg2, _ = out2
        else:
            x_deg1, x_deg2 = out1, out2

        return x_deg1, x_deg2, params1, params2

    def sample_pair(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, dict, dict]:
        """Generate a pair of degraded images with different parameters.

        Args:
            x: Input image, (B, C, H, W) in [0, 1].

        Returns:
            (x_deg1, x_deg2, params1, params2)
        """
        params1 = self.degradation.sample_params(x.device, x.dtype, batch_size=x.size(0))
        params2 = self.degradation.sample_params(x.device, x.dtype, batch_size=x.size(0))

        # Handle return_params: if True, forward() returns (image, params)
        out1 = self.degradation(x, params1)
        out2 = self.degradation(x, params2)
        if self.degradation.return_params:
            x_deg1, _ = out1
            x_deg2, _ = out2
        else:
            x_deg1, x_deg2 = out1, out2

        return x_deg1, x_deg2, params1, params2
