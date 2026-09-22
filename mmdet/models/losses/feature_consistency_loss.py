# in mmdet/models/losses/feature_consistency_loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
from mmdet.registry import MODELS
from typing import List, Dict
from torch.utils.checkpoint import checkpoint


@MODELS.register_module()
class FeatureConsistencyLoss(nn.Module):
    """
    A loss module that enforces consistency in a feature space defined by a
    frozen, pretrained perception network.

    This loss serves as a lightweight alternative to a full ImageDecoder for
    regularizing the content encoder and validating disentanglement.
    """

    def __init__(self,
                 perception_model: str = 'resnet18',
                 # Define which layer's output from the perception model to use
                 perception_layer: str = 'layer2',
                 # Projector to match the channel dimensions
                 content_channels: int = 256,
                 target_channels: int = 128,  # ResNet18 layer2 output channels
                 use_checkpoint: bool = True):  # Enable gradient checkpointing
        super().__init__()
        self.perception_layer = perception_layer
        self.use_checkpoint = use_checkpoint

        # --- 1. Load the frozen perception network ---
        if perception_model == 'resnet18':
            self.perception_net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            raise NotImplementedError(f"Perception model {perception_model} not supported.")

        # Freeze the network
        for param in self.perception_net.parameters():
            param.requires_grad = False

        # --- 2. Register a forward hook to extract intermediate features ---
        self.features = {}
        self._hook_debug_count = 0

        def get_features_hook(module, input, output):

            detached_output = output.detach()
            self.features[self.perception_layer] = detached_output


        # Find the target layer and register the hook
        target_layer_found = False
        for name, layer in self.perception_net.named_modules():
            if name == self.perception_layer:
                layer.register_forward_hook(get_features_hook)
                target_layer_found = True
                break
        if not target_layer_found:
            raise ValueError(f"Layer {self.perception_layer} not found in {perception_model}.")

        # --- 3. Create a projector layer ---
        # This layer projects the reorganized features to match the channel dimension
        # of the perception network's features.
        self.projector = nn.Sequential(
            nn.Conv2d(content_channels, target_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(target_channels, target_channels, kernel_size=3, padding=1)
        )

        # --- 4. The core loss function ---
        self.criterion = nn.MSELoss()

    def to(self, *args, **kwargs):
        """Ensure the perception network is moved to the correct device."""
        super().to(*args, **kwargs)
        self.perception_net.to(*args, **kwargs)
        return self

    def half(self):
        """
        Override half() to ensure all components are converted to FP16.
        This is critical because perception_net has frozen parameters that
        may not be automatically converted by DeepSpeed.
        """
        super().half()
        # Explicitly convert perception_net (frozen parameters need manual conversion)
        self.perception_net.half()
        # projector should be converted by super().half(), but be explicit
        self.projector.half()
        return self

    def forward(self,
                reorganized_features: torch.Tensor,
                target_image: torch.Tensor) -> torch.Tensor:
        """
        Calculate the feature consistency loss.

        Args:
            reorganized_features (torch.Tensor): The feature map produced by
                reorganizing content and attribute features. Expected to be the
                highest resolution feature map from the neck.
            target_image (torch.Tensor): The original input image.

        Returns:
            torch.Tensor: The calculated loss.
        """
        self.perception_net.eval()

        # Auto-detect the dtype of this module (FP16 or FP32)
        # Use projector's parameters as reference (trainable, always converted)
        dtype = next(self.projector.parameters()).dtype
        debug_count = getattr(self, '_dtype_debug_count', 0)


        # Explicitly convert inputs to match module dtype
        # This ensures compatibility regardless of input dtype
        reorganized_features = reorganized_features.to(dtype)
        target_image = target_image.to(dtype)


        mean = torch.tensor([0.485, 0.456, 0.406], device=target_image.device, dtype=dtype).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=target_image.device, dtype=dtype).view(1, 3, 1, 1)
        normalized_image = (target_image - mean) / std

        # --- Get the target perception features ---
        # Downsample input to reduce memory usage (perception_net is memory-intensive)
        # Since perception_net is frozen and only used for feature extraction,
        # using lower resolution doesn't significantly affect the loss quality
        h, w = target_image.shape[2:]
        if h > 224 or w > 224:
            # Downsample to max 224x224 to save memory
            if target_image.dtype == torch.bfloat16:

                target_image_fp32 = target_image.float()
                target_image_small = F.interpolate(target_image_fp32, size=(224, 224), mode='bilinear', align_corners=False)
                target_image_small = target_image_small.to(target_image.dtype)
            else:
                target_image_small = F.interpolate(target_image, size=(224, 224), mode='bilinear', align_corners=False)
        else:
            target_image_small = target_image

        mean = torch.tensor([0.485, 0.456, 0.406], device=target_image_small.device, dtype=dtype).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=target_image_small.device, dtype=dtype).view(1, 3, 1, 1)
        normalized_image = (target_image_small - mean) / std

        # This forward pass will populate self.features via the hook
        with torch.no_grad():
            # perception_net runs with downsampled input to save memory
            self.perception_net(normalized_image)
            target_perception_features = self.features[self.perception_layer].detach()
            # Convert to FP32 immediately to avoid overflow in subsequent operations
            if target_perception_features.dtype in (torch.float16, torch.bfloat16):
                target_perception_features = target_perception_features.float()



        # --- Project the reorganized features ---
        projected_features = self.projector(reorganized_features)

        # --- Align spatial dimensions ---
        # The reorganized features might have a different size than the perception features
        h, w = target_perception_features.shape[2:]

        # === All operations in FP32 to avoid overflow ===
        # Convert projected_features to FP32 if needed
        if projected_features.dtype in (torch.float16, torch.bfloat16):
            projected_features = projected_features.float()

        # Interpolate in FP32
        projected_features_aligned = F.interpolate(projected_features, size=(h, w), mode='bilinear')

        # Calculate loss in FP32 (both inputs are already FP32)
        loss = self.criterion(projected_features_aligned, target_perception_features)

        # Keep loss in FP32 for numerical stability in mixed precision training
        # DeepSpeed will handle loss scaling automatically

        return loss
