# in mmdet/models/losses/lpips_loss.py
import torch
import lpips
from torch import nn
from mmdet.registry import MODELS


@MODELS.register_module()
class LPIPSLoss(nn.Module):
    def __init__(self, loss_weight=1.0, net='alex', pretrained=True, pnet_tune=False):
        """
        LPIPS Loss.
        Args:
            loss_weight (float): Weight of the loss.
            net (str): The network backbone to use. Options are 'alex', 'vgg', 'squeeze'.
                       'alex' is the default and most common. 'squeeze' is the lightest.
            pretrained (bool): Whether to load pretrained weights.
            pnet_tune (bool): Whether to finetune the backbone. Should be False.
        """
        super(LPIPSLoss, self).__init__()
        self.loss_weight = loss_weight
        self.loss_fn = lpips.LPIPS(net=net, pretrained=pretrained, pnet_tune=pnet_tune, verbose=False)

    def to(self, *args, **kwargs):
        """Override the to() method to move the lpips model to the correct device."""
        super().to(*args, **kwargs)
        self.loss_fn.to(*args, **kwargs)
        return self

    def forward(self, pred, target):
        """
        LPIPS expects input in range [-1, 1].
        Our data is in [0, 1]. We need to convert it.
        """
        # Convert from [0, 1] to [-1, 1]
        pred_norm = pred * 2 - 1
        target_norm = target * 2 - 1

        loss = self.loss_fn(pred_norm, target_norm).mean()
        return loss * self.loss_weight