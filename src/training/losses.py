import torch
import torch.nn as nn
from monai.losses import DiceLoss


class CombinedLoss(nn.Module):
    """
    This module combines two distinct loss functions to optimize training
    in scenarios with severe class imbalance:

    1. BCEWithLogitsLoss: Evaluates pixel-wise classification error.
    2. DiceLoss: Evaluates the spatial overlap between the predicted mask and the ground truth.
    """

    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(sigmoid=True, squared_pred=True)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)

        return bce_loss + dice_loss
