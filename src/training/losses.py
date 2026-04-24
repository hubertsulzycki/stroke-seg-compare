import torch
import torch.nn as nn
from monai.losses import TverskyLoss


class CombinedLoss(nn.Module):
    """
    This module combines two distinct loss functions to optimize training
    in scenarios with severe class imbalance:

    1. BCEWithLogitsLoss: Evaluates pixel-wise classification error.
    2. DiceLoss: Evaluates the spatial overlap between the predicted mask and the ground truth - not used anymore
    3. TverskyLoss: By setting alpha=0.3 (False Positive weight) and beta=0.7 (False Negative weight), it heavily penalizes missed stroke lesions, encouraging the model to make bolder predictions on underrepresented classes.
    """

    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.tversky = TverskyLoss(sigmoid=True, alpha=0.3, beta=0.7)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)
        tversky_loss = self.tversky(logits, targets)
        return bce_loss + tversky_loss
