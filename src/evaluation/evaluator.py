import torch
import torch.nn as nn
from monai.metrics import DiceMetric
import torch.nn.functional as F

import numpy as np
from scipy.ndimage import label


class Evaluator:
    """
    Handles the evaluation of 2D, 2.5D, and 3D models.
    Ensures fair comparison by reconstructing 2D predictions into 3D volumes
    before calculating the final Volume Dice Score.
    """

    def __init__(self, model: nn.Module, device: str, mode: str):
        super().__init__()
        self.model = model
        self.device = device
        self.mode = mode
        self.dice_metric = DiceMetric(include_background=False, reduction="mean")

    def _predict_2d_volume(self, volume: torch.Tensor) -> torch.Tensor:
        """
        Takes a 3D volume, processes it slice-by-slice through a 2D model,
        and reconstructs the 3D prediction volume.
        """
        B, C, D, H, W = volume.shape
        predicted_slices = []

        for d in range(D):
            if self.mode == "2d":
                slice_input = volume[:, :, d, :, :]

            elif self.mode == "2.5d":
                prev_idx = max(0, d - 1)
                next_idx = min(D - 1, d + 1)

                slice_prev = volume[:, :, prev_idx, :, :]
                slice_curr = volume[:, :, d, :, :]
                slice_next = volume[:, :, next_idx, :, :]

                slice_input = torch.cat([slice_prev, slice_curr, slice_next], dim=1)

            slice_pred = self.model(slice_input)

            predicted_slices.append(slice_pred)

        reconstructed_volume = torch.stack(tensors=predicted_slices, dim=2)

        return reconstructed_volume

    def _predict_3d_volume(self, volume: torch.Tensor) -> torch.Tensor:
        """
        Takes a 3D volume and processes it directly through a 3D model.
        """
        B, C, D, H, W = volume.shape
        pad_d = (16 - (D % 16)) % 16
        if pad_d > 0:
            volume = F.pad(volume, (0, 0, 0, 0, 0, pad_d))

        pred_volume = self.model(volume)

        return pred_volume[:, :, :D, :, :]

    def _apply_cca_filtering(
        self, binary_mask: torch.Tensor, min_size: int = 100
    ) -> torch.Tensor:
        """
        Removes isolated noise components smaller than min_size from the binary mask.
        """
        mask_np = binary_mask.cpu().numpy().astype(bool)
        labeled_mask, num_features = label(mask_np)

        if num_features == 0:
            return binary_mask

        component_sizes = np.bincount(labeled_mask.ravel())

        valid_components = component_sizes >= min_size
        valid_components[0] = False

        filtered_mask_np = valid_components[labeled_mask]

        return torch.from_numpy(filtered_mask_np).float().to(self.device)

    def evaluate_patient(
        self, volume: torch.Tensor, ground_truth: torch.Tensor
    ) -> float:
        """
        Public method to evaluate a single patient's volume.
        """
        volume = volume.to(self.device)
        ground_truth = ground_truth.to(self.device)

        self.model.eval()

        with torch.no_grad():
            if self.mode in ["2d", "2.5d"]:
                predictions = self._predict_2d_volume(volume)
            elif self.mode == "3d":
                predictions = self._predict_3d_volume(volume)
            else:
                raise ValueError(f"Unknown mode: {self.mode}")

            # 1. Convert logits to probabilities using Sigmoid
            probs = torch.sigmoid(predictions)

            # 2. Threshold probabilities to create a binary mask (0.0 or 1.0)
            binary_mask = (probs > 0.5).float()
            binary_mask = self._apply_cca_filtering(binary_mask, min_size=100)

            # 3. Calculate Dice Score using MONAI metric
            self.dice_metric(y_pred=binary_mask, y=ground_truth)

            # 4. Extract the actual scalar value
            dice_score = self.dice_metric.aggregate().item()

            # 5. Reset metric for the next patient
            self.dice_metric.reset()

        return dice_score
