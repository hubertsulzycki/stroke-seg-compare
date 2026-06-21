import torch
import torch.nn as nn
from monai.metrics import DiceMetric, HausdorffDistanceMetric, ConfusionMatrixMetric
import torch.nn.functional as F

import numpy as np
from scipy.ndimage import label


class Evaluator:
    """
    Handles the evaluation of 2D, 2.5D, and 3D models.
    Ensures fair comparison by reconstructing 2D predictions into 3D volumes
    before calculating the final Volume Dice Score.
    """

    def __init__(self, architecture: str, model: nn.Module, device: str, mode: str):
        super().__init__()
        self.architecture = architecture
        self.model = model
        self.device = device
        self.mode = mode

        # --- METRICS INITIALIZATION ---
        self.dice_metric = DiceMetric(include_background=False, reduction="mean")
        self.hd95_metric = HausdorffDistanceMetric(
            percentile=95, include_background=False, reduction="mean"
        )
        self.conf_matrix = ConfusionMatrixMetric(
            metric_name=["sensitivity", "precision"],
            include_background=False,
            reduction="mean",
        )

    def _predict_2d_volume(self, volume: torch.Tensor) -> torch.Tensor:
        """
        Takes a 3D volume, processes it slice-by-slice through a 2D model,
        and reconstructs the 3D prediction volume.
        """
        B, C, D, H, W = volume.shape
        predicted_slices = []

        for d in range(D):
            if self.mode in ["2d", "2dr"]:
                slice_input = volume[:, :, d, :, :]

            elif self.mode in ["2.5d", "2.5dr"]:
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

        k = 1
        if self.architecture == "swin_unetr":
            k = 32
        elif self.architecture in ["segresnet", "vnet"]:
            k = 16

        pad_d = (k - (D % k)) % k

        if pad_d > 0:
            volume = F.pad(volume, (0, 0, 0, 0, 0, pad_d))

        pred_volume = self.model(volume)

        if pad_d > 0:
            pred_volume = pred_volume[:, :, :D, :, :]

        return pred_volume

    def _apply_cca_filtering(
        self, binary_mask: torch.Tensor, min_size: int = 100
    ) -> torch.Tensor:
        """
        Removes isolated noise components smaller than min_size from the binary mask.
        Operates on spatial dimensions only to ensure correct 3D connectivity.
        """
        original_shape = binary_mask.shape

        # Squeeze batch and channel dimensions to get a pure (D, H, W) spatial array
        mask_np = binary_mask.squeeze().cpu().numpy().astype(bool)
        labeled_mask, num_features = label(mask_np)

        if num_features == 0:
            return binary_mask

        component_sizes = np.bincount(labeled_mask.ravel())

        valid_components = component_sizes >= min_size
        valid_components[0] = False

        filtered_mask_np = valid_components[labeled_mask]

        # Convert back to tensor and restore the original (B, C, D, H, W) shape
        filtered_tensor = torch.from_numpy(filtered_mask_np).float().to(self.device)
        filtered_tensor = filtered_tensor.view(original_shape)

        return filtered_tensor

    def evaluate_patient(
        self, volume: torch.Tensor, ground_truth: torch.Tensor
    ) -> float:
        """
        Public method to evaluate a single patient's volume, using Test-Time Augmentation.
        The final probalities are average values from prediction on base and mirrored slice.
        """
        volume = volume.to(self.device)
        ground_truth = ground_truth.to(self.device)

        mirror_volume = torch.flip(volume, dims=[-1])

        self.model.eval()

        with torch.no_grad():
            if self.mode in ["2d", "2dr", "2.5d", "2.5dr"]:
                predictions = self._predict_2d_volume(volume)
                mirror_predictions = self._predict_2d_volume(mirror_volume)
            elif self.mode in ["3d", "3dr"]:
                predictions = self._predict_3d_volume(volume)
                mirror_predictions = self._predict_3d_volume(mirror_volume)
            else:
                raise ValueError(f"Unknown mode: {self.mode}")

            # 1. Convert logits to probabilities using Sigmoid
            normal_probs = torch.sigmoid(predictions)
            mirror_probs = torch.sigmoid(torch.flip(mirror_predictions, dims=[-1]))
            final_probs = (normal_probs + mirror_probs) / 2

            # 2. Threshold probabilities to create a binary mask (0.0 or 1.0)
            binary_mask = (final_probs > 0.5).float()
            binary_mask = self._apply_cca_filtering(binary_mask, min_size=100)

            # 3. Calculate Metrics using MONAI
            self.dice_metric(y_pred=binary_mask, y=ground_truth)
            self.hd95_metric(y_pred=binary_mask, y=ground_truth)
            self.conf_matrix(y_pred=binary_mask, y=ground_truth)

            # 4. Extract the actual scalar values
            dice_score = self.dice_metric.aggregate().item()

            # HD95 can return 'inf' if the predicted mask is completely empty.
            hd95_tensor = self.hd95_metric.aggregate()
            hd95_score = (
                hd95_tensor.item() if not torch.isinf(hd95_tensor) else float("nan")
            )

            # Confusion Matrix returns a list of tensors when multiple metrics are requested
            conf_metrics = self.conf_matrix.aggregate()
            sensitivity_score = conf_metrics[0].item()
            precision_score = conf_metrics[1].item()

            # 5. Reset metrics for the next patient
            self.dice_metric.reset()
            self.hd95_metric.reset()
            self.conf_matrix.reset()

        return {
            "dice": dice_score,
            "hd95": hd95_score,
            "sensitivity": sensitivity_score,
            "precision": precision_score,
        }

        return dice_score
