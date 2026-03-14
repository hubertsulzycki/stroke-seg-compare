import torch
import torch.nn as nn
from monai.networks.nets import UNet


class UNet3D(nn.Module):
    """
    A 3D U-Net wrapper using the MONAI library.
    Allows for easy switching between a standard U-Net and a Residual U-Net
    by adjusting the 'num_res_units' parameter.
    """

    def __init__(self, num_res_units: int = 0):
        super().__init__()

        self.model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=num_res_units,
        )

    def forward(self, x: torch.Tensor):
        return self.model(x)
