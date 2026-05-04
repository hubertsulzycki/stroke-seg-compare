from monai.networks.nets import SwinUNETR

swin_unetr = {
    "2d": SwinUNETR(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        feature_size=24,
    ),
    "2.5d": SwinUNETR(
        spatial_dims=2,
        in_channels=3,
        out_channels=1,
        feature_size=24,
    ),
    "3d": SwinUNETR(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        feature_size=24,
    ),
}
