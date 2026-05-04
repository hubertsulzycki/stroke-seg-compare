from monai.networks.nets import VNet

# VNet doesn't support 2.5 natively
vnet = {
    "2d": VNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
    ),
    "3d": VNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
    ),
}
