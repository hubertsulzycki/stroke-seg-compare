from monai.networks.nets import SegResNet

segresnet = {
    "2d": SegResNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        init_filters=32,
        blocks_down=[1, 2, 2, 4],
        blocks_up=[1, 1, 1],
    ),
    "2.5d": SegResNet(
        spatial_dims=2,
        in_channels=3,
        out_channels=1,
        init_filters=32,
        blocks_down=[1, 2, 2, 4],
        blocks_up=[1, 1, 1],
    ),
    "3d": SegResNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        init_filters=32,
        blocks_down=[1, 2, 2, 4],
        blocks_up=[1, 1, 1],
    ),
}
