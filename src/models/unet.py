from monai.networks.nets import UNet

LAYERS = [(64, 128, 256, 512, 1024), (32, 64, 128, 256, 512)]
unet = {
    "2d": UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        channels=LAYERS[0],
        strides=(2, 2, 2, 2),
        num_res_units=0,
    ),
    "2dr": UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        channels=LAYERS[0],
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ),
    "2.5d": UNet(
        spatial_dims=2,
        in_channels=3,
        out_channels=1,
        channels=LAYERS[0],
        strides=(2, 2, 2, 2),
        num_res_units=0,
    ),
    "2.5dr": UNet(
        spatial_dims=2,
        in_channels=3,
        out_channels=1,
        channels=LAYERS[0],
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ),
    "3d": UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=LAYERS[1],
        # Stride (1, 2, 2) ensures Z-axis (Depth) is not compressed,
        # maintaining full resolution along the patient axis.
        strides=((1, 2, 2), (1, 2, 2), (1, 2, 2), (1, 2, 2)),
        num_res_units=0,
    ),
    "3dr": UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=LAYERS[1],
        # Stride (1, 2, 2) ensures Z-axis (Depth) is not compressed,
        # maintaining full resolution along the patient axis.
        strides=((1, 2, 2), (1, 2, 2), (1, 2, 2), (1, 2, 2)),
        num_res_units=2,
    ),
}
