from monai.networks.nets import AttentionUnet

LAYERS = [(64, 128, 256, 512, 1024), (32, 64, 128, 256, 512)]

attention_unet = {
    "2d": AttentionUnet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        channels=LAYERS[0],
        strides=(2, 2, 2, 2),
    ),
    "2.5d": AttentionUnet(
        spatial_dims=2,
        in_channels=3,
        out_channels=1,
        channels=LAYERS[0],
        strides=(2, 2, 2, 2),
    ),
    "3d": AttentionUnet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=LAYERS[1],
        strides=((1, 2, 2), (1, 2, 2), (1, 2, 2), (1, 2, 2)),
    ),
}
