import torch
from src.models.unet_2d import UNet2D


def run_tests():
    # Test for 2D Paradigm (1 input channel)
    model_2d = UNet2D(in_channels=1)

    # Create a dummy tensor representing a batch of 2 images of size 512x512
    # Shape: (Batch_size, Channels, Height, Width) -> (2, 1, 512, 512)
    dummy_input_2d = torch.randn(2, 1, 512, 512)

    # Pass the tensor through the model
    output_2d = model_2d(dummy_input_2d)

    print("--- 2D Model Test ---")
    print(f"Input shape : {dummy_input_2d.shape}")
    print(f"Output shape: {output_2d.shape}")

    # The expected output shape is (2, 1, 512, 512)
    assert output_2d.shape == (2, 1, 512, 512), "Output shape mismatch in 2D!"
    print("2D Test Passed Successfully!\n")

    # Test for 2.5D Paradigm (3 input channels)
    model_25d = UNet2D(in_channels=3)
    dummy_input_25d = torch.randn(2, 3, 512, 512)
    output_25d = model_25d(dummy_input_25d)

    print("--- 2.5D Model Test ---")
    print(f"Input shape : {dummy_input_25d.shape}")
    print(f"Output shape: {output_25d.shape}")

    # The expected output shape is still (2, 1, 512, 512) because we predict 1 mask
    assert output_25d.shape == (2, 1, 512, 512), "Output shape mismatch in 2.5D!"
    print("2.5D Test Passed Successfully!")


if __name__ == "__main__":
    run_tests()
