import torch
from src.models.unet_3d import UNet3D


def run_tests():
    # Test for 3D Paradigm (1 input channel)
    model_3d = UNet3D()

    # Create a dummy tensor representing a batch of 1 sample consists of 16 slices of size 512x512
    # Shape (Batch, Channels, Depth, Height, Width) -> (1, 1, 16, 512, 512)
    dummy_input_3d = torch.randn(1, 1, 16, 512, 512)

    # Pass the tensor through the model
    output_3d = model_3d(dummy_input_3d)

    print("--- 3D Model Test ---")
    print(f"Input shape : {dummy_input_3d.shape}")
    print(f"Output shape: {output_3d.shape}")

    # The expected output shape is (1, 1, 16, 512, 512)
    assert output_3d.shape == (1, 1, 16, 512, 512), "Output shape mismatch!"
    print("3D Test Passed Successfully!\n")


if __name__ == "__main__":
    run_tests()
