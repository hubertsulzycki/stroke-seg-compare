import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import datetime

from src.data.dataset import StrokeDataset
from src.models.unet_2d import UNet2D
from src.models.unet_3d import UNet3D
from src.training.losses import CombinedLoss
from src.training.trainer import Trainer

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    # --- MAIN CONFIGURATION ---
    MODE = "2d"  # Availbe mods: '2d', '2.5d', '3d'
    BATCH_SIZE = (
        8 if MODE != "3d" else 1
    )  # 3D needs batch size 1 due to VRAM limitations
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 5
    NUM_WORKERS = 0

    # --- DEVICE CONFIGURATION ---
    torch.backends.cudnn.benchmark = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device.upper()} for {MODE.upper()} training.")

    # --- DATA PREPARATION ---
    data_dir = PROJECT_ROOT / "data"
    splits_path = PROJECT_ROOT / "configs" / "data_splits.json"

    # Load patient splits from JSON
    with open(splits_path, "r") as f:
        splits = json.load(f)

    train_patients = splits["train"]
    val_patients = splits["val"]

    print(
        f"Loaded {len(train_patients)} training patients and {len(val_patients)} validation patients."
    )

    # --- DATASETS INITIALIZATION ---
    train_dataset = StrokeDataset(str(data_dir), train_patients, mode=MODE)
    val_dataset = StrokeDataset(str(data_dir), val_patients, mode=MODE)

    # --- DATALOADERS INITIALIZATION ---
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # --- MODEL INITIALIZATION ---
    if MODE == "2d":
        model = UNet2D(in_channels=1).to(device)
        best_model_filename = "unet_2d.pth"
    elif MODE == "2.5d":
        model = UNet2D(in_channels=3).to(device)
        best_model_filename = "unet_2.5d.pth"
    elif MODE == "3d":
        model = UNet3D(num_res_units=0).to(device)
        best_model_filename = "unet_3d.pth"
    else:
        raise ValueError("Invalid mode!")

    model = torch.compile(model)

    # --- LOSS AND OPTIMIZER INITIALIZATION ---
    criterion = CombinedLoss()
    optimizer = optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

    # --- TRAINING EXECUTION ---
    print(f"\nStarting Training Engine: {datetime.datetime.now().time()}")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        best_model_filename=best_model_filename,
        num_epochs=NUM_EPOCHS,
        training_logs=True,
    )

    trainer.train()
    print("Training Completed Successfully!")


if __name__ == "__main__":
    main()
