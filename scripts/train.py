import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import datetime
import monai.transforms as mt

from monai.networks.nets import UNet
from src.data.dataset import StrokeDataset
from src.training.losses import CombinedLoss
from src.training.trainer import Trainer

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    # --- MAIN CONFIGURATION ---
    MODE = "2d"  # Available modes: '2d', '2.5d', '3d'
    BATCH_SIZE = 16 if MODE != "3d" else 1
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 40
    NUM_WORKERS = 8
    ACCUMULATION_STEPS = 4 if MODE == "3d" else 1

    # --- DEVICE CONFIGURATION ---
    torch.backends.cudnn.benchmark = MODE != "3d"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- TIMESTAMP & FILENAMES ---
    timestamp = datetime.datetime.now().strftime("%d_%m_%Y_%H_%M")
    best_model_filename = f"unet_{MODE}_{timestamp}.pth"

    # --- DATA PREPARATION ---
    data_dir = PROJECT_ROOT / "data"
    splits_path = PROJECT_ROOT / "configs" / "data_splits.json"

    # Load patient splits from JSON
    with open(splits_path, "r") as f:
        splits = json.load(f)

    train_patients = splits["train"]
    val_patients = splits["val"]

    # --- TRANSORMS CONFIGURATION ---
    train_transforms = mt.Compose(
        [
            mt.ScaleIntensityRanged(
                keys=["image"], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True
            ),
            mt.RandFlipd(keys=["image", "label"], spatial_axis=-1, prob=0.5),
            mt.RandRotated(
                keys=["image", "label"],
                prob=0.5,
                keep_size=True,
                mode=["bilinear", "nearest"],
                padding_mode=["zeros", "zeros"],
                range_x=0.26,
            ),
        ]
    )

    val_transforms = mt.Compose(
        [
            mt.ScaleIntensityRanged(
                keys=["image"], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True
            ),
        ]
    )

    # --- DATASETS INITIALIZATION ---
    train_dataset = StrokeDataset(
        str(data_dir), train_patients, mode=MODE, transform=train_transforms
    )
    val_dataset = StrokeDataset(
        str(data_dir), val_patients, mode=MODE, transform=val_transforms
    )

    # --- DATALOADERS INITIALIZATION ---
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    # --- MODEL INITIALIZATION ---
    if MODE == "2d":
        model = UNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            channels=(32, 64, 128, 256, 512),
            strides=(2, 2, 2, 2),
            num_res_units=0,
        ).to(device)
    elif MODE == "2.5d":
        model = UNet(
            spatial_dims=2,
            in_channels=3,
            out_channels=1,
            channels=(32, 64, 128, 256, 512),
            strides=(2, 2, 2, 2),
            num_res_units=0,
        ).to(device)
    elif MODE == "3d":
        model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            channels=(32, 64, 128, 256, 512),
            # Stride (1, 2, 2) ensures Z-axis (Depth) is not compressed,
            # maintaining full resolution along the patient axis.
            strides=((1, 2, 2), (1, 2, 2), (1, 2, 2), (1, 2, 2)),
            num_res_units=0,
        ).to(device)
    else:
        raise ValueError("Invalid mode!")

    # --- LOSS AND OPTIMIZER INITIALIZATION ---
    criterion = CombinedLoss()
    optimizer = optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

    # --- SCHEUDLER INITIALIZATION ---
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # --- EXPERIMENT CONFIGURATION (For Logging) ---
    experiment_config = {
        "MODE": MODE,
        "BATCH_SIZE": BATCH_SIZE,
        "LEARNING_RATE": LEARNING_RATE,
        "NUM_EPOCHS": NUM_EPOCHS,
        "NUM_WORKERS": NUM_WORKERS,
        "DEVICE": device,
        "AUTOMATIC MIXED PRECISION": True,
        "OPTIMIZER": optimizer.__class__.__name__,
        "ACCUMULATION_STEPS": ACCUMULATION_STEPS,
        "SCHEDULER": scheduler.__class__.__name__,
        "LOSS": criterion.__class__.__name__,
        "TRAIN_PATIENTS": len(train_patients),
        "VAL_PATIENTS": len(val_patients),
        "TIMESTAMP": timestamp,
    }

    # --- TRAINING EXECUTION ---
    print(f"\nStarting Training Engine: {datetime.datetime.now().time()}")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        device=device,
        best_model_filename=best_model_filename,
        num_epochs=NUM_EPOCHS,
        config=experiment_config,
        training_logs=False,
    )

    trainer.train()


if __name__ == "__main__":
    main()
