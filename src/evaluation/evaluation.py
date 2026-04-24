import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import monai.transforms as mt

from src.models.unet import unet
from src.data.dataset import StrokeDataset
from src.evaluation.evaluator import Evaluator


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def log_message(message: str, filepath: Path):
    """Prints a message to the console and appends it to the log file."""
    print(message)
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(message + "\n")


def main():
    print(PROJECT_ROOT)
    # --- MAIN CONFIGURATION ---
    MODE = "2dr"
    MODEL_FILENAME = "unet_2dr_20_04_2026_21_09.pth"
    BATCH_SIZE = 1
    NUM_WORKERS = 8

    # --- DEVICE CONFIGURATION ---
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- PATHS CONFIGURATION ---
    data_dir = PROJECT_ROOT / "data"
    model_dir = PROJECT_ROOT / "trained_models"
    splits_path = PROJECT_ROOT / "configs" / "data_splits.json"

    # --- LOGGING SETUP ---
    log_filename = MODEL_FILENAME.replace(".pth", "_eval.txt")
    log_filepath = model_dir / log_filename

    # Load patient splits from JSON
    with open(splits_path, "r") as f:
        splits = json.load(f)

    test_patients = splits["test"]

    # --- DATASETS INITIALIZATION ---
    test_transforms = mt.Compose(
        [
            mt.ScaleIntensityRanged(
                keys=["image"], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True
            ),
        ]
    )
    test_dataset = StrokeDataset(
        str(data_dir), test_patients, mode="3d", transform=test_transforms
    )  # 3D is always required to ensure fair comparission between models

    # --- DATALOADERS INITIALIZATION ---
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # --- MODEL INITIALIZATION ---
    try:
        model = unet[MODE].to(device)
    except:
        raise ValueError("Invalid mode!")

    model.load_state_dict(torch.load(model_dir / MODEL_FILENAME, weights_only=True))

    # --- EVALUATION ENGINE ---
    evaluator = Evaluator(model=model, device=device, mode=MODE)

    log_message("=" * 50, log_filepath)
    log_message("--- STARTING EVALUATION ---", log_filepath)
    log_message(f"Model File    : {MODEL_FILENAME}", log_filepath)
    log_message(f"Mode          : {MODE.upper()}", log_filepath)
    log_message(f"Test Patients : {len(test_patients)}", log_filepath)
    log_message("=" * 50 + "\n", log_filepath)

    all_dice_scores = []

    for i, batch in enumerate(test_loader):
        volume = batch["image"]
        ground_truth = batch["label"]

        patient_dice = evaluator.evaluate_patient(
            volume=volume, ground_truth=ground_truth
        )
        all_dice_scores.append(patient_dice)

        log_message(
            f"Patient {i+1:03d}/{len(test_loader):03d} - Volume Dice: {patient_dice:.4f}",
            log_filepath,
        )

    mean_dice = sum(all_dice_scores) / len(all_dice_scores)

    log_message("\n" + "=" * 50, log_filepath)
    log_message(f"FINAL MODEL SCORE ({MODE.upper()}):", log_filepath)
    log_message(f"Mean Volume Dice Score on Test Set: {mean_dice:.4f}", log_filepath)
    log_message("=" * 50, log_filepath)


if __name__ == "__main__":
    main()
