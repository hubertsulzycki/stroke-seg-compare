import json
import math
import torch
import monai.transforms as mt

from torch.utils.data import DataLoader
from pathlib import Path
from src.models.unet import unet
from src.models.attention_unet import attention_unet
from src.models.segresnet import segresnet
from src.models.vnet import vnet
from src.models.swin_unetr import swin_unetr
from src.data.dataset import StrokeDataset
from src.evaluation.evaluator import Evaluator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "unet": unet,
    "attention_unet": attention_unet,
    "segresnet": segresnet,
    "vnet": vnet,
    "swin_unetr": swin_unetr,
}


def log_message(message: str, filepath: Path):
    """Prints a message to the console and appends it to the log file."""
    print(message)
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(message + "\n")


def main():
    print(PROJECT_ROOT)
    # --- MAIN CONFIGURATION ---
    ARCHITECTURE = "vnet"
    MODE = "3d"
    MODEL_FILENAME = "vnet_3d_14_06_2026_17_10.pth"
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
        model = MODELS[ARCHITECTURE][MODE].to(device)
    except:
        raise ValueError("Invalid mode!")

    model.load_state_dict(torch.load(model_dir / MODEL_FILENAME, weights_only=True))

    # --- EVALUATION ENGINE ---
    evaluator = Evaluator(
        architecture=ARCHITECTURE, model=model, device=device, mode=MODE
    )

    log_message("=" * 50, log_filepath)
    log_message("--- STARTING EVALUATION ---", log_filepath)
    log_message(f"Architecture  : {ARCHITECTURE}", log_filepath)
    log_message(f"Model File    : {MODEL_FILENAME}", log_filepath)
    log_message(f"Mode          : {MODE.upper()}", log_filepath)
    log_message(f"Test Patients : {len(test_patients)}", log_filepath)
    log_message("=" * 50 + "\n", log_filepath)

    all_metrics = {"dice": [], "hd95": [], "sensitivity": [], "precision": []}

    for i, batch in enumerate(test_loader):
        volume = batch["image"]
        ground_truth = batch["label"]

        patient_metrics = evaluator.evaluate_patient(
            volume=volume, ground_truth=ground_truth
        )

        all_metrics["dice"].append(patient_metrics["dice"])
        all_metrics["hd95"].append(patient_metrics["hd95"])
        all_metrics["sensitivity"].append(patient_metrics["sensitivity"])
        all_metrics["precision"].append(patient_metrics["precision"])

        log_message(
            f"Patient {i+1:03d}/{len(test_loader):03d} - "
            f"Dice: {patient_metrics['dice']:.4f} | "
            f"HD95: {patient_metrics['hd95']:.4f} | "
            f"Sens: {patient_metrics['sensitivity']:.4f} | "
            f"Prec: {patient_metrics['precision']:.4f}",
            log_filepath,
        )

    mean_dice = sum(all_metrics["dice"]) / len(all_metrics["dice"])

    valid_hd95 = [x for x in all_metrics["hd95"] if not math.isnan(x)]
    mean_hd95 = sum(valid_hd95) / len(valid_hd95) if valid_hd95 else float("nan")

    mean_sensitivity = sum(all_metrics["sensitivity"]) / len(all_metrics["sensitivity"])
    mean_precision = sum(all_metrics["precision"]) / len(all_metrics["precision"])

    log_message("\n" + "=" * 50, log_filepath)
    log_message(f"FINAL MODEL SCORES ({MODE.upper()}):", log_filepath)
    log_message(f"Mean Volume Dice   : {mean_dice:.4f}", log_filepath)
    log_message(f"Mean HD95          : {mean_hd95:.4f}", log_filepath)
    log_message(f"Mean Sensitivity   : {mean_sensitivity:.4f}", log_filepath)
    log_message(f"Mean Precision     : {mean_precision:.4f}", log_filepath)
    log_message("=" * 50, log_filepath)


if __name__ == "__main__":
    main()
