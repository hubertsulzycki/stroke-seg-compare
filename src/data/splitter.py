import json
from pathlib import Path
from sklearn.model_selection import train_test_split

def create_patient_splits(data_dir: str, val_size: float = 0.15, test_size: float = 0.15, random_state: int = 42):
    """
    Performs patient-level split to prevent data leakage.
    """
    images_dir = Path(data_dir) / "raw" / "images"
    
    # Get directory only names inside data/raw/images/  
    patients = [p.name for p in images_dir.iterdir() if p.is_dir()]
    patients = sorted(patients)
    
    if not patients:
        raise ValueError(f"No patient folders found in: {images_dir}")

    # Extract the test set from the entire pool
    train_val_patients, test_patients = train_test_split(
        patients, test_size=test_size, random_state=random_state
    )
    
    # Extract the validation set from the remaining pool
    # Adjust the validation ratio since the pool is already reduced by test_size
    val_ratio = val_size / (1.0 - test_size)
    train_patients, val_patients = train_test_split(
        train_val_patients, test_size=val_ratio, random_state=random_state
    )
    
    print("--- Patient Split Completed ---")
    print(f"Total patients : {len(patients)}")
    print(f"Train          : {len(train_patients)}")
    print(f"Validation     : {len(val_patients)}")
    print(f"Test           : {len(test_patients)}")
    
    return {
        "train": train_patients,
        "val": val_patients,
        "test": test_patients
    }

def save_splits(splits: dict, output_path: str):
    """Saves the split configuration to a JSON file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(splits, f, indent=4)
    print(f"Split configuration saved to: {output_path}")

if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    DATA_DIR = PROJECT_ROOT / "data" 
    OUTPUT_JSON = PROJECT_ROOT / "configs" / "data_splits.json"
    
    try:
        # 70% train / 15% val / 15% test
        splits = create_patient_splits(DATA_DIR, val_size=0.15, test_size=0.15)
        save_splits(splits, OUTPUT_JSON)
    except Exception as e:
        print(f"Critical error: {e}")