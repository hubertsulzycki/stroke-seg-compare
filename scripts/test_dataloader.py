import sys
from pathlib import Path
from src.data.dataset import StrokeDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def run_tests():
    data_dir = PROJECT_ROOT / "data"
    images_dir = data_dir / "raw" / "images"
    
    patients = sorted([p.name for p in images_dir.iterdir() if p.is_dir()])
    
    if not patients:
        print("ERROR: No patients found in data/raw/images/.")
        print("Please add at least one patient folder with some PNG files to run the test.")
        return

    # Get a maximum of two patients for the test    
    test_patients = patients[:2]
    print(f"Found patients for testing: {test_patients}")
    
    # List of paradigms to test
    modes = ['2d', '2.5d', '3d']
    
    for mode in modes:
        print(f"\n{'='*50}")
        print(f"--- TESTING MODE: {mode.upper()} ---")
        print(f"{'='*50}")
        
        try:
            # Initialize the dataset
            dataset = StrokeDataset(
                data_dir=str(data_dir), 
                patient_list=test_patients, 
                mode=mode
            )
            
            # Fetch the first sample (index 0)
            sample = dataset[0]
            
            img_tensor = sample['image']
            mask_tensor = sample['label']
            meta_data = sample['meta']
            
            print(f"-> Image Tensor Shape : {img_tensor.shape}")
            print(f"-> Label Tensor Shape : {mask_tensor.shape}")
            print(f"-> Metadata (Meta)    : {meta_data}")
            print(f"-> Image Data Type    : {img_tensor.dtype}")
            print(f"-> Unique Label Values: {mask_tensor.unique().tolist()}")
            
        except Exception as e:
            print(f"!!! ERROR OCCURRED IN {mode.upper()} MODE: {e}")

if __name__ == "__main__":
    run_tests()