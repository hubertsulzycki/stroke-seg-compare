import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from pathlib import Path

class StrokeDataset(Dataset):
    """
    Universal PyTorch Dataset for 2D, 2.5D, and 3D stroke segmentation.
    Returns a dictionary: {"image": torch.Tensor, "label": torch.Tensor}
    """
    def __init__(self, data_dir: str, patient_list: list, mode: str = '2d', transform=None):
        self.data_dir = Path(data_dir)
        self.images_dir = self.data_dir / "raw" / "images"
        self.masks_dir = self.data_dir / "raw" / "masks"
        
        self.patient_list = patient_list
        self.mode = mode.lower()
        self.transform = transform
        
        if self.mode not in ['2d', '2.5d', '3d']:
            raise ValueError(f"Unsupported mode: {self.mode}. Choose from '2d', '2.5d', '3d'.")
            
        self.samples = self._prepare_samples()
        print(f"Dataset initialized in {self.mode.upper()} mode. Total samples: {len(self)}")

    def _prepare_samples(self) -> list:
        """
        Populates self.samples based on the selected mode (2D, 2.5D, or 3D).
        """
        samples = []
        
        for patient_id in self.patient_list:
            patient_img_dir = self.images_dir / patient_id
            patient_mask_dir = self.masks_dir / patient_id
            
            # Get all PNG files and sort them to ensure correct Z-axis order (bottom to top)
            slices = sorted([f.name for f in patient_img_dir.iterdir() if f.suffix == '.png'])
            
            if self.mode == '2d':
                # In 2D, every single slice is a distinct sample
                for slice_name in slices:
                    samples.append({
                        'img_path': patient_img_dir / slice_name,
                        'mask_path': patient_mask_dir / slice_name,
                        'patient_id': patient_id,
                        'slice_name': slice_name
                    })
            
            elif self.mode == '2.5d':
                # In 2.5D, each sample consist of z-1, z and z+1 slices
                # Usage of max() and min() handles border cases
                for i, slice_name in enumerate(slices):
                    prev_idx = max(0, i - 1)
                    next_idx = min(len(slices) - 1, i + 1)
                    samples.append({
                        'img_path_prev': patient_img_dir / slices[prev_idx],
                        'img_path_curr': patient_img_dir / slice_name,
                        'img_path_next': patient_img_dir / slices[next_idx],
                        'mask_path': patient_mask_dir / slice_name,
                        'patient_id': patient_id,
                        'slice_name': slice_name
                    })

            elif self.mode == '3d':
                # In 3D, all slices are used as one sample
                samples.append({
                        'patient_img_path': patient_img_dir,
                        'patient_mask_path': patient_mask_dir,
                        'slices': slices,                        
                        'patient_id': patient_id,
                    })
                
        return samples

    def _map_mask(self, mask_array: np.ndarray) -> np.ndarray:
        """
        CRITICAL: Maps mask values according to the dataset rule:
        Target (1): 1 (remote), 2 (clear acute), 3 (blurred acute), 5 (infarct)
        Background (0): 0 (background), 4 (invisible)
        """
        mapped_mask = np.zeros_like(mask_array, dtype=np.float32)
        target_mask = np.isin(mask_array, [1, 2, 3, 5])
        mapped_mask[target_mask] = 1.0
        
        return mapped_mask

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample_info = self.samples[idx]
        
        if self.mode == '2d':
            # Load image and mask
            img = np.array(Image.open(sample_info['img_path']).convert('L'), dtype=np.float32) / 255.0
            mask = np.array(Image.open(sample_info['mask_path']).convert('L'), dtype=np.uint8)
            
            mask = self._map_mask(mask)
            
            # Add channel dimension (1, H, W)
            img = np.expand_dims(img, axis=0)
            mask = np.expand_dims(mask, axis=0)
            
            # Convert to PyTorch tensors
            item = {
                "image": torch.from_numpy(img),
                "label": torch.from_numpy(mask),
                # Storing metadata for 2D->3D reconstruction later
                "meta": {
                    "patient_id": sample_info['patient_id'],
                    "slice_name": sample_info['slice_name']
                }
            }
            
            # Apply MONAI transforms if any
            if self.transform:
                item = self.transform(item)
                
            return item
        
        elif self.mode == '2.5d':
            # Load 3 adjacent slices and mask
            img_prev = np.array(Image.open(sample_info['img_path_prev']).convert('L'), dtype=np.float32) / 255.0
            img_curr = np.array(Image.open(sample_info['img_path_curr']).convert('L'), dtype=np.float32) / 255.0
            img_next = np.array(Image.open(sample_info['img_path_next']).convert('L'), dtype=np.float32) / 255.0
            mask = np.array(Image.open(sample_info['mask_path']).convert('L'), dtype=np.uint8)

            mask = self._map_mask(mask)

            # Stack images along a first axis (axis=0) to create channels: (3, H, W) and channel dimension for mask to get (1, H, W)
            img = np.stack((img_prev, img_curr, img_next), axis=0)
            mask = np.expand_dims(mask, axis=0)
            
            # Convert to PyTorch tensors
            item = {                
                "image": torch.from_numpy(img),
                "label": torch.from_numpy(mask),
                # Storing metadata for 2D->3D reconstruction later
                "meta": {
                    "patient_id": sample_info['patient_id'],
                    "slice_name": sample_info['slice_name']
                }
            }

            # Apply MONAI transforms if any
            if self.transform:
                item = self.transform(item)
                
            return item            

        elif self.mode == '3d':
            patient_img_path = sample_info['patient_img_path']
            patient_mask_path = sample_info['patient_mask_path']

            # Initialize list to hold 2D arrays before stacking
            img_volume = []
            mask_volume = []

            # Load and process each slice
            for slice_name in sample_info['slices']:
                img_path = patient_img_path / slice_name
                mask_path = patient_mask_path / slice_name

                img = np.array(Image.open(img_path).convert('L'), dtype=np.float32) / 255.0
                mask = np.array(Image.open(mask_path).convert('L'), dtype=np.uint8)
                mask = self._map_mask(mask)

                img_volume.append(img)
                mask_volume.append(mask)

            # Stack 2D arrays along axis 0 to create the Depth dimension (D, H, W)
            img = np.stack(img_volume, axis=0)
            mask = np.stack(mask_volume, axis=0)

            # Add channel dimension (D, H, W) -> (1, D, H, W)
            img = np.expand_dims(img, axis=0)
            mask = np.expand_dims(mask, axis=0)

            # Convert to PyTorch tensors
            item = {                
                "image": torch.from_numpy(img),
                "label": torch.from_numpy(mask),
                # Storing metadata for 2D->3D reconstruction later
                "meta": {
                    "patient_id": sample_info['patient_id']
                }
            }

            # Apply MONAI transforms if any
            if self.transform:
                item = self.transform(item)
                
            return item 
