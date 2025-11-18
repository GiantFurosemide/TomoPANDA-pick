"""
Data loader for cryoET 3D particle picking
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Union
import mrcfile
import h5py
from pathlib import Path
import json
from loguru import logger


class CryoETDataset(Dataset):
    """
    Dataset class for cryoET 3D particle picking
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        transform: Optional[callable] = None,
        target_transform: Optional[callable] = None,
        patch_size: Tuple[int, int, int] = (64, 64, 64),
        overlap: float = 0.5
    ):
        """
        Initialize the dataset
        
        Args:
            data_dir: Path to the data directory
            split: Dataset split ('train', 'val', 'test')
            transform: Transform to apply to input data
            target_transform: Transform to apply to target data
            patch_size: Size of patches to extract
            overlap: Overlap ratio between patches
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.patch_size = patch_size
        self.overlap = overlap
        
        # Load data paths
        self.data_paths = self._load_data_paths()
        logger.info(f"Loaded {len(self.data_paths)} samples for {split} split")
    
    def _load_data_paths(self) -> List[Dict[str, str]]:
        """Load paths to data files"""
        data_paths = []
        split_dir = self.data_dir / "processed" / self.split
        
        if not split_dir.exists():
            logger.warning(f"Split directory {split_dir} does not exist")
            return data_paths
        
        # Look for tomogram files
        for ext in ['.mrc', '.em', '.h5', '.hdf5']:
            for tomogram_path in split_dir.glob(f"*{ext}"):
                annotation_path = tomogram_path.with_suffix('.json')
                if annotation_path.exists():
                    data_paths.append({
                        'tomogram': str(tomogram_path),
                        'annotation': str(annotation_path)
                    })
        
        return data_paths
    
    def _load_tomogram(self, path: str) -> np.ndarray:
        """Load tomogram from file"""
        path = Path(path)
        
        if path.suffix == '.mrc':
            with mrcfile.open(path, mode='r') as mrc:
                return mrc.data.astype(np.float32)
        elif path.suffix == '.em':
            # EM file loading (simplified)
            with open(path, 'rb') as f:
                # This is a simplified version - real EM loading would be more complex
                data = np.frombuffer(f.read(), dtype=np.float32)
                # Reshape based on header info (simplified)
                return data.reshape((64, 64, 64))  # Placeholder
        elif path.suffix in ['.h5', '.hdf5']:
            with h5py.File(path, 'r') as f:
                return f['data'][:].astype(np.float32)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
    
    def _load_annotations(self, path: str) -> Dict:
        """Load particle annotations"""
        with open(path, 'r') as f:
            return json.load(f)
    
    def _extract_patches(self, tomogram: np.ndarray, annotations: Dict) -> List[Dict]:
        """Extract patches from tomogram"""
        patches = []
        h, w, d = tomogram.shape
        ph, pw, pd = self.patch_size
        
        # Calculate step size based on overlap
        step_h = int(ph * (1 - self.overlap))
        step_w = int(pw * (1 - self.overlap))
        step_d = int(pd * (1 - self.overlap))
        
        for z in range(0, d - pd + 1, step_d):
            for y in range(0, h - ph + 1, step_h):
                for x in range(0, w - pw + 1, step_w):
                    # Extract patch
                    patch = tomogram[y:y+ph, x:x+pw, z:z+pd]
                    
                    # Create binary mask for particles in this patch
                    mask = np.zeros_like(patch)
                    
                    # Check if any particles are in this patch
                    for particle in annotations.get('particles', []):
                        px, py, pz = particle['center']
                        if (x <= px < x + pw and 
                            y <= py < y + ph and 
                            z <= pz < z + pd):
                            # Mark particle location in patch coordinates
                            local_x = int(px - x)
                            local_y = int(py - y)
                            local_z = int(pz - z)
                            
                            # Create particle mask (simplified as sphere)
                            radius = particle.get('radius', 5)
                            for dz in range(-radius, radius + 1):
                                for dy in range(-radius, radius + 1):
                                    for dx in range(-radius, radius + 1):
                                        if (dx*dx + dy*dy + dz*dz <= radius*radius):
                                            nz, ny, nx = local_z + dz, local_y + dy, local_x + dx
                                            if (0 <= nx < pw and 0 <= ny < ph and 0 <= nz < pd):
                                                mask[ny, nx, nz] = 1
                    
                    patches.append({
                        'image': patch,
                        'mask': mask,
                        'bbox': (x, y, z, x + pw, y + ph, z + pd)
                    })
        
        return patches
    
    def __len__(self) -> int:
        """Return the number of patches"""
        total_patches = 0
        for data_path in self.data_paths:
            tomogram = self._load_tomogram(data_path['tomogram'])
            annotations = self._load_annotations(data_path['annotation'])
            patches = self._extract_patches(tomogram, annotations)
            total_patches += len(patches)
        return total_patches
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sample from the dataset"""
        current_idx = 0
        
        for data_path in self.data_paths:
            tomogram = self._load_tomogram(data_path['tomogram'])
            annotations = self._load_annotations(data_path['annotation'])
            patches = self._extract_patches(tomogram, annotations)
            
            if current_idx + len(patches) > idx:
                patch_idx = idx - current_idx
                patch = patches[patch_idx]
                
                # Convert to tensors
                image = torch.from_numpy(patch['image']).unsqueeze(0)  # Add channel dimension
                mask = torch.from_numpy(patch['mask']).unsqueeze(0)
                
                # Apply transforms
                if self.transform:
                    image = self.transform(image)
                if self.target_transform:
                    mask = self.target_transform(mask)
                
                return {
                    'image': image,
                    'mask': mask,
                    'bbox': patch['bbox']
                }
            
            current_idx += len(patches)
        
        raise IndexError(f"Index {idx} out of range")


class SubtomogramDataset(Dataset):
    """
    Dataset class for pre-extracted subtomogram particles
    用于已提取的subtomogram颗粒数据
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        transform: Optional[callable] = None,
        target_transform: Optional[callable] = None,
        create_mask: bool = True,
        mask_type: str = "full",  # "full" or "center"
        mask_radius: Optional[float] = None  # Radius for center mask (None = auto, or specify value)
    ):
        """
        Initialize the subtomogram dataset
        
        Args:
            data_dir: Path to the data directory
            split: Dataset split ('train', 'val', 'test')
            transform: Transform to apply to input data
            target_transform: Transform to apply to target data
            create_mask: Whether to create a mask (for segmentation tasks)
            mask_type: Type of mask - "full" (entire volume) or "center" (center region)
            mask_radius: Radius for center mask. If None, uses min(shape) // 4. 
                        Can be absolute value (e.g., 10) or relative (0.0-1.0, e.g., 0.25 = 25% of min dimension)
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.create_mask = create_mask
        self.mask_type = mask_type
        self.mask_radius = mask_radius
        
        # Load data paths
        self.data_paths = self._load_data_paths()
        logger.info(f"Loaded {len(self.data_paths)} subtomogram particles for {split} split")
    
    def _load_data_paths(self) -> List[str]:
        """Load paths to subtomogram files"""
        data_paths = []
        split_dir = self.data_dir / "processed" / self.split
        
        if not split_dir.exists():
            logger.warning(f"Split directory {split_dir} does not exist")
            return data_paths
        
        # Look for subtomogram files (each file is a particle)
        for ext in ['.mrc', '.em', '.h5', '.hdf5', '.nii', '.nii.gz']:
            for file_path in split_dir.glob(f"*{ext}"):
                # Skip if it's a compressed nii.gz and we already found the .nii.gz version
                if ext == '.nii' and file_path.with_suffix('.nii.gz').exists():
                    continue
                data_paths.append(str(file_path))
        
        # Sort for reproducibility
        data_paths.sort()
        return data_paths
    
    def _load_subtomogram(self, path: str) -> np.ndarray:
        """Load subtomogram from file"""
        path = Path(path)
        
        if path.suffix == '.mrc':
            with mrcfile.open(path, mode='r') as mrc:
                data = mrc.data.astype(np.float32)
        elif path.suffix == '.em':
            # EM file loading (simplified)
            with open(path, 'rb') as f:
                # This is a simplified version - real EM loading would be more complex
                data = np.frombuffer(f.read(), dtype=np.float32)
                # Reshape based on header info (simplified)
                data = data.reshape((64, 64, 64))  # Placeholder
        elif path.suffix in ['.h5', '.hdf5']:
            with h5py.File(path, 'r') as f:
                # Try common keys
                if 'data' in f:
                    data = f['data'][:].astype(np.float32)
                elif 'subtomogram' in f:
                    data = f['subtomogram'][:].astype(np.float32)
                else:
                    # Get first dataset
                    key = list(f.keys())[0]
                    data = f[key][:].astype(np.float32)
        elif path.suffix in ['.nii', '.gz']:
            try:
                import nibabel as nib
                img = nib.load(str(path))
                data = img.get_fdata().astype(np.float32)
            except ImportError:
                raise ImportError("nibabel is required for NIfTI files. Install with: pip install nibabel")
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
        
        return data
    
    def _create_mask(self, shape: Tuple[int, ...]) -> np.ndarray:
        """Create a mask for the subtomogram"""
        mask = np.zeros(shape, dtype=np.float32)
        
        if self.mask_type == "full":
            # Full volume mask (for segmentation tasks where entire volume is particle)
            mask.fill(1.0)
        elif self.mask_type == "center":
            # Center region mask (for tasks where center is particle)
            center = tuple(s // 2 for s in shape)
            
            # Calculate radius
            if self.mask_radius is None:
                # Default: 1/4 of minimum dimension
                radius = min(shape) // 4
            elif 0 < self.mask_radius < 1:
                # Relative radius (0.0-1.0): percentage of minimum dimension
                radius = min(shape) * self.mask_radius
            else:
                # Absolute radius value
                radius = self.mask_radius
            
            # Create spherical mask
            coords = np.ogrid[:shape[0], :shape[1], :shape[2]]
            dist = np.sqrt(sum((c - c0)**2 for c, c0 in zip(coords, center)))
            mask[dist <= radius] = 1.0
        
        return mask
    
    def __len__(self) -> int:
        """Return the number of subtomograms"""
        return len(self.data_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a subtomogram sample"""
        if idx >= len(self.data_paths):
            raise IndexError(f"Index {idx} out of range [0, {len(self.data_paths)})")
        
        # Load subtomogram
        subtomogram = self._load_subtomogram(self.data_paths[idx])
        
        # Create mask if needed
        if self.create_mask:
            mask = self._create_mask(subtomogram.shape)
        else:
            mask = np.ones_like(subtomogram)
        
        # Convert to tensors
        image = torch.from_numpy(subtomogram).unsqueeze(0)  # Add channel dimension
        mask = torch.from_numpy(mask).unsqueeze(0)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)
        
        return {
            'image': image,
            'mask': mask,
            'path': self.data_paths[idx]
        }


class CryoETDataLoader:
    """
    Data loader factory for cryoET datasets
    """
    
    @staticmethod
    def create_dataloaders(
        data_dir: str,
        batch_size: int = 4,
        num_workers: int = 4,
        pin_memory: bool = True,
        patch_size: Tuple[int, int, int] = (64, 64, 64),
        overlap: float = 0.5,
        train_transform: Optional[callable] = None,
        val_transform: Optional[callable] = None,
        target_transform: Optional[callable] = None,
        use_subtomograms: bool = False,
        mask_type: str = "full",
        mask_radius: Optional[float] = None
    ) -> Dict[str, DataLoader]:
        """
        Create data loaders for train, validation, and test sets
        
        Args:
            data_dir: Path to the data directory
            batch_size: Batch size for data loaders
            num_workers: Number of worker processes
            pin_memory: Whether to pin memory
            patch_size: Size of patches to extract (only used if use_subtomograms=False)
            overlap: Overlap ratio between patches (only used if use_subtomograms=False)
            train_transform: Transform for training data
            val_transform: Transform for validation data
            target_transform: Transform for target data
            use_subtomograms: If True, use SubtomogramDataset (for pre-extracted particles)
            mask_type: Type of mask for subtomograms ("full" or "center")
            mask_radius: Radius for center mask. None = auto (min(shape)//4), 
                        float 0-1 = relative (e.g., 0.25 = 25%), 
                        float >1 = absolute pixels
            
        Returns:
            Dictionary containing train, val, and test data loaders
        """
        dataloaders = {}
        
        for split in ['train', 'val', 'test']:
            if use_subtomograms:
                # Use subtomogram dataset (pre-extracted particles)
                dataset = SubtomogramDataset(
                    data_dir=data_dir,
                    split=split,
                    transform=train_transform if split == 'train' else val_transform,
                    target_transform=target_transform,
                    create_mask=True,
                    mask_type=mask_type,
                    mask_radius=mask_radius
                )
            else:
                # Use full tomogram dataset (extract patches from full tomograms)
                dataset = CryoETDataset(
                    data_dir=data_dir,
                    split=split,
                    transform=train_transform if split == 'train' else val_transform,
                    target_transform=target_transform,
                    patch_size=patch_size,
                    overlap=overlap
                )
            
            shuffle = (split == 'train')
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=pin_memory,
                drop_last=(split == 'train')
            )
            
            dataloaders[split] = dataloader
            logger.info(f"Created {split} dataloader with {len(dataset)} samples")
        
        return dataloaders
    
    @staticmethod
    def get_dataset_info(data_dir: str) -> Dict[str, int]:
        """
        Get information about the dataset
        
        Args:
            data_dir: Path to the data directory
            
        Returns:
            Dictionary with dataset information
        """
        info = {}
        
        for split in ['train', 'val', 'test']:
            dataset = CryoETDataset(data_dir=data_dir, split=split)
            info[split] = len(dataset)
        
        return info
