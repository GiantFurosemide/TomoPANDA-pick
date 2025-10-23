"""
Data preprocessing utilities for cryoET data
"""

import numpy as np
import torch
from typing import Tuple, Optional, Union, List
from scipy import ndimage
from scipy.ndimage import gaussian_filter, median_filter
from skimage import exposure, filters
from loguru import logger


class CryoETPreprocessor:
    """
    Preprocessing utilities for cryoET tomogram data
    """
    
    def __init__(
        self,
        normalize: bool = True,
        noise_reduction: bool = True,
        contrast_enhancement: bool = True,
        target_size: Optional[Tuple[int, int, int]] = None
    ):
        """
        Initialize preprocessor
        
        Args:
            normalize: Whether to normalize the data
            noise_reduction: Whether to apply noise reduction
            contrast_enhancement: Whether to enhance contrast
            target_size: Target size for resizing (H, W, D)
        """
        self.normalize = normalize
        self.noise_reduction = noise_reduction
        self.contrast_enhancement = contrast_enhancement
        self.target_size = target_size
    
    def preprocess_tomogram(self, tomogram: np.ndarray) -> np.ndarray:
        """
        Preprocess a single tomogram
        
        Args:
            tomogram: Input tomogram as numpy array
            
        Returns:
            Preprocessed tomogram
        """
        processed = tomogram.copy()
        
        # Noise reduction
        if self.noise_reduction:
            processed = self._reduce_noise(processed)
        
        # Contrast enhancement
        if self.contrast_enhancement:
            processed = self._enhance_contrast(processed)
        
        # Normalization
        if self.normalize:
            processed = self._normalize(processed)
        
        # Resize if target size is specified
        if self.target_size is not None:
            processed = self._resize(processed, self.target_size)
        
        return processed
    
    def _reduce_noise(self, image: np.ndarray) -> np.ndarray:
        """Apply noise reduction techniques"""
        # Gaussian filter for noise reduction
        sigma = 1.0
        filtered = gaussian_filter(image, sigma=sigma)
        
        # Median filter for salt-and-pepper noise
        filtered = median_filter(filtered, size=3)
        
        return filtered
    
    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Enhance contrast of the image"""
        # Histogram equalization
        enhanced = exposure.equalize_hist(image)
        
        # Adaptive histogram equalization
        enhanced = exposure.equalize_adapthist(enhanced, clip_limit=0.03)
        
        return enhanced
    
    def _normalize(self, image: np.ndarray) -> np.ndarray:
        """Normalize image to [0, 1] range"""
        # Min-max normalization
        img_min = np.min(image)
        img_max = np.max(image)
        
        if img_max > img_min:
            normalized = (image - img_min) / (img_max - img_min)
        else:
            normalized = np.zeros_like(image)
        
        return normalized.astype(np.float32)
    
    def _resize(self, image: np.ndarray, target_size: Tuple[int, int, int]) -> np.ndarray:
        """Resize image to target size"""
        from scipy.ndimage import zoom
        
        current_size = image.shape
        zoom_factors = [
            target_size[i] / current_size[i] for i in range(len(target_size))
        ]
        
        resized = zoom(image, zoom_factors, order=1)  # Linear interpolation
        return resized
    
    def preprocess_batch(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Preprocess a batch of tomograms
        
        Args:
            batch: Batch of tomograms as torch tensor (B, C, H, W, D)
            
        Returns:
            Preprocessed batch
        """
        processed_batch = []
        
        for i in range(batch.shape[0]):
            # Convert to numpy for processing
            tomogram = batch[i, 0].numpy()  # Remove channel dimension
            
            # Preprocess
            processed = self.preprocess_tomogram(tomogram)
            
            # Convert back to tensor
            processed_tensor = torch.from_numpy(processed).unsqueeze(0)
            processed_batch.append(processed_tensor)
        
        return torch.stack(processed_batch)
    
    def create_patches(
        self, 
        tomogram: np.ndarray, 
        patch_size: Tuple[int, int, int] = (64, 64, 64),
        overlap: float = 0.5
    ) -> List[np.ndarray]:
        """
        Create patches from tomogram
        
        Args:
            tomogram: Input tomogram
            patch_size: Size of patches (H, W, D)
            overlap: Overlap ratio between patches
            
        Returns:
            List of patches
        """
        patches = []
        h, w, d = tomogram.shape
        ph, pw, pd = patch_size
        
        # Calculate step size based on overlap
        step_h = int(ph * (1 - overlap))
        step_w = int(pw * (1 - overlap))
        step_d = int(pd * (1 - overlap))
        
        for z in range(0, d - pd + 1, step_d):
            for y in range(0, h - ph + 1, step_h):
                for x in range(0, w - pw + 1, step_w):
                    patch = tomogram[y:y+ph, x:x+pw, z:z+pd]
                    patches.append(patch)
        
        return patches
    
    def reconstruct_from_patches(
        self, 
        patches: List[np.ndarray], 
        original_shape: Tuple[int, int, int],
        patch_size: Tuple[int, int, int] = (64, 64, 64),
        overlap: float = 0.5
    ) -> np.ndarray:
        """
        Reconstruct tomogram from patches
        
        Args:
            patches: List of patches
            original_shape: Original tomogram shape
            patch_size: Size of patches
            overlap: Overlap ratio used during patching
            
        Returns:
            Reconstructed tomogram
        """
        h, w, d = original_shape
        ph, pw, pd = patch_size
        
        # Initialize output array
        output = np.zeros(original_shape, dtype=np.float32)
        count = np.zeros(original_shape, dtype=np.float32)
        
        # Calculate step size
        step_h = int(ph * (1 - overlap))
        step_w = int(pw * (1 - overlap))
        step_d = int(pd * (1 - overlap))
        
        patch_idx = 0
        for z in range(0, d - pd + 1, step_d):
            for y in range(0, h - ph + 1, step_h):
                for x in range(0, w - pw + 1, step_w):
                    if patch_idx < len(patches):
                        patch = patches[patch_idx]
                        output[y:y+ph, x:x+pw, z:z+pd] += patch
                        count[y:y+ph, x:x+pw, z:z+pd] += 1
                        patch_idx += 1
        
        # Average overlapping regions
        count[count == 0] = 1  # Avoid division by zero
        output = output / count
        
        return output


class CryoETDataSplitter:
    """
    Utility for splitting cryoET data into train/val/test sets
    """
    
    @staticmethod
    def split_data(
        data_dir: str,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_seed: int = 42
    ) -> Dict[str, List[str]]:
        """
        Split data into train/val/test sets
        
        Args:
            data_dir: Path to raw data directory
            train_ratio: Ratio for training set
            val_ratio: Ratio for validation set
            test_ratio: Ratio for test set
            random_seed: Random seed for reproducibility
            
        Returns:
            Dictionary with file lists for each split
        """
        import random
        from pathlib import Path
        
        # Set random seed
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # Get all tomogram files
        data_path = Path(data_dir)
        tomogram_files = []
        
        for ext in ['.mrc', '.em', '.h5', '.hdf5']:
            tomogram_files.extend(list(data_path.glob(f"*{ext}")))
        
        # Shuffle files
        random.shuffle(tomogram_files)
        
        # Calculate split indices
        total_files = len(tomogram_files)
        train_end = int(total_files * train_ratio)
        val_end = train_end + int(total_files * val_ratio)
        
        # Split files
        splits = {
            'train': [str(f) for f in tomogram_files[:train_end]],
            'val': [str(f) for f in tomogram_files[train_end:val_end]],
            'test': [str(f) for f in tomogram_files[val_end:]]
        }
        
        logger.info(f"Data split: Train={len(splits['train'])}, "
                   f"Val={len(splits['val'])}, Test={len(splits['test'])}")
        
        return splits
