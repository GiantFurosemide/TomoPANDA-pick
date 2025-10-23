"""
Data augmentation utilities for cryoET 3D data
"""

import numpy as np
import torch
from typing import Tuple, Optional, Union, List
import random
from scipy.ndimage import rotate, zoom, map_coordinates
from scipy.ndimage.interpolation import affine_transform
from loguru import logger


class CryoETAugmentation:
    """
    Data augmentation for cryoET 3D data
    """
    
    def __init__(
        self,
        rotation_prob: float = 0.5,
        flip_prob: float = 0.5,
        elastic_prob: float = 0.3,
        noise_prob: float = 0.2,
        intensity_prob: float = 0.3,
        rotation_range: Tuple[float, float] = (-15, 15),
        noise_std: float = 0.1,
        intensity_range: Tuple[float, float] = (0.8, 1.2)
    ):
        """
        Initialize augmentation parameters
        
        Args:
            rotation_prob: Probability of applying rotation
            flip_prob: Probability of applying flip
            elastic_prob: Probability of applying elastic deformation
            noise_prob: Probability of adding noise
            intensity_prob: Probability of intensity variation
            rotation_range: Range of rotation angles in degrees
            noise_std: Standard deviation of noise
            intensity_range: Range of intensity scaling
        """
        self.rotation_prob = rotation_prob
        self.flip_prob = flip_prob
        self.elastic_prob = elastic_prob
        self.noise_prob = noise_prob
        self.intensity_prob = intensity_prob
        self.rotation_range = rotation_range
        self.noise_std = noise_std
        self.intensity_range = intensity_range
    
    def __call__(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Apply augmentations to image and mask
        
        Args:
            image: Input image
            mask: Optional mask (same transformations will be applied)
            
        Returns:
            Augmented image and mask (if provided)
        """
        augmented_image = image.copy()
        augmented_mask = mask.copy() if mask is not None else None
        
        # Random rotation
        if random.random() < self.rotation_prob:
            angle = random.uniform(*self.rotation_range)
            augmented_image = self._rotate_3d(augmented_image, angle)
            if augmented_mask is not None:
                augmented_mask = self._rotate_3d(augmented_mask, angle)
        
        # Random flip
        if random.random() < self.flip_prob:
            axis = random.choice([0, 1, 2])  # Random axis
            augmented_image = np.flip(augmented_image, axis=axis)
            if augmented_mask is not None:
                augmented_mask = np.flip(augmented_mask, axis=axis)
        
        # Elastic deformation
        if random.random() < self.elastic_prob:
            augmented_image = self._elastic_deformation(augmented_image)
            if augmented_mask is not None:
                augmented_mask = self._elastic_deformation(augmented_mask)
        
        # Add noise
        if random.random() < self.noise_prob:
            augmented_image = self._add_noise(augmented_image)
        
        # Intensity variation
        if random.random() < self.intensity_prob:
            augmented_image = self._intensity_variation(augmented_image)
        
        if augmented_mask is not None:
            return augmented_image, augmented_mask
        else:
            return augmented_image
    
    def _rotate_3d(self, image: np.ndarray, angle: float) -> np.ndarray:
        """Apply 3D rotation"""
        # Random rotation axis
        axis = random.choice([0, 1, 2])
        
        # Apply rotation
        rotated = rotate(image, angle, axes=(axis, (axis + 1) % 3), 
                        reshape=False, order=1)
        
        return rotated
    
    def _elastic_deformation(self, image: np.ndarray, alpha: float = 1000, sigma: float = 50) -> np.ndarray:
        """Apply elastic deformation"""
        shape = image.shape
        
        # Generate random displacement fields
        dx = gaussian_filter((np.random.random(shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        dy = gaussian_filter((np.random.random(shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        dz = gaussian_filter((np.random.random(shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        
        # Create coordinate grids
        z, y, x = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing='ij')
        
        # Apply displacement
        indices = [z + dz, y + dy, x + dx]
        
        # Interpolate
        deformed = map_coordinates(image, indices, order=1, mode='reflect')
        
        return deformed
    
    def _add_noise(self, image: np.ndarray) -> np.ndarray:
        """Add Gaussian noise"""
        noise = np.random.normal(0, self.noise_std, image.shape)
        return image + noise
    
    def _intensity_variation(self, image: np.ndarray) -> np.ndarray:
        """Apply intensity variation"""
        scale = random.uniform(*self.intensity_range)
        offset = random.uniform(-0.1, 0.1)
        
        return np.clip(image * scale + offset, 0, 1)
    
    def _gaussian_filter(self, image: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        """Apply Gaussian filter"""
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(image, sigma=sigma)


class CryoETAugmentationPipeline:
    """
    Pipeline for applying multiple augmentations
    """
    
    def __init__(self, augmentations: List[CryoETAugmentation]):
        """
        Initialize augmentation pipeline
        
        Args:
            augmentations: List of augmentation objects
        """
        self.augmentations = augmentations
    
    def __call__(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Apply all augmentations in sequence
        
        Args:
            image: Input image
            mask: Optional mask
            
        Returns:
            Augmented image and mask
        """
        current_image = image
        current_mask = mask
        
        for augmentation in self.augmentations:
            if current_mask is not None:
                current_image, current_mask = augmentation(current_image, current_mask)
            else:
                current_image = augmentation(current_image)
        
        if current_mask is not None:
            return current_image, current_mask
        else:
            return current_image


class CryoETAugmentationFactory:
    """
    Factory for creating augmentation pipelines
    """
    
    @staticmethod
    def create_training_augmentation() -> CryoETAugmentationPipeline:
        """Create augmentation pipeline for training"""
        augmentations = [
            CryoETAugmentation(
                rotation_prob=0.5,
                flip_prob=0.5,
                elastic_prob=0.3,
                noise_prob=0.2,
                intensity_prob=0.3
            )
        ]
        return CryoETAugmentationPipeline(augmentations)
    
    @staticmethod
    def create_validation_augmentation() -> CryoETAugmentationPipeline:
        """Create augmentation pipeline for validation (minimal augmentation)"""
        augmentations = [
            CryoETAugmentation(
                rotation_prob=0.0,
                flip_prob=0.0,
                elastic_prob=0.0,
                noise_prob=0.0,
                intensity_prob=0.0
            )
        ]
        return CryoETAugmentationPipeline(augmentations)
    
    @staticmethod
    def create_test_augmentation() -> CryoETAugmentationPipeline:
        """Create augmentation pipeline for testing (no augmentation)"""
        augmentations = [
            CryoETAugmentation(
                rotation_prob=0.0,
                flip_prob=0.0,
                elastic_prob=0.0,
                noise_prob=0.0,
                intensity_prob=0.0
            )
        ]
        return CryoETAugmentationPipeline(augmentations)


# PyTorch transforms for integration with DataLoader
class CryoETTransform:
    """
    PyTorch transform wrapper for cryoET augmentations
    """
    
    def __init__(self, augmentation_pipeline: CryoETAugmentationPipeline):
        """
        Initialize transform
        
        Args:
            augmentation_pipeline: Augmentation pipeline to apply
        """
        self.augmentation_pipeline = augmentation_pipeline
    
    def __call__(self, sample: dict) -> dict:
        """
        Apply transform to sample
        
        Args:
            sample: Dictionary containing 'image' and 'mask' keys
            
        Returns:
            Transformed sample
        """
        image = sample['image'].numpy()
        mask = sample['mask'].numpy() if 'mask' in sample else None
        
        if mask is not None:
            augmented_image, augmented_mask = self.augmentation_pipeline(image, mask)
            sample['image'] = torch.from_numpy(augmented_image)
            sample['mask'] = torch.from_numpy(augmented_mask)
        else:
            augmented_image = self.augmentation_pipeline(image)
            sample['image'] = torch.from_numpy(augmented_image)
        
        return sample
