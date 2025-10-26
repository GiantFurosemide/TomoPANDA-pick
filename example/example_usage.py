#!/usr/bin/env python3
"""
Example usage of TomoPANDA-pick framework
"""

import torch
import numpy as np
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from models import ModelFactory
from data.utils import CryoETDataLoader, CryoETPreprocessor, CryoETAugmentation
from training.loss_functions import DiceFocalLoss


def create_sample_data():
    """Create sample data for demonstration"""
    print("Creating sample data...")
    
    # Create sample tomogram
    tomogram = np.random.randn(64, 64, 64).astype(np.float32)
    
    # Create sample annotations
    annotations = {
        "particles": [
            {"center": [20, 20, 20], "radius": 5, "confidence": 0.95},
            {"center": [40, 40, 40], "radius": 3, "confidence": 0.88},
            {"center": [60, 60, 60], "radius": 4, "confidence": 0.92}
        ]
    }
    
    return tomogram, annotations


def demonstrate_preprocessing():
    """Demonstrate data preprocessing"""
    print("\n=== Data Preprocessing Demo ===")
    
    # Create sample data
    tomogram, annotations = create_sample_data()
    
    # Initialize preprocessor
    preprocessor = CryoETPreprocessor(
        normalize=True,
        noise_reduction=True,
        contrast_enhancement=True
    )
    
    # Preprocess tomogram
    processed_tomogram = preprocessor.preprocess_tomogram(tomogram)
    
    print(f"Original shape: {tomogram.shape}")
    print(f"Processed shape: {processed_tomogram.shape}")
    print(f"Original range: [{tomogram.min():.3f}, {tomogram.max():.3f}]")
    print(f"Processed range: [{processed_tomogram.min():.3f}, {processed_tomogram.max():.3f}]")


def demonstrate_augmentation():
    """Demonstrate data augmentation"""
    print("\n=== Data Augmentation Demo ===")
    
    # Create sample data
    tomogram, annotations = create_sample_data()
    
    # Initialize augmentation
    augmentation = CryoETAugmentation(
        rotation_prob=0.5,
        flip_prob=0.5,
        noise_prob=0.3
    )
    
    # Apply augmentation
    augmented_tomogram = augmentation(tomogram)
    
    print(f"Original shape: {tomogram.shape}")
    print(f"Augmented shape: {augmented_tomogram.shape}")


def demonstrate_model_creation():
    """Demonstrate model creation"""
    print("\n=== Model Creation Demo ===")
    
    # Create different models
    models = {
        'UNet3D': ModelFactory.create_model('unet3d', in_channels=1, num_classes=2),
        'ResNet3D': ModelFactory.create_model('resnet3d', in_channels=1, num_classes=2),
        'Transformer3D': ModelFactory.create_model('transformer3d', in_channels=1, num_classes=2)
    }
    
    # Test each model
    input_tensor = torch.randn(1, 1, 32, 32, 32)
    
    for name, model in models.items():
        print(f"\n{name}:")
        print(f"  Model info: {model.get_model_info()}")
        
        # Forward pass
        with torch.no_grad():
            output = model(input_tensor)
            print(f"  Input shape: {input_tensor.shape}")
            print(f"  Output shape: {output.shape}")
            print(f"  Output range: [{output.min():.3f}, {output.max():.3f}]")


def demonstrate_training_step():
    """Demonstrate training step"""
    print("\n=== Training Step Demo ===")
    
    # Create model
    model = ModelFactory.create_model('unet3d', in_channels=1, num_classes=2)
    
    # Create sample batch
    batch = {
        'image': torch.randn(2, 1, 32, 32, 32),
        'mask': torch.randint(0, 2, (2, 32, 32, 32))
    }
    
    # Training step
    model.train()
    loss = model.training_step(batch, 0)
    
    print(f"Training loss: {loss.item():.4f}")
    
    # Validation step
    model.eval()
    with torch.no_grad():
        val_loss = model.validation_step(batch, 0)
        print(f"Validation loss: {val_loss.item():.4f}")


def demonstrate_loss_functions():
    """Demonstrate loss functions"""
    print("\n=== Loss Functions Demo ===")
    
    # Create sample predictions and targets
    predictions = torch.randn(2, 2, 16, 16, 16)
    targets = torch.randint(0, 2, (2, 16, 16, 16))
    
    # Test different loss functions
    losses = {
        'DiceFocal': DiceFocalLoss(),
    }
    
    for name, loss_fn in losses.items():
        loss = loss_fn(predictions, targets)
        print(f"{name} Loss: {loss.item():.4f}")


def demonstrate_data_loader():
    """Demonstrate data loader (simplified)"""
    print("\n=== Data Loader Demo ===")
    
    # This would normally load real data
    print("Data loader would load real cryoET data from:")
    print("  - MRC files (.mrc)")
    print("  - EM files (.em)")
    print("  - HDF5 files (.h5)")
    print("  - JSON annotations")
    
    # Show expected data structure
    print("\nExpected data structure:")
    print("data/")
    print("├── raw/")
    print("│   ├── tomograms/")
    print("│   └── annotations/")
    print("└── processed/")
    print("    ├── train/")
    print("    ├── val/")
    print("    └── test/")


def main():
    """Main demonstration function"""
    print("TomoPANDA-pick Framework Demo")
    print("=" * 40)
    
    try:
        # Demonstrate different components
        demonstrate_preprocessing()
        demonstrate_augmentation()
        demonstrate_model_creation()
        demonstrate_training_step()
        demonstrate_loss_functions()
        demonstrate_data_loader()
        
        print("\n" + "=" * 40)
        print("Demo completed successfully!")
        print("\nNext steps:")
        print("1. Prepare your cryoET data")
        print("2. Run: python scripts/train.py --model unet3d")
        print("3. Evaluate your model")
        print("4. Make predictions on new data")
        
    except Exception as e:
        print(f"Error during demo: {e}")
        print("Make sure all dependencies are installed:")
        print("pip install -r requirements.txt")


if __name__ == '__main__':
    main()
