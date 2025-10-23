"""
Tests for model implementations
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from models import ModelFactory, UNet3D, ResNet3D, Transformer3D


class TestModelFactory:
    """Test model factory"""
    
    def test_create_unet3d(self):
        """Test creating UNet3D model"""
        model = ModelFactory.create_model('unet3d', in_channels=1, num_classes=2)
        assert isinstance(model, UNet3D)
        assert model.in_channels == 1
        assert model.num_classes == 2
    
    def test_create_resnet3d(self):
        """Test creating ResNet3D model"""
        model = ModelFactory.create_model('resnet3d', in_channels=1, num_classes=2)
        assert isinstance(model, ResNet3D)
        assert model.in_channels == 1
        assert model.num_classes == 2
    
    def test_create_transformer3d(self):
        """Test creating Transformer3D model"""
        model = ModelFactory.create_model('transformer3d', in_channels=1, num_classes=2)
        assert isinstance(model, Transformer3D)
        assert model.in_channels == 1
        assert model.num_classes == 2
    
    def test_invalid_model_type(self):
        """Test creating invalid model type"""
        with pytest.raises(ValueError):
            ModelFactory.create_model('invalid_model')
    
    def test_get_available_models(self):
        """Test getting available models"""
        models = ModelFactory.get_available_models()
        assert 'unet3d' in models
        assert 'resnet3d' in models
        assert 'transformer3d' in models
        assert 'ensemble' in models


class TestUNet3D:
    """Test UNet3D model"""
    
    def setup_method(self):
        """Setup test model"""
        self.model = UNet3D(in_channels=1, num_classes=2, base_channels=32, depth=3)
        self.batch_size = 2
        self.input_shape = (self.batch_size, 1, 32, 32, 32)
    
    def test_forward_pass(self):
        """Test forward pass"""
        x = torch.randn(self.input_shape)
        output = self.model(x)
        
        assert output.shape == (self.batch_size, 2, 32, 32, 32)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_model_info(self):
        """Test model information"""
        info = self.model.get_model_info()
        assert 'model_name' in info
        assert 'total_parameters' in info
        assert 'trainable_parameters' in info
        assert info['input_channels'] == 1
        assert info['num_classes'] == 2
    
    def test_training_step(self):
        """Test training step"""
        batch = {
            'image': torch.randn(self.input_shape),
            'mask': torch.randint(0, 2, (self.batch_size, 32, 32, 32))
        }
        
        loss = self.model.training_step(batch, 0)
        assert isinstance(loss, torch.Tensor)
        assert loss.item() > 0
    
    def test_validation_step(self):
        """Test validation step"""
        batch = {
            'image': torch.randn(self.input_shape),
            'mask': torch.randint(0, 2, (self.batch_size, 32, 32, 32))
        }
        
        loss = self.model.validation_step(batch, 0)
        assert isinstance(loss, torch.Tensor)
        assert loss.item() > 0


class TestResNet3D:
    """Test ResNet3D model"""
    
    def setup_method(self):
        """Setup test model"""
        self.model = ResNet3D(in_channels=1, num_classes=2, base_channels=32, depth=18)
        self.batch_size = 2
        self.input_shape = (self.batch_size, 1, 32, 32, 32)
    
    def test_forward_pass(self):
        """Test forward pass"""
        x = torch.randn(self.input_shape)
        output = self.model(x)
        
        assert output.shape == (self.batch_size, 2, 32, 32, 32)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


class TestTransformer3D:
    """Test Transformer3D model"""
    
    def setup_method(self):
        """Setup test model"""
        self.model = Transformer3D(
            in_channels=1, 
            num_classes=2, 
            embed_dim=128, 
            depth=4,
            patch_size=(8, 8, 8)
        )
        self.batch_size = 2
        self.input_shape = (self.batch_size, 1, 32, 32, 32)
    
    def test_forward_pass(self):
        """Test forward pass"""
        x = torch.randn(self.input_shape)
        output = self.model(x)
        
        assert output.shape == (self.batch_size, 2, 32, 32, 32)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


class TestModelConsistency:
    """Test model consistency across different configurations"""
    
    def test_different_input_sizes(self):
        """Test models with different input sizes"""
        model = UNet3D(in_channels=1, num_classes=2, base_channels=32)
        
        # Test different input sizes
        sizes = [(1, 1, 16, 16, 16), (1, 1, 32, 32, 32), (1, 1, 64, 64, 64)]
        
        for size in sizes:
            x = torch.randn(size)
            output = model(x)
            assert output.shape[0] == size[0]  # Batch size
            assert output.shape[1] == 2  # Number of classes
            assert output.shape[2:] == size[2:]  # Spatial dimensions
    
    def test_different_batch_sizes(self):
        """Test models with different batch sizes"""
        model = UNet3D(in_channels=1, num_classes=2, base_channels=32)
        
        batch_sizes = [1, 2, 4]
        
        for batch_size in batch_sizes:
            x = torch.randn(batch_size, 1, 32, 32, 32)
            output = model(x)
            assert output.shape[0] == batch_size
    
    def test_gradient_flow(self):
        """Test gradient flow through models"""
        model = UNet3D(in_channels=1, num_classes=2, base_channels=32)
        x = torch.randn(1, 1, 32, 32, 32, requires_grad=True)
        
        output = model(x)
        loss = output.sum()
        loss.backward()
        
        # Check that gradients are computed
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        
        # Check that model parameters have gradients
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None
                assert not torch.isnan(param.grad).any()


if __name__ == '__main__':
    pytest.main([__file__])
