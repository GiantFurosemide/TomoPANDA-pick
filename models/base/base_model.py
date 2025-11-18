"""
Base model class for TomoPANDA-pick
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import pytorch_lightning as pl
from loguru import logger


class BaseModel(pl.LightningModule, ABC):
    """
    Base model class for all TomoPANDA-pick models
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        **kwargs
    ):
        """
        Initialize base model
        
        Args:
            in_channels: Number of input channels
            num_classes: Number of output classes
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
        """
        super().__init__()
        self.save_hyperparameters()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Initialize model architecture
        self._build_model()
        
        # Loss function
        self.criterion = self._get_loss_function()
        
        # Metrics
        self.train_metrics = self._get_metrics()
        self.val_metrics = self._get_metrics()
        self.test_metrics = self._get_metrics()
    
    @abstractmethod
    def _build_model(self):
        """Build the model architecture"""
        pass
    
    def _get_loss_function(self) -> nn.Module:
        """Get loss function"""
        from training.loss_functions import DiceFocalLoss
        return DiceFocalLoss()
    
    def _get_metrics(self):
        """Get metrics for evaluation"""
        from training.metrics import SegmentationMetrics
        return SegmentationMetrics(num_classes=self.num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        return self.model(x)
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Training step
        
        Args:
            batch: Batch of data
            batch_idx: Batch index
            
        Returns:
            Loss tensor
        """
        images = batch['image']
        masks = batch['mask']
        
        # Forward pass
        outputs = self(images)
        
        # Calculate loss
        loss = self.criterion(outputs, masks)
        
        # Calculate metrics
        preds = torch.argmax(outputs, dim=1)
        self.train_metrics(preds, masks)
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log_dict(self.train_metrics, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Validation step
        
        Args:
            batch: Batch of data
            batch_idx: Batch index
            
        Returns:
            Loss tensor
        """
        images = batch['image']
        masks = batch['mask']
        
        # Forward pass
        outputs = self(images)
        
        # Calculate loss
        loss = self.criterion(outputs, masks)
        
        # Calculate metrics
        preds = torch.argmax(outputs, dim=1)
        self.val_metrics(preds, masks)
        
        # Log metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log_dict(self.val_metrics, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Test step
        
        Args:
            batch: Batch of data
            batch_idx: Batch index
            
        Returns:
            Loss tensor
        """
        images = batch['image']
        masks = batch['mask']
        
        # Forward pass
        outputs = self(images)
        
        # Calculate loss
        loss = self.criterion(outputs, masks)
        
        # Calculate metrics
        preds = torch.argmax(outputs, dim=1)
        self.test_metrics(preds, masks)
        
        # Log metrics
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log_dict(self.test_metrics, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):
        """Configure optimizer and scheduler"""
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=1e-6
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }
    
    def predict_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Prediction step
        
        Args:
            batch: Batch of data
            batch_idx: Batch index
            
        Returns:
            Dictionary with predictions
        """
        images = batch['image']
        
        # Forward pass
        with torch.no_grad():
            outputs = self(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
        
        return {
            'predictions': preds,
            'probabilities': probs,
            'logits': outputs
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information
        
        Returns:
            Dictionary with model information
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': self.__class__.__name__,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'input_channels': self.in_channels,
            'num_classes': self.num_classes,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay
        }


class ModelFactory:
    """
    Factory for creating models
    """
    
    @staticmethod
    def create_model(
        model_type: str,
        in_channels: int = 1,
        num_classes: int = 2,
        **kwargs
    ) -> BaseModel:
        """
        Create a model instance
        
        Args:
            model_type: Type of model to create
            in_channels: Number of input channels
            num_classes: Number of output classes
            **kwargs: Additional model parameters
            
        Returns:
            Model instance
        """
        if model_type.lower() == 'unet3d':
            from .unet3d.unet3d import UNet3D
            return UNet3D(in_channels=in_channels, num_classes=num_classes, **kwargs)
        
        elif model_type.lower() == 'resnet3d':
            from .resnet3d.resnet3d import ResNet3D
            return ResNet3D(in_channels=in_channels, num_classes=num_classes, **kwargs)
        
        elif model_type.lower() == 'transformer3d':
            from .transformer3d.transformer3d import Transformer3D
            return Transformer3D(in_channels=in_channels, num_classes=num_classes, **kwargs)
        
        elif model_type.lower() == 'ensemble':
            from .ensemble.ensemble_model import EnsembleModel
            return EnsembleModel(in_channels=in_channels, num_classes=num_classes, **kwargs)
        
        elif model_type.lower() in ['cnn3d', 'simplecnn3d', 'simple_cnn3d']:
            from .cnn3d.cnn3d import SimpleCNN3D
            return SimpleCNN3D(in_channels=in_channels, num_classes=num_classes, **kwargs)
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    @staticmethod
    def get_available_models() -> list:
        """Get list of available model types"""
        return ['unet3d', 'resnet3d', 'transformer3d', 'ensemble', 'cnn3d']
