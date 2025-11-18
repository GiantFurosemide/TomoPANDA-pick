"""
Simple 3D CNN implementation for cryoET particle picking
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from ..base.base_model import BaseModel


class SimpleCNN3D(BaseModel):
    """
    Simple 3D CNN model for segmentation
    A straightforward encoder-decoder architecture with 3D convolutions
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        base_channels: int = 32,
        num_layers: int = 3,
        dropout: float = 0.1,
        **kwargs
    ):
        """
        Initialize Simple 3D CNN
        
        Args:
            in_channels: Number of input channels
            num_classes: Number of output classes
            base_channels: Base number of channels (will be doubled in each layer)
            num_layers: Number of downsampling/upsampling layers
            dropout: Dropout rate
        """
        self.base_channels = base_channels
        self.num_layers = num_layers
        self.dropout = dropout
        
        super().__init__(
            in_channels=in_channels,
            num_classes=num_classes,
            **kwargs
        )
    
    def _build_model(self):
        """Build the simple 3D CNN architecture"""
        # Encoder (downsampling path)
        encoder_layers = []
        in_ch = self.in_channels
        
        for i in range(self.num_layers):
            out_ch = self.base_channels * (2 ** i)
            encoder_layers.append(
                nn.Sequential(
                    nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm3d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.Dropout3d(self.dropout),
                    nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm3d(out_ch),
                    nn.ReLU(inplace=True)
                )
            )
            in_ch = out_ch
        
        self.encoder = nn.ModuleList(encoder_layers)
        
        # Bottleneck
        bottleneck_ch = self.base_channels * (2 ** self.num_layers)
        self.bottleneck = nn.Sequential(
            nn.Conv3d(in_ch, bottleneck_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(bottleneck_ch),
            nn.ReLU(inplace=True),
            nn.Dropout3d(self.dropout),
            nn.Conv3d(bottleneck_ch, bottleneck_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(bottleneck_ch),
            nn.ReLU(inplace=True)
        )
        
        # Decoder (upsampling path)
        decoder_layers = []
        in_ch = bottleneck_ch
        
        for i in range(self.num_layers - 1, -1, -1):
            out_ch = self.base_channels * (2 ** i)
            # Upsampling layer
            upsample = nn.Sequential(
                nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2, bias=False),
                nn.BatchNorm3d(out_ch),
                nn.ReLU(inplace=True)
            )
            # Convolution layers
            conv_block = nn.Sequential(
                nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm3d(out_ch),
                nn.ReLU(inplace=True),
                nn.Dropout3d(self.dropout),
                nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm3d(out_ch),
                nn.ReLU(inplace=True)
            )
            decoder_layers.append((upsample, conv_block))
            in_ch = out_ch
        
        self.decoder = nn.ModuleList([nn.ModuleList(layer) for layer in decoder_layers])
        
        # Output layer
        self.output = nn.Conv3d(
            self.base_channels, 
            self.num_classes, 
            kernel_size=1
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (B, C, D, H, W)
            
        Returns:
            Output tensor of shape (B, num_classes, D, H, W)
        """
        # Encoder path (store outputs for skip connections)
        encoder_outputs = []
        for encoder_layer in self.encoder:
            x = encoder_layer(x)
            encoder_outputs.append(x)
            # Downsample
            x = F.max_pool3d(x, kernel_size=2, stride=2)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder path with skip connections
        for i, (upsample_layer, conv_block) in enumerate(self.decoder):
            # Upsample
            x = upsample_layer(x)
            
            # Skip connection (add)
            if i < len(encoder_outputs):
                skip = encoder_outputs[-(i+1)]
                # Handle dimension mismatch
                if x.shape[2:] != skip.shape[2:]:
                    # Adjust skip connection size if needed
                    _, _, d, h, w = x.shape
                    skip = F.interpolate(skip, size=(d, h, w), mode='trilinear', align_corners=False)
                # Add skip connection (element-wise addition)
                if x.shape[1] == skip.shape[1]:
                    x = x + skip
            
            # Convolution block
            x = conv_block(x)
        
        # Output layer
        x = self.output(x)
        
        return x
    
    def get_model_info(self) -> dict:
        """Get model information"""
        info = super().get_model_info()
        info.update({
            'base_channels': self.base_channels,
            'num_layers': self.num_layers,
            'dropout': self.dropout
        })
        return info

