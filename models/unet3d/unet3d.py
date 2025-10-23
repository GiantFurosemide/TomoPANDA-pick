"""
3D U-Net implementation for cryoET particle picking
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional
from ..base.base_model import BaseModel


class DoubleConv3D(nn.Module):
    """Double 3D convolution block"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: Optional[int] = None,
        dropout: float = 0.1
    ):
        super().__init__()
        
        if mid_channels is None:
            mid_channels = out_channels
        
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Dropout3d(dropout),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout3d(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down3D(nn.Module):
    """Downsampling block for 3D U-Net"""
    
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.1):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv3D(in_channels, out_channels, dropout=dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up3D(nn.Module):
    """Upsampling block for 3D U-Net"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bilinear: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            self.conv = DoubleConv3D(in_channels, out_channels, in_channels // 2, dropout=dropout)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv3D(in_channels, out_channels, dropout=dropout)
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        
        # Handle different sizes
        diff_d = x2.size()[2] - x1.size()[2]
        diff_h = x2.size()[3] - x1.size()[3]
        diff_w = x2.size()[4] - x1.size()[4]
        
        x1 = F.pad(x1, [diff_w // 2, diff_w - diff_w // 2,
                        diff_h // 2, diff_h - diff_h // 2,
                        diff_d // 2, diff_d - diff_d // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv3D(nn.Module):
    """Output convolution for 3D U-Net"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet3D(BaseModel):
    """
    3D U-Net for cryoET particle segmentation
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        base_channels: int = 64,
        depth: int = 4,
        bilinear: bool = True,
        dropout: float = 0.1,
        attention: bool = False,
        **kwargs
    ):
        """
        Initialize 3D U-Net
        
        Args:
            in_channels: Number of input channels
            num_classes: Number of output classes
            base_channels: Number of base channels
            depth: Network depth
            bilinear: Whether to use bilinear upsampling
            dropout: Dropout rate
            attention: Whether to use attention mechanism
        """
        self.base_channels = base_channels
        self.depth = depth
        self.bilinear = bilinear
        self.dropout = dropout
        self.attention = attention
        
        super().__init__(in_channels, num_classes, **kwargs)
    
    def _build_model(self):
        """Build the 3D U-Net architecture"""
        # Encoder
        self.inc = DoubleConv3D(self.in_channels, self.base_channels, dropout=self.dropout)
        
        self.down_layers = nn.ModuleList()
        in_ch = self.base_channels
        for i in range(self.depth):
            out_ch = self.base_channels * (2 ** i)
            self.down_layers.append(Down3D(in_ch, out_ch, dropout=self.dropout))
            in_ch = out_ch
        
        # Decoder
        self.up_layers = nn.ModuleList()
        for i in range(self.depth):
            in_ch = self.base_channels * (2 ** (self.depth - i))
            out_ch = self.base_channels * (2 ** (self.depth - i - 1))
            if i == self.depth - 1:
                out_ch = self.base_channels
            self.up_layers.append(Up3D(in_ch, out_ch, self.bilinear, dropout=self.dropout))
        
        # Output layer
        self.outc = OutConv3D(self.base_channels, self.num_classes)
        
        # Attention mechanism (optional)
        if self.attention:
            self.attention_layers = nn.ModuleList()
            for i in range(self.depth):
                ch = self.base_channels * (2 ** i)
                self.attention_layers.append(
                    nn.Sequential(
                        nn.AdaptiveAvgPool3d(1),
                        nn.Conv3d(ch, ch // 16, 1),
                        nn.ReLU(inplace=True),
                        nn.Conv3d(ch // 16, ch, 1),
                        nn.Sigmoid()
                    )
                )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        # Encoder
        x1 = self.inc(x)
        
        # Store skip connections
        skip_connections = [x1]
        
        # Downsampling
        for i, down in enumerate(self.down_layers):
            x1 = down(x1)
            skip_connections.append(x1)
        
        # Remove the last skip connection (it's not needed)
        skip_connections = skip_connections[:-1]
        
        # Upsampling
        for i, up in enumerate(self.up_layers):
            x1 = up(x1, skip_connections[-(i+1)])
        
        # Output
        logits = self.outc(x1)
        
        return logits
    
    def get_model_info(self) -> dict:
        """Get model information"""
        info = super().get_model_info()
        info.update({
            'base_channels': self.base_channels,
            'depth': self.depth,
            'bilinear': self.bilinear,
            'dropout': self.dropout,
            'attention': self.attention
        })
        return info


class UNet3DPlus(UNet3D):
    """
    Enhanced 3D U-Net with additional features
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        base_channels: int = 64,
        depth: int = 4,
        bilinear: bool = True,
        dropout: float = 0.1,
        attention: bool = True,
        deep_supervision: bool = True,
        **kwargs
    ):
        """
        Initialize enhanced 3D U-Net
        
        Args:
            deep_supervision: Whether to use deep supervision
        """
        self.deep_supervision = deep_supervision
        super().__init__(
            in_channels, num_classes, base_channels, depth, 
            bilinear, dropout, attention, **kwargs
        )
    
    def _build_model(self):
        """Build the enhanced 3D U-Net architecture"""
        super()._build_model()
        
        # Deep supervision heads
        if self.deep_supervision:
            self.deep_supervision_heads = nn.ModuleList()
            for i in range(self.depth - 1):
                ch = self.base_channels * (2 ** (self.depth - i - 2))
                self.deep_supervision_heads.append(
                    nn.Conv3d(ch, self.num_classes, kernel_size=1)
                )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with deep supervision"""
        # Encoder
        x1 = self.inc(x)
        skip_connections = [x1]
        
        # Downsampling
        for i, down in enumerate(self.down_layers):
            x1 = down(x1)
            skip_connections.append(x1)
        
        skip_connections = skip_connections[:-1]
        
        # Upsampling with deep supervision
        deep_supervision_outputs = []
        
        for i, up in enumerate(self.up_layers):
            x1 = up(x1, skip_connections[-(i+1)])
            
            # Deep supervision
            if self.deep_supervision and i < len(self.deep_supervision_heads):
                ds_output = self.deep_supervision_heads[i](x1)
                # Upsample to original size
                ds_output = F.interpolate(
                    ds_output, 
                    size=x.shape[2:], 
                    mode='trilinear', 
                    align_corners=True
                )
                deep_supervision_outputs.append(ds_output)
        
        # Main output
        main_output = self.outc(x1)
        
        if self.deep_supervision and self.training:
            return main_output, deep_supervision_outputs
        else:
            return main_output
