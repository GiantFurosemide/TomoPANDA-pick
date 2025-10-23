"""
Loss functions for cryoET particle picking
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation
    """
    
    def __init__(
        self,
        smooth: float = 1e-6,
        ignore_index: Optional[int] = None,
        reduction: str = 'mean'
    ):
        """
        Initialize Dice Loss
        
        Args:
            smooth: Smoothing factor to avoid division by zero
            ignore_index: Index to ignore in loss calculation
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate Dice Loss
        
        Args:
            inputs: Predicted logits (B, C, H, W, D)
            targets: Ground truth masks (B, H, W, D)
            
        Returns:
            Dice loss
        """
        # Convert logits to probabilities
        inputs = F.softmax(inputs, dim=1)
        
        # Get number of classes
        num_classes = inputs.shape[1]
        
        # Calculate Dice loss for each class
        dice_losses = []
        
        for c in range(num_classes):
            if self.ignore_index is not None and c == self.ignore_index:
                continue
                
            # Get predictions and targets for class c
            pred_c = inputs[:, c, ...]
            target_c = (targets == c).float()
            
            # Calculate intersection and union
            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum()
            
            # Calculate Dice coefficient
            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
            dice_loss = 1.0 - dice
            
            dice_losses.append(dice_loss)
        
        # Average over classes
        if dice_losses:
            loss = torch.stack(dice_losses).mean()
        else:
            loss = torch.tensor(0.0, device=inputs.device)
        
        return loss


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance
    """
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        ignore_index: Optional[int] = None,
        reduction: str = 'mean'
    ):
        """
        Initialize Focal Loss
        
        Args:
            alpha: Weighting factor for rare class
            gamma: Focusing parameter
            ignore_index: Index to ignore in loss calculation
            reduction: Reduction method
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate Focal Loss
        
        Args:
            inputs: Predicted logits (B, C, H, W, D)
            targets: Ground truth masks (B, H, W, D)
            
        Returns:
            Focal loss
        """
        # Convert logits to probabilities
        probs = F.softmax(inputs, dim=1)
        
        # Create one-hot encoding for targets
        num_classes = inputs.shape[1]
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 4, 1, 2, 3).float()
        
        # Calculate cross entropy
        ce_loss = F.cross_entropy(inputs, targets, ignore_index=self.ignore_index, reduction='none')
        
        # Calculate p_t
        p_t = probs * targets_one_hot
        p_t = p_t.sum(dim=1)
        
        # Calculate alpha_t
        alpha_t = self.alpha * targets_one_hot.sum(dim=1) + (1 - self.alpha) * (1 - targets_one_hot.sum(dim=1))
        
        # Calculate focal loss
        focal_loss = alpha_t * (1 - p_t) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class DiceFocalLoss(nn.Module):
    """
    Combined Dice and Focal Loss
    """
    
    def __init__(
        self,
        dice_weight: float = 0.5,
        focal_weight: float = 0.5,
        dice_smooth: float = 1e-6,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        ignore_index: Optional[int] = None
    ):
        """
        Initialize combined loss
        
        Args:
            dice_weight: Weight for Dice loss
            focal_weight: Weight for Focal loss
            dice_smooth: Smoothing factor for Dice loss
            focal_alpha: Alpha parameter for Focal loss
            focal_gamma: Gamma parameter for Focal loss
            ignore_index: Index to ignore in loss calculation
        """
        super().__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        
        self.dice_loss = DiceLoss(
            smooth=dice_smooth,
            ignore_index=ignore_index
        )
        
        self.focal_loss = FocalLoss(
            alpha=focal_alpha,
            gamma=focal_gamma,
            ignore_index=ignore_index
        )
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate combined loss
        
        Args:
            inputs: Predicted logits
            targets: Ground truth masks
            
        Returns:
            Combined loss
        """
        dice_loss = self.dice_loss(inputs, targets)
        focal_loss = self.focal_loss(inputs, targets)
        
        total_loss = self.dice_weight * dice_loss + self.focal_weight * focal_loss
        
        return total_loss


class TverskyLoss(nn.Module):
    """
    Tversky Loss for segmentation
    """
    
    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        smooth: float = 1e-6
    ):
        """
        Initialize Tversky Loss
        
        Args:
            alpha: Weight for false positives
            beta: Weight for false negatives
            smooth: Smoothing factor
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate Tversky Loss
        
        Args:
            inputs: Predicted logits (B, C, H, W, D)
            targets: Ground truth masks (B, H, W, D)
            
        Returns:
            Tversky loss
        """
        # Convert logits to probabilities
        inputs = F.softmax(inputs, dim=1)
        
        # Get number of classes
        num_classes = inputs.shape[1]
        
        # Calculate Tversky loss for each class
        tversky_losses = []
        
        for c in range(num_classes):
            # Get predictions and targets for class c
            pred_c = inputs[:, c, ...]
            target_c = (targets == c).float()
            
            # Calculate true positives, false positives, and false negatives
            tp = (pred_c * target_c).sum()
            fp = (pred_c * (1 - target_c)).sum()
            fn = ((1 - pred_c) * target_c).sum()
            
            # Calculate Tversky coefficient
            tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
            tversky_loss = 1.0 - tversky
            
            tversky_losses.append(tversky_loss)
        
        # Average over classes
        if tversky_losses:
            loss = torch.stack(tversky_losses).mean()
        else:
            loss = torch.tensor(0.0, device=inputs.device)
        
        return loss


class IoULoss(nn.Module):
    """
    IoU (Jaccard) Loss for segmentation
    """
    
    def __init__(self, smooth: float = 1e-6):
        """
        Initialize IoU Loss
        
        Args:
            smooth: Smoothing factor
        """
        super().__init__()
        self.smooth = smooth
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate IoU Loss
        
        Args:
            inputs: Predicted logits (B, C, H, W, D)
            targets: Ground truth masks (B, H, W, D)
            
        Returns:
            IoU loss
        """
        # Convert logits to probabilities
        inputs = F.softmax(inputs, dim=1)
        
        # Get number of classes
        num_classes = inputs.shape[1]
        
        # Calculate IoU loss for each class
        iou_losses = []
        
        for c in range(num_classes):
            # Get predictions and targets for class c
            pred_c = inputs[:, c, ...]
            target_c = (targets == c).float()
            
            # Calculate intersection and union
            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum() - intersection
            
            # Calculate IoU
            iou = (intersection + self.smooth) / (union + self.smooth)
            iou_loss = 1.0 - iou
            
            iou_losses.append(iou_loss)
        
        # Average over classes
        if iou_losses:
            loss = torch.stack(iou_losses).mean()
        else:
            loss = torch.tensor(0.0, device=inputs.device)
        
        return loss
