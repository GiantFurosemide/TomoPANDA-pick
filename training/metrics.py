"""
Metrics for segmentation evaluation
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score


class SegmentationMetrics:
    """
    Metrics for 3D segmentation evaluation
    """
    
    def __init__(self, num_classes: int = 2, threshold: float = 0.5):
        """
        Initialize metrics
        
        Args:
            num_classes: Number of classes
            threshold: Threshold for binary classification
        """
        self.num_classes = num_classes
        self.threshold = threshold
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0
        self.total_samples = 0
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        """
        Update metrics with a batch of predictions and targets
        
        Args:
            predictions: Model predictions (B, C, H, W, D) or (B, H, W, D)
            targets: Ground truth labels (B, C, H, W, D) or (B, H, W, D)
        """
        # Convert to binary if needed
        if predictions.dim() == 5:  # (B, C, H, W, D)
            predictions = torch.argmax(predictions, dim=1)
        elif predictions.dim() == 4:  # (B, H, W, D)
            if predictions.max() <= 1.0:  # Probabilities
                predictions = (predictions > self.threshold).long()
        
        if targets.dim() == 5:  # (B, C, H, W, D)
            targets = torch.argmax(targets, dim=1)
        elif targets.dim() == 4:  # (B, H, W, D)
            targets = targets.long()
        
        # Flatten for metric calculation
        pred_flat = predictions.flatten().cpu().numpy()
        target_flat = targets.flatten().cpu().numpy()
        
        # Calculate confusion matrix components
        tp = np.sum((pred_flat == 1) & (target_flat == 1))
        fp = np.sum((pred_flat == 1) & (target_flat == 0))
        fn = np.sum((pred_flat == 0) & (target_flat == 1))
        tn = np.sum((pred_flat == 0) & (target_flat == 0))
        
        self.tp += tp
        self.fp += fp
        self.fn += fn
        self.tn += tn
        self.total_samples += len(pred_flat)
    
    def compute(self) -> Dict[str, float]:
        """
        Compute all metrics
        
        Returns:
            Dictionary of metric values
        """
        metrics = {}
        
        # Precision
        if self.tp + self.fp > 0:
            metrics['precision'] = self.tp / (self.tp + self.fp)
        else:
            metrics['precision'] = 0.0
        
        # Recall
        if self.tp + self.fn > 0:
            metrics['recall'] = self.tp / (self.tp + self.fn)
        else:
            metrics['recall'] = 0.0
        
        # F1 Score
        if metrics['precision'] + metrics['recall'] > 0:
            metrics['f1'] = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall'])
        else:
            metrics['f1'] = 0.0
        
        # IoU (Jaccard Index)
        if self.tp + self.fp + self.fn > 0:
            metrics['iou'] = self.tp / (self.tp + self.fp + self.fn)
        else:
            metrics['iou'] = 0.0
        
        # Dice Coefficient
        if self.tp + self.fp + self.fn > 0:
            metrics['dice'] = 2 * self.tp / (2 * self.tp + self.fp + self.fn)
        else:
            metrics['dice'] = 0.0
        
        # Accuracy
        if self.total_samples > 0:
            metrics['accuracy'] = (self.tp + self.tn) / self.total_samples
        else:
            metrics['accuracy'] = 0.0
        
        return metrics
    
    def __call__(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """Compute metrics for given predictions and targets"""
        self.update(predictions, targets)
        return self.compute()

