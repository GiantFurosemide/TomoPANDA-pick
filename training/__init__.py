"""
Training package for TomoPANDA-pick
"""

from .loss_functions import DiceLoss, FocalLoss, DiceFocalLoss

# Try to import optional modules
try:
    from .metrics import SegmentationMetrics
except ImportError:
    SegmentationMetrics = None

try:
    from .callbacks import CryoETCallbacks
except ImportError:
    CryoETCallbacks = None

try:
    from .trainer import CryoETTrainer
except ImportError:
    CryoETTrainer = None

__all__ = [
    "DiceLoss",
    "FocalLoss", 
    "DiceFocalLoss",
]

if SegmentationMetrics is not None:
    __all__.append("SegmentationMetrics")
if CryoETCallbacks is not None:
    __all__.append("CryoETCallbacks")
if CryoETTrainer is not None:
    __all__.append("CryoETTrainer")
