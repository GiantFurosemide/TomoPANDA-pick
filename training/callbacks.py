"""
Custom callbacks for PyTorch Lightning training
"""

from typing import List
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback


class CryoETCallbacks:
    """
    Factory for creating custom callbacks for cryoET training
    """
    
    def __init__(self):
        """Initialize callback factory"""
        pass
    
    def get_callbacks(self) -> List[Callback]:
        """
        Get list of custom callbacks
        
        Returns:
            List of callback instances
        """
        callbacks = []
        
        # Add any custom callbacks here
        # For now, we return an empty list as PyTorch Lightning
        # already provides most needed callbacks (checkpoint, early stopping, etc.)
        
        return callbacks


class ValidationVisualizationCallback(Callback):
    """
    Callback to visualize validation results
    """
    
    def __init__(self, save_dir: str = "experiments/results/visualizations", save_every_n_epochs: int = 10):
        """
        Initialize visualization callback
        
        Args:
            save_dir: Directory to save visualizations
            save_every_n_epochs: Save visualization every N epochs
        """
        super().__init__()
        self.save_dir = save_dir
        self.save_every_n_epochs = save_every_n_epochs
    
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Called at the end of validation epoch"""
        if trainer.current_epoch % self.save_every_n_epochs == 0:
            # Add visualization logic here if needed
            pass

