#!/usr/bin/env python3
"""
Training script for TomoPANDA-pick
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from models import ModelFactory
from data.utils import CryoETDataLoader
from training.loss_functions import DiceFocalLoss
from training.metrics import SegmentationMetrics
from training.callbacks import CryoETCallbacks


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train TomoPANDA-pick model')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='unet3d',
                       choices=['unet3d', 'resnet3d', 'transformer3d', 'ensemble'],
                       help='Model type to train')
    parser.add_argument('--config', type=str, default='config/default_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--model-config', type=str, default=None,
                       help='Path to model-specific configuration file')
    
    # Data arguments
    parser.add_argument('--data-dir', type=str, default='data',
                       help='Path to data directory')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Batch size for training')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of worker processes for data loading')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--gpus', type=int, default=1,
                       help='Number of GPUs to use')
    parser.add_argument('--precision', type=int, default=32,
                       choices=[16, 32],
                       help='Training precision')
    
    # Experiment tracking
    parser.add_argument('--experiment-name', type=str, default=None,
                       help='Name for the experiment')
    parser.add_argument('--wandb-project', type=str, default='tomopanda-pick',
                       help='Weights & Biases project name')
    parser.add_argument('--wandb-entity', type=str, default=None,
                       help='Weights & Biases entity/team')
    parser.add_argument('--no-wandb', action='store_true',
                       help='Disable Weights & Biases logging')
    
    # Other arguments
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--fast-dev-run', action='store_true',
                       help='Run a quick test with a few batches')
    
    return parser.parse_args()


def load_config(config_path: str, model_config_path: str = None) -> dict:
    """Load configuration from YAML files"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if model_config_path and os.path.exists(model_config_path):
        with open(model_config_path, 'r') as f:
            model_config = yaml.safe_load(f)
            config.update(model_config)
    
    return config


def setup_loggers(config: dict, args) -> list:
    """Setup logging"""
    loggers = []
    
    # TensorBoard logger
    if config.get('experiment', {}).get('tensorboard', {}).get('enabled', True):
        tb_logger = TensorBoardLogger(
            save_dir=config['paths']['logs_dir'],
            name=args.experiment_name or f"{args.model}_experiment"
        )
        loggers.append(tb_logger)
    
    # Weights & Biases logger
    if not args.no_wandb and config.get('experiment', {}).get('wandb', {}).get('enabled', True):
        wandb_config = config['experiment']['wandb']
        wandb_logger = WandbLogger(
            project=args.wandb_project or wandb_config.get('project', 'tomopanda-pick'),
            entity=args.wandb_entity or wandb_config.get('entity'),
            name=args.experiment_name or f"{args.model}_experiment",
            tags=wandb_config.get('tags', ['cryoet', '3d', 'particle-picking'])
        )
        loggers.append(wandb_logger)
    
    return loggers


def setup_callbacks(config: dict) -> list:
    """Setup training callbacks"""
    callbacks = []
    
    # Model checkpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath=config['paths']['checkpoints_dir'],
        filename='{epoch:02d}-{val_loss:.2f}',
        monitor='val_loss',
        mode='min',
        save_top_k=3,
        save_last=True
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping
    early_stopping = EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=config.get('training', {}).get('early_stopping', {}).get('patience', 10),
        verbose=True
    )
    callbacks.append(early_stopping)
    
    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    callbacks.append(lr_monitor)
    
    # Custom callbacks
    custom_callbacks = CryoETCallbacks()
    callbacks.extend(custom_callbacks.get_callbacks())
    
    return callbacks


def main():
    """Main training function"""
    args = parse_args()
    
    # Set random seed
    pl.seed_everything(args.seed)
    
    # Load configuration
    config = load_config(args.config, args.model_config)
    
    # Override config with command line arguments
    if args.batch_size:
        config['model']['batch_size'] = args.batch_size
    if args.learning_rate:
        config['model']['learning_rate'] = args.learning_rate
    if args.weight_decay:
        config['model']['weight_decay'] = args.weight_decay
    if args.epochs:
        config['model']['num_epochs'] = args.epochs
    
    # Create model
    model = ModelFactory.create_model(
        model_type=args.model,
        in_channels=config['model']['input_channels'],
        num_classes=config['model']['num_classes'],
        learning_rate=config['model']['learning_rate'],
        weight_decay=config['model']['weight_decay'],
        **config.get('model', {}).get('architecture', {})
    )
    
    # Create data loaders
    dataloaders = CryoETDataLoader.create_dataloaders(
        data_dir=args.data_dir,
        batch_size=config['model']['batch_size'],
        num_workers=args.num_workers,
        pin_memory=True,
        patch_size=tuple(config['data']['preprocessing']['patch_size']),
        overlap=config['data']['preprocessing']['overlap']
    )
    
    # Setup loggers
    loggers = setup_loggers(config, args)
    
    # Setup callbacks
    callbacks = setup_callbacks(config)
    
    # Setup trainer
    trainer = pl.Trainer(
        max_epochs=config['model']['num_epochs'],
        gpus=args.gpus,
        precision=args.precision,
        loggers=loggers,
        callbacks=callbacks,
        fast_dev_run=args.fast_dev_run,
        resume_from_checkpoint=args.resume
    )
    
    # Start training
    print(f"Starting training with {args.model} model...")
    print(f"Training samples: {len(dataloaders['train'].dataset)}")
    print(f"Validation samples: {len(dataloaders['val'].dataset)}")
    print(f"Test samples: {len(dataloaders['test'].dataset)}")
    
    trainer.fit(
        model,
        train_dataloader=dataloaders['train'],
        val_dataloaders=dataloaders['val']
    )
    
    # Test the model
    print("Testing the model...")
    trainer.test(model, dataloaders=dataloaders['test'])
    
    print("Training completed!")


if __name__ == '__main__':
    main()
