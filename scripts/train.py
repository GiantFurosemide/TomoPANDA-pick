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
    
    # Only keep config argument - all other parameters are in YAML
    parser.add_argument('--config', type=str, default='config/default_config.yaml',
                       help='Path to configuration file (YAML format)')
    
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_loggers(config: dict) -> list:
    """Setup logging"""
    loggers = []
    
    model_type = config.get('model', {}).get('type', 'unet3d')
    experiment_name = config.get('experiment', {}).get('name') or f"{model_type}_experiment"
    
    # TensorBoard logger
    if config.get('experiment', {}).get('tensorboard', {}).get('enabled', True):
        tb_logger = TensorBoardLogger(
            save_dir=config['paths']['logs_dir'],
            name=experiment_name
        )
        loggers.append(tb_logger)
    
    # Weights & Biases logger
    if config.get('experiment', {}).get('wandb', {}).get('enabled', True):
        wandb_config = config['experiment']['wandb']
        wandb_logger = WandbLogger(
            project=wandb_config.get('project', 'tomopanda-pick'),
            entity=wandb_config.get('entity'),
            name=experiment_name,
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
    
    # Load configuration from YAML
    config = load_config(args.config)
    
    # Set random seed
    seed = config.get('training', {}).get('seed', 42)
    pl.seed_everything(seed)
    
    # Get model type from config
    model_type = config.get('model', {}).get('type', 'unet3d')
    
    # Create model
    model = ModelFactory.create_model(
        model_type=model_type,
        in_channels=config['model']['input_channels'],
        num_classes=config['model']['num_classes'],
        learning_rate=config['model']['learning_rate'],
        weight_decay=config['model']['weight_decay'],
        **config.get('model', {}).get('architecture', {})
    )
    
    # Get data settings from config
    data_config = config.get('data', {})
    training_config = config.get('training', {})
    subtomogram_config = data_config.get('subtomogram', {})
    
    # Create data loaders
    dataloaders = CryoETDataLoader.create_dataloaders(
        data_dir=data_config.get('data_dir', 'data'),
        batch_size=config['model']['batch_size'],
        num_workers=training_config.get('num_workers', 4),
        pin_memory=training_config.get('pin_memory', True),
        patch_size=tuple(config['data']['preprocessing']['patch_size']),
        overlap=config['data']['preprocessing']['overlap'],
        use_subtomograms=data_config.get('use_subtomograms', False),
        mask_type=subtomogram_config.get('mask_type', 'full'),
        mask_radius=subtomogram_config.get('mask_radius')
    )
    
    # Setup loggers
    loggers = setup_loggers(config)
    
    # Setup callbacks
    callbacks = setup_callbacks(config)
    
    # Setup trainer
    trainer = pl.Trainer(
        max_epochs=config['model']['num_epochs'],
        gpus=training_config.get('gpus', 1),
        precision=training_config.get('precision', 32),
        loggers=loggers,
        callbacks=callbacks,
        fast_dev_run=training_config.get('fast_dev_run', False),
        resume_from_checkpoint=training_config.get('resume_from_checkpoint')
    )
    
    # Start training
    print(f"Starting training with {model_type} model...")
    print(f"Configuration file: {args.config}")
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
