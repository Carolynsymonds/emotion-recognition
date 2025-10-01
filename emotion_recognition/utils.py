"""
Utility functions for emotion recognition experiments.
This module provides common utilities for configuration, device setup, and visualization.
"""

import yaml
import torch
import matplotlib.pyplot as plt
import os
from typing import Dict, Any, Optional


def load_config(config_path: str = 'config.yaml') -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dictionary containing configuration parameters
    """
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        print(f"Configuration loaded from {config_path}")
        return config
    except FileNotFoundError:
        print(f"Configuration file {config_path} not found. Using default config.")
        return get_default_config()
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        return get_default_config()


def get_default_config() -> Dict[str, Any]:
    """Return default configuration parameters."""
    return {
        'dataset': 'affectnet',
        'batch_size': 128,
        'num_epochs': 30,
        'learning_rate': 0.0001,
        'weight_decay': 0.05,
        'num_classes': 7,
        'image_size': 224,
        'num_workers': 4,
        'checkpoint_dir': 'checkpoints',
        'output_dir': 'outputs'
    }


def setup_device() -> torch.device:
    """
    Setup and return the appropriate device for training.
    
    Returns:
        PyTorch device (cuda if available, otherwise cpu)
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    return device


def plot_metrics(metrics_history: Dict[str, list], save_dir: Optional[str] = None) -> None:
    """
    Plot training metrics over epochs.
    
    Args:
        metrics_history: Dictionary containing metric names and their values over epochs
        save_dir: Directory to save plots (optional)
    """
    if not metrics_history:
        print("No metrics to plot")
        return
    
    # Create subplots for different metric types
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    metric_types = {
        'loss': ['train_loss', 'val_loss'],
        'accuracy': ['train_acc', 'val_acc'],
        'f1': ['train_f1', 'val_f1'],
        'other': []
    }
    
    # Categorize metrics
    for key in metrics_history.keys():
        categorized = False
        for category, patterns in metric_types.items():
            if any(pattern in key.lower() for pattern in patterns):
                if category != 'other':
                    metric_types[category].append(key)
                categorized = True
                break
        if not categorized:
            metric_types['other'].append(key)
    
    # Plot each category
    plot_idx = 0
    for category, metrics in metric_types.items():
        if metrics and plot_idx < len(axes):
            ax = axes[plot_idx]
            for metric in metrics:
                if metric in metrics_history:
                    ax.plot(metrics_history[metric], label=metric, marker='o')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Value')
            ax.set_title(f'{category.title()} Metrics')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plot_idx += 1
    
    # Hide unused subplots
    for i in range(plot_idx, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'training_metrics.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Metrics plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, 
                   epoch: int, loss: float, metrics: Dict[str, Any], 
                   filepath: str) -> None:
    """
    Save model checkpoint.
    
    Args:
        model: PyTorch model to save
        optimizer: Optimizer state
        epoch: Current epoch number
        loss: Current loss value
        metrics: Dictionary of metrics
        filepath: Path to save checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'metrics': metrics
    }
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(filepath: str, model: torch.nn.Module, 
                   optimizer: Optional[torch.optim.Optimizer] = None) -> Dict[str, Any]:
    """
    Load model checkpoint.
    
    Args:
        filepath: Path to checkpoint file
        model: PyTorch model to load state into
        optimizer: Optimizer to load state into (optional)
        
    Returns:
        Dictionary containing checkpoint information
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint file {filepath} not found")
    
    checkpoint = torch.load(filepath, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"Checkpoint loaded from {filepath}")
    return checkpoint


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration parameters.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        True if configuration is valid, False otherwise
    """
    required_keys = ['dataset', 'batch_size', 'num_epochs', 'learning_rate']
    
    for key in required_keys:
        if key not in config:
            print(f"Missing required configuration key: {key}")
            return False
    
    # Validate dataset paths
    if config['dataset'] == 'affectnet':
        required_paths = ['train_full', 'val_full']
    elif config['dataset'] == 'rafdb':
        required_paths = ['root']
    else:
        print(f"Unknown dataset: {config['dataset']}")
        return False
    
    for path_key in required_paths:
        if path_key not in config:
            print(f"Missing required path for {config['dataset']}: {path_key}")
            return False
    
    return True


def create_directories(config: Dict[str, Any]) -> None:
    """
    Create necessary directories for the experiment.
    
    Args:
        config: Configuration dictionary
    """
    directories = [
        config.get('checkpoint_dir', 'checkpoints'),
        config.get('output_dir', 'outputs'),
        config.get('output_dir', 'outputs') + '/visualizations'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")