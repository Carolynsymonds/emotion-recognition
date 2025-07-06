import torch
import matplotlib.pyplot as plt
import os
import yaml
import json
import numpy as np

def setup_device():
    # Set device to GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True
    return device

def load_config(config_path):
    # Load YAML configuration file
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def plot_metrics(metrics_history, output_dir):
    # Create directory for metrics plots
    metrics_dir = os.path.join(output_dir, 'metrics')
    os.makedirs(metrics_dir, exist_ok=True)

    # Plot loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(metrics_history['train_loss'], label='Training Loss')
    plt.plot(metrics_history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(metrics_dir, 'loss_curves.png'))
    plt.close()

    # Plot loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(metrics_history['accuracy'], label='Training Accuracy')
    plt.plot(metrics_history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(metrics_dir, 'accuracy_curves.png'))
    plt.close()


    # Save metrics to JSON for later analysis
    with open(os.path.join(metrics_dir, 'metrics_history.json'), 'w') as f:
        json.dump(metrics_history, f, indent=4)
