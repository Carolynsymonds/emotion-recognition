"""
Emotion Recognition with CLIP and Multi-Feature Fusion

This package contains experiments for emotion recognition using CLIP backbone 
with different fusion strategies and uncertainty quantification methods.

Experiments:
- Experiment 1: Baseline CLIP (train_baseline_freeze)
- Experiment 2: Multi-Feature Fusion (train_mfd) 
- Experiment 3: MFD + RUC Uncertainty Quantification (train_mfd_ruc)
- Experiment 4: MFD + RUC with Per-Image Prompts (train_mfd_cl)
"""

__version__ = "1.0.0"
__author__ = "Carolyn Symonds"

from . import models
from . import metrics
from . import utils

__all__ = [
    'models',
    'metrics',
    'utils'
]
