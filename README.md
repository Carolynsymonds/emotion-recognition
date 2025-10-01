# CLIP-FER: Improving Facial Expression Recognition via Feature Decoupling, Contrastive Prompts, and Evidential Uncertainty

This repository contains experiments for emotion recognition using CLIP backbone with different fusion strategies and uncertainty quantification methods.

## Experiments Overview

### Experiment 1: Baseline CLIP (train_baseline_freeze)
- **Backbone**: CLIP backbone with frozen visual encoder (only last transformer block trainable)
- **Data Source**: AffectNet dataset via `data_affectnet.py` and RAF-DB dataset via `data_raf_db.py`
- **Method**: Standard CLIP zero-shot classification with trainable last visual transformer block
- **Features**: Uses global CLS token features for classification

### Experiment 2: Multi-Feature Fusion (train_mfd)
- **Backbone**: CLIP backbone with frozen visual encoder (only last transformer block trainable)
- **Data Source**: AffectNet dataset via `data_affectnet.py` and RAF-DB dataset via `data_raf_db.py`
- **Method**: Multi-Feature Decoupling (MFD) approach that combines global and local patch features
- **Features**: Fuses global CLS token with top-K local patch affinities for enhanced classification

### Experiment 3: MFD + RUC Uncertainty Quantification (train_mfd_ruc)
- **Backbone**: CLIP backbone with frozen visual encoder (only last transformer block trainable)
- **Data Source**: AffectNet dataset via `data_affectnet.py`
- **Method**: Multi-Feature Decoupling (MFD) combined with Robust Uncertainty Quantification (RUC)
- **Features**: Extends MFD with evidence-based uncertainty estimation using Evidential Deep Learning

### Experiment 4: MFD + RUC with Per-Image Prompts (train_mfd_cl)
- **Backbone**: CLIP backbone with frozen visual encoder (only last transformer block trainable)
- **Data Source**: RAF-DB dataset via `data_features.py`
- **Method**: Multi-Feature Decoupling (MFD) combined with Robust Uncertainty Quantification (RUC) and per-image prompt optimization
- **Features**: Uses MFD+RUC approach with additional per-image prompt conditioning for improved performance

## Dataset Information

- **AffectNet**: Large-scale facial expression dataset with 7 emotion classes (Neutral, Happiness, Sadness, Surprise, Fear, Disgust, Anger)
- **RAF-DB**: Real-world Affective Faces Database with 7 emotion classes

### Dataset Access

**AffectNet Dataset**
The AffectNet dataset was obtained by request from Mohammad Mahoor (https://www.mohammadmahoor.com/pages/databases/affectnet/). Access requires contacting the dataset owner directly.

**RAF-DB Dataset**
The RAF-DB dataset was accessed via http://www.whdeng.cn/RAF/model1.html. To obtain access, a password must be requested by emailing Shan Li.

Both datasets include documentation to guide their use and interpretation.

## Key Features

- **Multi-Feature Decoupling (MFD)**: Combines global and local visual features for robust emotion recognition
- **Uncertainty Quantification**: Evidential Deep Learning approach for measuring prediction confidence
- **Per-Image Prompting**: Dynamic prompt generation for improved CLIP performance
- **Frozen CLIP Backbone**: Efficient training by only fine-tuning the last transformer block

## Setup Instructions

### 1. Data Setup

#### AffectNet Dataset
The AffectNet dataset should be organized in the following structure:

**For Local Development:**
```
data/
└── datasets/
    └── affectnet/
        ├── train_set/
        │   ├── images/           # Training images (.jpg files)
        │   └── annotations/      # Training labels (_exp.npy files)
        ├── val_set/
        │   ├── images/          # Validation images
        │   └── annotations/      # Validation labels
        └── test_set/
            ├── images/          # Test images
            └── annotations/     # Test labels
```

**For Cluster/Server (Hyperion):**
```
/mnt/scratch/users/[username]/affectnet_upload/
├── train_set/
│   ├── images/
│   └── annotations/
├── val_set/
│   ├── images/
│   └── annotations/
└── test_set/
    ├── images/
    └── annotations/
```

#### RAF-DB Dataset
The RAF-DB dataset should be organized as follows:

**For Local Development:**
```
data/
└── datasets/
    └── rafdb/
        ├── train_set/
        │   ├── 1/               # Class folders (1-7)
        │   ├── 2/
        │   └── ...
        ├── val_set/
        │   ├── 1/
        │   └── ...
        ├── train_labels.csv     # Training labels
        └── test_labels.csv      # Test labels
```

**For Cluster/Server (Hyperion):**
```
/mnt/scratch/users/[username]/DATASET/
├── train_set/
│   ├── 1/
│   ├── 2/
│   └── ...
├── val_set/
│   ├── 1/
│   └── ...
├── train_labels.csv
└── test_labels.csv
```

### 2. Configuration Setup

Update the `config.yaml` file with the appropriate paths for your environment:

**Local Configuration:**
```yaml
# AffectNet paths
train_full: "data/datasets/affectnet/train_set"
val_full: "data/datasets/affectnet/val_set"
test_full: "data/datasets/affectnet/test_set"

# RAF-DB paths
train_raf: "data/datasets/rafdb/train_set"
test_raf: "data/datasets/rafdb/val_set"
root: "data/datasets/rafdb"
```

**Cluster Configuration:**
```yaml
# AffectNet paths
train_full: "/mnt/scratch/users/[username]/affectnet_upload/train_set"
val_full: "/mnt/scratch/users/[username]/affectnet_upload/val_set"
test_full: "/mnt/scratch/users/[username]/affectnet_upload/test_set"

# RAF-DB paths
train_raf: "/mnt/scratch/users/[username]/DATASET/train_set"
test_raf: "/mnt/scratch/users/[username]/DATASET/val_set"
root: "/mnt/scratch/users/[username]/DATASET"
```

### 3. Running Experiments

#### Prerequisites
```bash
pip install -r requirements.txt
```

#### Experiment Execution
```bash
# Experiment 1: Baseline CLIP
python emotion_recognition/train/train_baseline_freeze.py

# Experiment 2: Multi-Feature Fusion
python emotion_recognition/train/train_mfd.py

# Experiment 3: MFD + RUC Uncertainty Quantification
python emotion_recognition/train/train_mfd_ruc.py

# Experiment 4: MFD + RUC with Per-Image Prompts
python emotion_recognition/train/train_mfd_cl.py
```

#### Using the Main Script
```bash
# Run all experiments with predefined configurations
cd emotion_recognition
bash main.sh
```

### 4. Dataset Selection

The experiments automatically select the appropriate dataset based on the configuration:

- **Experiments 1-3**: Use AffectNet dataset (via `data_baseline_freeze.py`)
- **Experiment 4**: Uses RAF-DB dataset (via `data_features.py`)

To switch datasets, modify the `dataset` field in `config.yaml`:
```yaml
dataset: "affectnet"  # or "rafdb"
```

## File Structure

```
emotion-recognition-1/
├── README.md                       # Project documentation
├── setup.py                       # Package installation script
├── requirements.txt               # Python dependencies
├── data/                          # Actual datasets directory
│   └── datasets/                  # Dataset storage
│       ├── affectnet/             # AffectNet dataset
│       └── rafdb/                 # RAF-DB dataset
├── src/                           # Source code package
│   ├── __init__.py
│   └── data/                      # Data loading code
│       ├── __init__.py
│       ├── data_affectnet.py      # AffectNet dataset loader
│       └── data_raf_db.py         # RAF-DB dataset loader
└── emotion_recognition/           # Main project package
    ├── __init__.py                # Package initialization
    ├── config.yaml               # Configuration file
    ├── main.sh                   # SLURM batch script
    ├── models.py                 # Model definitions
    ├── metrics.py                # Metrics logging
    ├── utils.py                  # Utility functions
    └── train/                    # Training scripts package
        ├── __init__.py
        ├── train_baseline_freeze.py  # Experiment 1: Baseline CLIP
        ├── train_mfd.py              # Experiment 2: MFD approach
        ├── train_mfd_ruc.py         # Experiment 3: MFD + RUC
        └── train_mfd_cl.py          # Experiment 4: MFD + RUC + per-image prompts
```

## Data Directory Structure

The project expects datasets to be organized in the following structure:

```
data/
├── datasets/                  # Dataset storage
│   ├── affectnet/             # AffectNet dataset
│   │   ├── train_set/
│   │   ├── val_set/
│   │   └── test_set/
│   └── rafdb/                 # RAF-DB dataset
│       ├── train_set/
│       └── val_set/
```

See the README files in each data subdirectory for detailed structure requirements.

## Installation

### Development Installation
```bash
# Clone the repository
git clone <repository-url>
cd emotion-recognition-1

# Install in development mode
pip install -e .

# Or install dependencies manually
pip install -r requirements.txt
```

### Package Structure Benefits
- **Modular Design**: Clear separation of data, models, training, and utilities
- **Proper Imports**: Python package structure with `__init__.py` files
- **Reusable Components**: Models and utilities can be imported and reused
- **Easy Testing**: Package structure enables proper unit testing
- **Professional Structure**: Follows Python packaging best practices
