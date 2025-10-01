# AffectNet Dataset Structure

This directory should contain the AffectNet dataset organized as follows:

```
affectnet/
├── train_set/
│   ├── images/           # Training images (.jpg files)
│   │   ├── 0000001.jpg
│   │   ├── 0000002.jpg
│   │   └── ...
│   └── annotations/      # Training labels (_exp.npy files)
│       ├── 0000001_exp.npy
│       ├── 0000002_exp.npy
│       └── ...
├── val_set/
│   ├── images/          # Validation images
│   └── annotations/     # Validation labels
└── test_set/
    ├── images/          # Test images
    └── annotations/     # Test labels
```

## Label Mapping
- 0: Neutral
- 1: Happiness
- 2: Sadness
- 3: Surprise
- 4: Fear
- 5: Disgust
- 6: Anger

## Usage
Update the `config.yaml` file with the correct path:
```yaml
train_full: "data/datasets/affectnet/train_set"
val_full: "data/datasets/affectnet/val_set"
test_full: "data/datasets/affectnet/test_set"
```
