# RAF-DB Dataset Structure

This directory should contain the RAF-DB dataset organized as follows:

```
rafdb/
├── train_set/
│   ├── 1/               # Surprise class
│   │   ├── image1.jpg
│   │   └── ...
│   ├── 2/               # Fear class
│   ├── 3/               # Disgust class
│   ├── 4/               # Happiness class
│   ├── 5/               # Sadness class
│   ├── 6/               # Anger class
│   └── 7/               # Neutral class
├── val_set/
│   ├── 1/               # Surprise class
│   ├── 2/               # Fear class
│   ├── 3/               # Disgust class
│   ├── 4/               # Happiness class
│   ├── 5/               # Sadness class
│   ├── 6/               # Anger class
│   └── 7/               # Neutral class
├── train_labels.csv     # Training labels (optional)
└── test_labels.csv      # Test labels (optional)
```

## Label Mapping
- 1: Surprise
- 2: Fear
- 3: Disgust
- 4: Happiness
- 5: Sadness
- 6: Anger
- 7: Neutral

## Usage
Update the `config.yaml` file with the correct path:
```yaml
root: "data/datasets/rafdb"
train_raf: "data/datasets/rafdb/train_set"
test_raf: "data/datasets/rafdb/val_set"
```
