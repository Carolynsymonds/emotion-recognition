from torch.utils import data
import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import DataLoader
import clip
from torch.utils.data import random_split

from torchvision import transforms
import matplotlib.pyplot as plt
from collections import Counter
import os

labels_map_full = {
    0: "Neutral",
    1: "Happiness",
    2: "Sadness",
    3: "Surprise",
    4: "Fear",
    5: "Disgust",
    6: "Anger",
    # 7: "Contempt",
    # 8: "None",
    # 9: "Uncertain",
    # 10: "No-Face"
}

class AffectNetDatasetCLIP(data.Dataset):
    def __init__(self, img_dir, transform, exclude_labels = None):
        self.img_dir = os.path.join(img_dir, "images")
        self.label_dir = os.path.join(img_dir, "annotations")
        self.transform = transform  # CLIP preprocess (expects PIL)
        self.exclude_labels = set([]) if exclude_labels is None else set(exclude_labels)
        self.exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        samples = []

        for f in os.listdir(self.img_dir):
            name, ext = os.path.splitext(f)
            if ext.lower() not in exts:
                continue
            lbl_path = os.path.join(self.label_dir, f"{name}_exp.npy")
            if not os.path.isfile(lbl_path):
                continue

            lbl = int(np.load(lbl_path))
            if lbl in self.exclude_labels:
                continue

            samples.append((f, lbl))

        # deterministic order
        samples.sort(key=lambda x: x[0])
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filename, label = self.samples[idx]
        img_path = os.path.join(self.img_dir, filename)
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.tensor(label)



def _to_int(x):
    # Safely convert tensor/np/int to Python int
    try:
        return int(x.item())  # torch / numpy scalar
    except Exception:
        return int(x)

def plot_class_distribution(
    dataset,
    labels_map,
    title="Class Distribution",
    save_dir=None,
    filename=None,
    annotate_pct=True
):

    # Collect labels as ints
    labels = [_to_int(dataset[i][1]) for i in range(len(dataset))]
    counts = Counter(labels)

    # Ensure consistent class order and include zero-count classes
    class_ids = sorted(labels_map.keys())
    classes = [labels_map[i] for i in class_ids]
    values = [counts.get(i, 0) for i in class_ids]


    # Plot
    plt.figure(figsize=(11, 5))
    bars = plt.bar(classes, values)
    plt.xticks(rotation=45, ha="right")
    plt.title(title)
    plt.ylabel("Number of Images")
    plt.tight_layout()

    # Optional % annotations
    if annotate_pct:
        total = sum(values) if sum(values) > 0 else 1
        for rect, v in zip(bars, values):
            pct = 100.0 * v / total
            plt.text(
                rect.get_x() + rect.get_width()/2.0,
                rect.get_height(),
                f"{pct:.1f}%",
                ha="center", va="bottom", fontsize=9
            )


    # Save or show
    if save_dir and filename:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, filename)
        plt.savefig(path, dpi=300)
        print(f"Saved plot to {path}")
        plt.close()
    else:
        plt.show()

def get_data_loaders_clip(config, device):

    _, clip_preprocess = clip.load("ViT-B/16", device=device)

    # CLIP normalization (ViT-B/16)
    CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
    CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

    # ---- TRAIN: UA-FER-style DA(·) ----
    # random flip, rotation, random crop+scaling, blur, then normalize
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.85, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.02),
        transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 2.0))], p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(CLIP_MEAN, CLIP_STD),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.08), ratio=(0.3, 3.3), value='random'),
    ])

    # ---- VAL/TEST: deterministic, no augmentation ----
    val_transform = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(CLIP_MEAN, CLIP_STD),
    ])

    # Load training dataset
    train_dataset = AffectNetDatasetCLIP(config['train_full'], transform=train_transform, exclude_labels={7})
    val_dataset = AffectNetDatasetCLIP(config['val_full'], transform=val_transform, exclude_labels={7})

    # Use 10% of train set and val set for faster experimentation
    train_len = int(0.1 * len(train_dataset))
    val_len = int(0.1 * len(val_dataset))

    g = torch.Generator().manual_seed(42)
    _, train_subset = random_split(train_dataset, [len(train_dataset) - train_len, train_len], generator=g)
    _, val_subset = random_split(val_dataset, [len(val_dataset) - val_len, val_len], generator=g)

    print(f'Full train_set: {len(train_dataset)}')
    print(f'Full val_set: {len(val_dataset)}')
    print(f'Training subset (10%): {len(train_subset)}')
    print(f'Validation subset (10%): {len(val_subset)}')

    train_loader = DataLoader(train_subset, batch_size=config.get('batch_size', 256), shuffle=True, num_workers=4)
    val_loader = DataLoader(val_subset, batch_size=config.get('batch_size', 256), shuffle=False, num_workers=4)

    return train_loader, val_loader, []
