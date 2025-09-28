from torch.utils import data
import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import DataLoader
import clip
from torch.utils.data import random_split


labels_map_full = {
    1: "Surprise",
    2: "Fear",
    3: "Disgust",
    4: "Happiness",
    5: "Sadness",
    6: "Anger",
    7: "Neutral",
}

import os, csv, re
from typing import Optional, Iterable
from PIL import Image
import torch
from torch.utils.data import Dataset

class RAFDBDatasetCLIP(Dataset):
    """
    RAF-DB loader that supports:
      • train/1..7/xxx.jpg  (folder-per-class)
      • CSV: <filename>,<label>  (labels can be 1-based; converted to 0-based)

    Returns: (image_tensor, label)  or (image_tensor, label, filename) if return_filename=True
    """
    def __init__(
            self,
            root: str,  # e.g. "data/DATASET"
            split: str,  # "train" or "test"
            transform=None,  # CLIP preprocess (expects PIL)
            csv_path: Optional[str] = None,
            label_base: int = 1,  # RAF-DB labels are usually 1..7
            to_zero_based: bool = True,  # convert to 0..6 if True
            exclude_labels: Optional[Iterable[int]] = None,
            return_filename: bool = False,
            strict: bool = True
        ):
        self.root = root
        self.split = split
        self.img_dir = os.path.join(root, split)
        self.transform = transform
        self.return_filename = return_filename
        self.label_base = label_base
        self.to_zero_based = to_zero_based
        self.exclude_labels = set([]) if exclude_labels is None else set(exclude_labels)

        if csv_path is None:
            csv_path = os.path.join(root, f"{split}_labels.csv")
        self.samples = self._load_from_csv(csv_path, strict)
        # deterministic order
        self.samples.sort(key=lambda x: x[0])

        print(f"[{split}] Loaded {len(self.samples)} samples from {os.path.basename(csv_path)}")

    def _is_image(self, fname: str) -> bool:
        return os.path.splitext(fname)[1].lower() in self.exts

    def _load_from_csv(self, csv_path: str, strict: bool):
        samples = []
        missing = 0

        def _clean(s: str) -> str:
            # strip BOM, spaces, CRLF; keep only basename for safety
            s = s.replace("\ufeff", "").strip()
            return os.path.basename(s)

        with open(csv_path, "r", newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    continue
                # skip header
                if re.match(r"(?i)\s*image", row[0]):
                    continue

                filename_raw = row[0]
                label_str = row[1]

                filename = _clean(filename_raw)
                label_raw = int(label_str.strip())  # RAF-DB CSV typically 1..7
                label_idx = label_raw - 1  # store 0..6 for training

                # ONLY allow class-subfolder path: <split>/<label_raw>/<filename>
                rel_path = os.path.join(str(label_raw).strip(), filename)
                full_path = os.path.normpath(os.path.join(self.img_dir, rel_path))

                if not os.path.isfile(full_path):
                    missing += 1
                    msg = (f"[{self.split}] Missing file for CSV row: "
                           f"filename={repr(filename_raw)}, label={repr(label_str)}\n"
                           f" -> tried: {full_path}\n"
                           f"   img_dir={self.img_dir}")
                    if strict:
                        raise FileNotFoundError(msg)
                    else:
                        print("[WARN]", msg)
                        continue

                samples.append((full_path, label_idx))

        print(f"[{self.split}] Loaded {len(samples)} samples from {os.path.basename(csv_path)}"
              + (f" | missing={missing}" if missing else ""))
        return samples

    def _load_from_folders(self):
        # expects train/<class_id>/image.jpg structure
        samples = []
        if not os.path.isdir(self.img_dir):
            return samples
        for cls_name in sorted(os.listdir(self.img_dir)):
            cls_path = os.path.join(self.img_dir, cls_name)
            if not os.path.isdir(cls_path):
                continue
            # numeric folder names (e.g., "1".."7")
            try:
                lbl = int(cls_name) - self.label_base
            except ValueError:
                # non-numeric folder name; skip
                continue
            if lbl in self.exclude_labels:
                continue
            for fname in sorted(os.listdir(cls_path)):
                if not self._is_image(fname):
                    continue
                rel_path = os.path.join(cls_name, fname)  # relative to split dir
                samples.append((rel_path, lbl))
        return samples

    def _load_samples(self, csv_path: Optional[str]):
        if csv_path and os.path.isfile(csv_path):
            return self._load_from_csv(csv_path)
        # Fallback to folder-per-class (common for train/)
        return self._load_from_folders()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        if self.return_filename:
            return image, torch.tensor(label, dtype=torch.long), os.path.basename(img_path)
        return image, torch.tensor(label, dtype=torch.long)

from torchvision import transforms
import matplotlib.pyplot as plt
from collections import Counter
import os


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

    print("labels")


    # Ensure consistent class order and include zero-count classes
    class_ids = sorted(labels_map.keys())
    classes = [labels_map[i] for i in class_ids]

    print("classes")

    values  = [counts.get(i, 0) for i in class_ids]
    print("values")


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

    print(f"Ready to save")

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

    root = config["root"]

    train_dataset = RAFDBDatasetCLIP(
        root=root,
        split="train",
        transform=train_transform,
        csv_path=os.path.join(root, "train_labels.csv"),  # or None to use train/1..7/
        label_base=1,
    )

    val_dataset = RAFDBDatasetCLIP(
        root=root,
        split="test",
        transform=val_transform,
        csv_path=os.path.join(root, "test_labels.csv"),  # or None to use train/1..7/
        label_base=1,
    )

    # 10% of train set and val set
    train_len = int(0.1 * len(train_dataset))
    val_len = int(0.1 * len(val_dataset))

    g = torch.Generator().manual_seed(42)
    _, train_subset = random_split(train_dataset, [len(train_dataset) - train_len, train_len], generator=g)
    _, val_subset = random_split(val_dataset, [len(val_dataset) - val_len, val_len], generator=g)

    print(f'train_set: {len(train_dataset)}')
    print(f'val_set: {len(val_dataset)}')

    print(f'train_set - % 10: {len(train_subset)}')
    print(f'val_set - % 10: {len(val_subset)}')

    print(f'training dataa! data')

    # plot_class_distribution(val_dataset, labels_map_full, title="Val Class Distribution", save_dir=None,
    #         filename=None,
    #         annotate_pct=True)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)
    val_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4)


    return train_loader, val_loader, []
