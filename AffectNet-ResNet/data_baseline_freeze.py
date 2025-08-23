from torch.utils import data
import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import DataLoader
import clip
from torch.utils.data import random_split


labels_map = {
    '0' : 'Anger',
    '1' : 'Contempt',
    '2' : 'Disgust',
    '3' : 'Fear',
    '4' : 'Happy',
    '5' : 'Neutral',
    '6' : 'Sad',
    '7' : 'Surprise'
}
labels_map_full = {
    0: "Neutral",
    1: "Happiness",
    2: "Sadness",
    3: "Surprise",
    4: "Fear",
    5: "Disgust",
    6: "Anger",
    7: "Contempt",
    8: "None",
    9: "Uncertain",
    10: "No-Face"
}

class AffectNetDatasetCLIP(data.Dataset):
    def __init__(self, img_dir, transform):
        self.img_dir = os.path.join(img_dir, "images")
        self.label_dir = os.path.join(img_dir, "annotations")
        self.transform = transform  # CLIP preprocess (expects PIL)
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

        files = []
        for f in os.listdir(self.img_dir):
            name, ext = os.path.splitext(f)
            if ext.lower() in exts:
                if os.path.isfile(os.path.join(self.label_dir, f"{name}_exp.npy")):
                    files.append(f)

        self.images = sorted(files)  # deterministic order

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        filename = self.images[idx]
        name = os.path.splitext(filename)[0]
        # image -> tensor via transform
        img_path = os.path.join(self.img_dir, filename)
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)  # now a tensor [C,H,W]

        # label
        label_path = os.path.join(self.label_dir, f"{name}_exp.npy")
        label = int(np.load(label_path))
        label = torch.tensor(label)  # convert to tensor

        return image, label
from torchvision import transforms
def get_data_loaders_clip(config, device):

    _, clip_preprocess = clip.load("ViT-B/16", device=device)

    # CLIP normalization (ViT-B/16)
    CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
    CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

    # ---- TRAIN: UA-FER-style DA(·) ----
    # random flip, rotation, random crop+scaling, blur, then normalize
    train_transform = transforms.Compose([
        # random cropping + scaling (keeps output 224×224)
        transforms.RandomResizedCrop(
            size=224,
            scale=(0.8, 1.0),
            interpolation=transforms.InterpolationMode.BICUBIC
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),  # small rotation for faces
        transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))],
            p=0.2  # light blur
        ),
        transforms.ToTensor(),
        transforms.Normalize(CLIP_MEAN, CLIP_STD),
    ])

    # ---- VAL/TEST: deterministic, no augmentation ----
    val_transform = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(CLIP_MEAN, CLIP_STD),
    ])

    train_dataset = AffectNetDatasetCLIP(config['train_full'], transform=train_transform)
    val_dataset = AffectNetDatasetCLIP(config['val_full'], transform=val_transform)

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

    train_loader = DataLoader(train_subset, batch_size=128, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_subset, batch_size=256, shuffle=False, num_workers=4)

    return train_loader, val_loader, []
