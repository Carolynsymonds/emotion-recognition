from torch.utils import data
from torchvision import transforms
from tqdm.notebook import tqdm
import os
import cv2
from PIL import Image
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
import seaborn as sns
import clip

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

class AffectNetDataset(data.Dataset):
    def __init__(self, img_dir):
        self.img_dir = img_dir + '/images'
        self.label_dir = img_dir + '/labels'
        self.transform_train = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
        ])
        self.transform_validation = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
        ])
        self.images = []
        for filename in os.listdir(self.img_dir):
            self.images.append(filename)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        image = Image.open(img_path).convert('RGB')
        if 'train' in self.img_dir:
            image = self.transform_train(image)
        else:
            image = self.transform_validation(image)

        file_base = os.path.splitext(self.images[idx])[0]
        label_path = os.path.join(self.label_dir, file_base + '.txt')
        with open(label_path, 'r') as f:
            # first character is the label
            target = f.read(1)

        label = int(target)  # convert string to int
        label = torch.tensor(label)  # convert to tensor
        return image, label


class AffectNetDatasetCLIP(data.Dataset):
    def __init__(self, img_dir, transform):
        self.img_dir = img_dir + '/images'
        self.label_dir = img_dir + '/labels'
        self.transform = transform  # Use CLIP's preprocess here
        self.images = [filename for filename in os.listdir(self.img_dir)]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)  # Use CLIP's transform

        file_base = os.path.splitext(self.images[idx])[0]
        label_path = os.path.join(self.label_dir, file_base + '.txt')
        with open(label_path, 'r') as f:
            target = f.read(1)
        label = int(target)
        label = torch.tensor(label)
        return image, label


# Function to load images and labels from the directory
def load_data(dataset_dir, label_map):
    images = []
    labels = []

    for label, idx in tqdm(label_map.items()):
        folder_path = os.path.join(dataset_dir, str(idx))  # +1 because folder names start from '1'
        for filename in os.listdir(folder_path):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img_rgb)
            labels.append(idx)

    return np.array(images), np.array(labels)

def view_image(dataset, idx, classes):
    """
    Display an image and its class label from a PyTorch dataset.

    Args:
        dataset: A PyTorch Dataset object returning (image_tensor, label).
        idx: Index of the image to display.
        classes: List of class names where classes[label - 1] maps to the label.
    """
    # Get the image tensor and label
    img, label = dataset[idx]

    # Convert PIL to tensor if needed
    if isinstance(img, torch.Tensor):
        img_tensor = img
    else:
        img_tensor = F.to_tensor(img)

        # Convert from CxHxW to HxWxC
    img_np = img_tensor.permute(1, 2, 0).cpu().numpy()

    # Normalize to [0,1] for display
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())

    # Display the image
    plt.figure(figsize=(4, 4))
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Class: {classes[label - 1]}")
    plt.show()

def get_data_loaders(config):

    train_dataset = AffectNetDataset(config['train'])
    val_dataset = AffectNetDataset(config['val'])
    # test_dataset = AffectNetDataset(config['test'])

    # VIEW IMAGE
    # Show 20 images from the dataset
    images, labels = zip(*[train_dataset[i] for i in range(20)])
    show_images_grid(images, labels, labels_map, n_cols=5)

    print(f'train_set: {len(train_dataset)}')
    print(f'val_set: {len(val_dataset)}')
    # print(f'test_set: {len(test_dataset)}')

    # START EDA
    # Step 1: Extract labels from the dataset
    labels = []
    for _, label in train_dataset:
        labels.append(label.item() if hasattr(label, 'item') else label)

    # Step 2: Map numeric labels to expression class names
    class_map = {
        0: 'Neutral',
        1: 'Happy',
        2: 'Sad',
        3: 'Surprise',
        4: 'Fear',
        5: 'Disgust',
        6: 'Anger',
        7: 'Contempt'
    }
    label_names = pd.Series(labels).map(class_map)

    # Step 3: Create distribution table
    distribution = label_names.value_counts().reset_index()
    distribution.columns = ['Expression Class', 'Count']
    distribution['Proportion (%)'] = (distribution['Count'] / distribution['Count'].sum() * 100).round(2)
    print(distribution)

    # Step 4: Plot using Seaborn
    plt.figure(figsize=(10, 6))
    sns.barplot(data=distribution, x='Expression Class', y='Count')
    plt.title('Training Set Class Distribution')
    plt.xlabel('Emotion')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    # END EDA

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4)
    # test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    return train_loader, val_loader, []

def get_data_loaders_clip(config, device):

    _, clip_preprocess = clip.load("ViT-B/32", device=device)  # or "cuda"

    train_dataset = AffectNetDatasetCLIP(config['train'], transform=clip_preprocess)
    val_dataset = AffectNetDatasetCLIP(config['val'], transform=clip_preprocess)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4)
    return train_loader, val_loader, []

def show_images_grid(images, labels, labels_map, n_cols=5, figsize=(15, 10)):
    n_images = len(images)
    n_rows = (n_images + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

    for i, ax in enumerate(axes.flat):
        if i < n_images:
            img = images[i].permute(1, 2, 0).cpu().numpy()
            img = (img - img.min()) / (img.max() - img.min())  # normalize to [0, 1]

            ax.imshow(img)
            label = labels[i]
            label_str = labels_map[str(label)] if isinstance(label, int) or isinstance(label, str) else labels_map[str(label.item())]
            ax.set_title(f"Label: {label_str}")
            ax.axis('off')
        else:
            ax.axis('off')  # Hide empty subplots

    plt.tight_layout()
    plt.show()


