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
import clip

class EmotionDataset(data.Dataset):
    def __init__(self, df, dataset_dir, label_map):
        self.df = df
        self.image_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4815, 0.4578, 0.4082), std=(0.2686, 0.2613, 0.2758))
        ])
        self.images = []
        self.labels = []
        self.dataset_dir = dataset_dir
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.preprocess = clip.load("ViT-B/32", device=device)

        for label, idx in tqdm(label_map.items()):
            folder_path = os.path.join(dataset_dir, str(idx))  # +1 because folder names start from '1'
            for filename in os.listdir(folder_path):
                img_path = os.path.join(folder_path, filename)
                img = cv2.imread(img_path)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                self.images.append(img_rgb)
                self.labels.append(idx)


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Get the image filename and label from the dataframe
        row = self.df.iloc[idx]
        img_path = os.path.join(self.dataset_dir, str(row['label']), row['image'])

        # Open the image file
        image = Image.open(img_path).convert('RGB')
        label = row['label']

        if self.image_transform:
            image = self.image_transform(image)

        # image = self.preprocess(image)  # THIS is important

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

import matplotlib.pyplot as plt
import torchvision.transforms.functional as F

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

def unnormalize(tensor, mean, std):
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    return tensor * std + mean
def show_examples(dataset, classes, num_examples=5):
    num_classes = len(classes)
    fig, axs = plt.subplots(num_classes, num_examples, figsize=(12, 14))

    for i, class_name in enumerate(classes):
        # Find indices for the current class (label values start at 1)
        class_indices = [idx for idx, (_, label) in enumerate(dataset) if label == i + 1]

        # Randomly choose a few examples from this class
        selected_indices = np.random.choice(class_indices, num_examples, replace=False)

        for j, idx in enumerate(selected_indices):
            img_tensor, _ = dataset[idx]
            if hasattr(img_tensor, 'permute'):  # tensor check
                img_tensor = unnormalize(img_tensor.clone(), mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                img_tensor = img_tensor.permute(1, 2, 0).numpy()

            axs[i, j].imshow(img_tensor)
            axs[i, j].axis('off')
            if j == 0:
                axs[i, j].set_title(class_name, fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.show()

def get_data():
    # Load the labels CSV files

    # GET Data
    train_labels = pd.read_csv('./dataset/train_labels.csv')
    test_labels = pd.read_csv('./dataset/test_labels.csv')

    #Labels
    classes = ['surprise', 'fear', 'disgust', 'happy', 'sad', 'angry', 'neutral']
    label_map = {label: (idx + 1) for idx, label in enumerate(classes)}

    # GET DataSet
    dataset_train = EmotionDataset(train_labels, "./dataset/train", label_map)
    dataset_test = EmotionDataset(test_labels, "./dataset/test", label_map)

    # TEST images
    # view_image(dataset_train, 60, classes)

    dataset_total = dataset_train + dataset_test

    dataloader = DataLoader(dataset_total, batch_size=8, shuffle=True)

    return dataloader

def distribution(classes, label_map, test_label_counts, train_label_counts):
    inv_label_map = {v: k for k, v in label_map.items()}
    train_label_counts_named = {inv_label_map[k]: v for k, v in train_label_counts.items()}
    test_label_counts_named = {inv_label_map[k]: v for k, v in test_label_counts.items()}
    train_counts = [train_label_counts_named.get(cls, 0) for cls in classes]
    test_counts = [test_label_counts_named.get(cls, 0) for cls in classes]
    # Totals
    total_train = sum(train_counts)
    total_test = sum(test_counts)
    # Percentages
    train_percentages = [(count / total_train) * 100 for count in train_counts]
    test_percentages = [(count / total_test) * 100 for count in test_counts]
    # Plot
    plt.figure(figsize=(10, 6))
    x = range(len(classes))
    bar_width = 0.35
    plt.bar(x, train_counts, width=bar_width, label="Train", alpha=0.7, color="cornflowerblue")
    plt.bar([p + bar_width for p in x], test_counts, width=bar_width, label="Test", alpha=0.7, color="crimson")
    # Annotate bars
    for i, (train_count, test_count) in enumerate(zip(train_counts, test_counts)):
        plt.text(i, train_count + 2, f"{train_percentages[i]:.1f}%", ha='center', color="blue", fontsize=9)
        plt.text(i + bar_width, test_count + 2, f"{test_percentages[i]:.1f}%", ha='center', color="red", fontsize=9)
    # Labels and layout
    plt.xticks([p + bar_width / 2 for p in x], classes, rotation=45)
    plt.xlabel("Emotion Class")
    plt.ylabel("Number of Examples")
    plt.title("Distribution of Emotion Classes in Train and Test Sets (%)")
    plt.legend()
    plt.tight_layout()
    plt.show()

