import argparse
import yaml
import time
import requests
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from utils.Data import Data
from utils.GestureDataModule import GestureDataModule
from utils.GestureResNet import GestureResNet
from utils.GestureSwinV2 import GestureSwinV2

def predict(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            images, labels = batch

            outputs = model.forward(images.to(DEVICE))
            preds = outputs.argmax(dim=1)
            
            # Collect predictions and true labels
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    accuracy = np.mean(all_preds == all_labels)
    print(f'ACCURACY: {accuracy}')

    return np.array(all_preds), np.array(all_labels)


def plot_confusion_matrix(preds, labels):
    cm = confusion_matrix(labels, preds)
    cm_percent = cm.astype('float') / cm.sum(axis=1, keepdims=True)  # Convert to percentages
    
    plt.figure(figsize=(18, 12))
    sns.heatmap(cm_percent, annot=True, fmt='.2f', cmap='Blues', cbar=True)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix (Percentage Heatmap)')
    plt.savefig("confusion_big.png")
    plt.close()



transform = transforms.Compose([
    #transforms.Resize((112, 112)),  # Resize to ResNet input size
    #transforms.RandomHorizontalFlip(),
    #transforms.RandomRotation(20),
    transforms.ToTensor(),          # Convert to PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize like ImageNet
])

# Device Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    
    torch.set_num_threads(8)
    torch.cuda.empty_cache()

    data_dir = "data_pruned/"

    logger = TensorBoardLogger("logs", name="resnet-no-leak")

    #data_module = GestureDataModule(data_dir=data_dir, batch_size=16, splits=(0.7, 0.15, 0.15))
    
    #num_classes = len(data_module.data_obj.get_dataloaders()[3])
    #class_weights = data_module.data_obj.get_dataloaders()[4]
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)

    # Define split sizes (80% train, 10% val, 10% test)
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size  # Ensure total length matches
    
    # Split dataset
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
    
    class_counts = Counter([label for _, label in dataset.samples])
    num_classes = len(class_counts)
    # Compute inverse frequency weights
    class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
    class_weights = torch.tensor([class_weights[i] for i in range(num_classes)], dtype=torch.float32)



    model = GestureResNet(num_classes)
    #model.set_weights(class_weights)

    trainer = pl.Trainer(
        max_epochs=200,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=logger
    )

    trainer.fit(model, train_loader, val_loader)

    trainer.test(model, test_loader)

    checkpoint_path = "gesture_model_no_leak.ckpt"
    trainer.save_checkpoint(checkpoint_path)
    print(f"Model saved to {checkpoint_path}")

