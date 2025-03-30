
import argparse
import yaml
import time
from collections import Counter
import random

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

from utils.GestureSequenceDataset import get_dataloaders
from utils.GestureResNet3D import GestureResNet3D
from utils.GestureSwin3D import GestureSwin3D
from utils.GestureSwin3DWithPose import GestureSwin3DWithPose

transform = transforms.Compose([
    transforms.Resize((112, 112)),  # Resize to ResNet input size
    transforms.ToTensor(),          # Convert to PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize like ImageNet
])
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

            torch.cuda.empty_cache()
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    accuracy = np.mean(all_preds == all_labels)
    print(f'ACCURACY: {accuracy}')

    return np.array(all_preds), np.array(all_labels)



def plot_confusion_matrix(preds, labels, name):
    cm = confusion_matrix(labels, preds)
    cm_percent = cm.astype('float') / cm.sum(axis=1, keepdims=True)  # Convert to percentages
    
    plt.figure(figsize=(18, 12))
    sns.heatmap(cm_percent, annot=True, fmt='.2f', cmap='Blues', cbar=True)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix {name} 3D')
    plt.savefig(f"plots/confusion_sequence_{name}.png")
    plt.close()


import torch.multiprocessing as mp

# Device Configuration
#DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    parser = argparse.ArgumentParser(description="Testing the Micro-Gesture model")
    parser.add_argument('--model', type=str, required=True, help='model to test')
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--epochs', type=int, required=True)
    args = parser.parse_args()
    
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    torch.set_num_threads(8)
    torch.cuda.empty_cache()

    data_dir = "data/training"

    logger = TensorBoardLogger("logs", name=args.model)

    
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)

    # Define split sizes (80% train, 10% val, 10% test)
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size  # Ensure total length matches
    
    # Split dataset
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    
    
    class_counts = Counter([label for _, label in dataset.samples])
    num_classes = len(class_counts)
    # Compute inverse frequency weights
    class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
    class_weights = torch.tensor([class_weights[i] for i in range(num_classes)], dtype=torch.float32)

    train_loader, val_loader, test_loader = get_dataloaders("data/training", batch_size=8, transform=transform)
    print(f'NUM CLASSES {num_classes}')

    if args.model == 'resnet': model = GestureResNet3D(num_classes)
    elif args.model == 'swin': model = GestureSwin3D(num_classes)
    elif args.model == 'pose': model = GestureSwin3DWithPose(num_classes)
    #model.set_weights(class_weights)

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=[args.gpu],
        logger=logger
    )

    trainer.fit(model, train_loader, val_loader)

    trainer.test(model, test_loader)

    model_path = "models/"+args.model+".ckpt"
    trainer.save_checkpoint(model_path)
    print(f"Model saved to {model_path}")
    
    if args.model == 'resnet': model = GestureResNet3D(num_classes)
    elif args.model == 'swin': model = GestureSwin3D(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))["state_dict"], strict=False)
    model.to(DEVICE)
    predictions, ground_truth = predict(model, test_loader)
    
    plot_confusion_matrix(predictions, ground_truth, args.model)

