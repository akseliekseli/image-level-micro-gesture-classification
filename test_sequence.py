import argparse
import random
import torch
import pytorch_lightning as pl
from torchvision import models
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from utils.GestureResNet import GestureResNet
from utils.GestureResNet3D import GestureResNet3D
from utils.GestureSequenceDataset import get_dataloaders
from utils.GestureSwin3DWithPose import GestureSwin3DWithPose

from collections import Counter

transform = transforms.Compose([
    transforms.Resize((112, 112)),  # Resize to ResNet input size
    #transforms.RandomHorizontalFlip(),
    #transforms.RandomRotation(20),
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
            print('ii')
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


def predict_pose(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            images, keypoints, labels = batch
            outputs = model.forward((images.to(DEVICE), keypoints.to(DEVICE)))
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

def plot_confusion_matrix(preds, labels):
    cm = confusion_matrix(labels, preds)
    cm_percent = cm.astype('float') / cm.sum(axis=1, keepdims=True)  # Convert to percentages
    
    plt.figure(figsize=(18, 12))
    sns.heatmap(cm_percent, annot=True, fmt='.2f', cmap='Blues', cbar=True)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix (Percentage Heatmap)')
    plt.savefig("confusion_sequence.png")
    plt.close()

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
# Device Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Testing the Micro-Gesture model")
    parser.add_argument('--model', type=str, required=True, help='model to test')
    args = parser.parse_args()
    
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    torch.set_num_threads(8)
    torch.cuda.empty_cache()
    
    data_dir = "data/training"
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

    train_loader, val_loader, test_loader = get_dataloaders("data/training", 'keypoints_112.json', batch_size=16, transform=transform)
    print(f'NUM CLASSES {num_classes}')
    model = GestureResNet3D(num_classes)
    
    model_path = "models/"+args.model+"112.ckpt"

    if args.model == 'resnet': model = GestureResNet3D(num_classes)
    elif args.model == 'swin': model = GestureSwin3D(num_classes)
    elif args.model == 'pose': model = GestureSwin3DWithPose(num_classes)
    
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))["state_dict"], strict=False)
    model.to(DEVICE)
    if args.model == 'pose': predictions, ground_truth = predict_pose(model, test_loader)
    else: predictions, ground_truth = predict(model, test_loader)
    
    plot_confusion_matrix(predictions, ground_truth)

    # Print predictions and ground truth (or handle them as needed)
    print(f"Predictions: {predictions}")
    print(f"Ground Truth: {ground_truth}")

