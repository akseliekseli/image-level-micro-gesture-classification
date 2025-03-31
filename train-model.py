import argparse
import random
import json
import os
import numpy as np

import torch
import torch.nn as nn
from torchvision import transforms
import pytorch_lightning as pl
from lightning.pytorch.loggers import TensorBoardLogger
from sklearn.metrics import confusion_matrix
import torch.multiprocessing as mp

import seaborn as sns
import matplotlib.pyplot as plt

from utils.GestureSequenceDataset import get_dataloaders
from utils.GestureResNet3D import GestureResNet3D
from utils.GestureSwin3D import GestureSwin3D
from utils.GestureSwin3DWithPose import GestureSwin3DWithPose
from utils.VitPoseKeypoints import generate_keypoints 

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
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    output_file = "keypoints.json"
    # Generating keypoints to use in the pose model
    if args.model == 'pose':
        generate_keypoints(data_dir, output_file)
    logger = TensorBoardLogger("logs", name=args.model)

    if args.model == 'pose': keypoints = 'keypoints.json'
    else: keypoints = None
    train_loader, val_loader, test_loader, num_classes = get_dataloaders("data/training", keypoints, batch_size=16, transform=transform)

    print(f'NUM CLASSES {num_classes}')
    
    model_path = "models/"+args.model+".ckpt"

    if args.model == 'resnet': model = GestureResNet3D(num_classes)
    elif args.model == 'swin': model = GestureSwin3D(num_classes)
    elif args.model == 'pose': model = GestureSwin3DWithPose(num_classes)   

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
    elif args.model == 'pose': model = GestureSwin3DWithPose(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))["state_dict"], strict=False)
    model.to(DEVICE)
    if args.model == 'pose':    predictions, ground_truth = predict_pose(model, test_loader)
    else:   predictions, ground_truth = predict(model, test_loader)
    
    plot_confusion_matrix(predictions, ground_truth, args.model)

