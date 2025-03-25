import argparse
import yaml
import time
import requests

import numpy as np
import torch
import torch.nn as nn
from utils.Data import Data
from PIL import Image
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
import lightning as pl
from lightning.pytorch import Trainer


# Device Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ResNetLightning(pl.LightningModule):
    def __init__(self, num_classes):
        super(ResNetLightning, self).__init__()
        self.model = models.resnet18(pretrained=True)

        # Modify classifier layer
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

        # Define loss function
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)

        # Log loss and accuracy
        acc = (outputs.argmax(dim=1) == labels).float().mean()
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)

        acc = (outputs.argmax(dim=1) == labels).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

    

if __name__ == "__main__":
    data_dir = "data/training"
    data = Data(data_dir, 16, (0.7, 0.15, 0.15))
    train_loader, val_loader, test_loader, class_to_idx = data.get_dataloaders()
    
    
    # Instantiate the model
    num_classes = class_to_idx
    model = ResNetLightning(num_classes)
    
    # Define the trainer
    trainer = Trainer(
        max_epochs=10,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        log_every_n_steps=10
    )
    
    # Train the model
    trainer.fit(model, train_loader, val_loader)
    
    trainer.validate(model, test_loader)

    # Save model
    trainer.save_checkpoint("resnet_model.ckpt")

    # Load model
    model = ResNetLightning.load_from_checkpoint("resnet_model.ckpt", num_classes=num_classes)
    model.eval()


