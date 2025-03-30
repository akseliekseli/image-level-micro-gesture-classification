import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchmetrics
from torchvision.models.video import r3d_18, R3D_18_Weights
from lightning.pytorch.loggers import TensorBoardLogger

class GestureResNet3D(pl.LightningModule):
    def __init__(self, num_classes, learning_rate=0.0001):
        super(GestureResNet3D, self).__init__()
        self.save_hyperparameters()

        # Load pre-trained ResNet3D-18
        self.model = r3d_18(weights=R3D_18_Weights.DEFAULT)
        
        # Freeze all layers except the last fully connected layer
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Modify the last fully connected layer for classification
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        
        # Unfreeze only the new layers
        for param in self.model.fc.parameters():
            param.requires_grad = True
        
        self.learning_rate = learning_rate
        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
    
    def forward(self, x):
        # ResNet3D expects input shape: (B, C, T, H, W)
        return self.model(x)

    def set_weights(self, weights):
        self.loss_fn = nn.CrossEntropyLoss(weight=weights)

    
    def training_step(self, batch, batch_idx):
        videos, labels = batch  # Shape: (B, C, T, H, W)
        outputs = self(videos)
        loss = self.loss_fn(outputs, labels)
        acc = self.accuracy(outputs, labels)
        
        self.log("train_loss", loss, sync_dist=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        videos, labels = batch
        outputs = self(videos)
        loss = self.loss_fn(outputs, labels)
        acc = self.accuracy(outputs, labels)
        
        self.log("val_loss", loss, sync_dist=True)
        self.log("val_acc", acc, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        videos, labels = batch
        outputs = self(videos)
        loss = self.loss_fn(outputs, labels)
        acc = self.accuracy(outputs, labels)
        
        self.log("test_loss", loss, sync_dist=True)
        self.log("test_acc", acc, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
    
        # Reduce LR when validation loss stops improving
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=3, factor=0.9, verbose=True
        )
    
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",  # Monitor validation loss
                "interval": "epoch",
                "frequency": 1,
            },
            'grad_clip': 1.0  # Clip gradients to a max norm of 1.0
        }
