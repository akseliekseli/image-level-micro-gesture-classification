import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchmetrics
from torchvision import models
from lightning.pytorch.loggers import TensorBoardLogger

class GestureResNet(pl.LightningModule):
    def __init__(self, num_classes, learning_rate=0.0001):
        super(GestureResNet, self).__init__()
        self.save_hyperparameters()  # Saves hyperparameters automatically

        # Load pre-trained ResNet18
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Freeze all layers except the last fully connected layer
        for param in self.model.parameters():
            param.requires_grad = False

        # Replace the last fully connected layer (classifier) to match the number of classes
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Identity()  # Remove the final classifier layer to add custom layers later

        # Create a Sequential block with dense layers
        self.dense_layers = nn.Sequential(
            nn.Linear(num_ftrs, 1024),  # First dense layer after ResNet
            nn.ReLU(),
            nn.Dropout(0.2),  # Dropout for regularization
            nn.Linear(1024, 512),  # Second dense layer
            nn.ReLU(),
            nn.Linear(512, num_classes)  # Final output layer
        )

        self.learning_rate = learning_rate
        self.num_classes = num_classes

        self.loss_fn = torch.nn.CrossEntropyLoss()
        # Accuracy metric
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        # Use ResNet for feature extraction (without the last fully connected layer)
        x = self.model(x)

        # Pass through the dense layers
        x = self.dense_layers(x)

        return x

    def set_weights(self, weights):
        self.loss_fn = torch.nn.CrossEntropyLoss(weight=weights)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.loss_fn(outputs, labels)
        acc = self.accuracy(outputs, labels)

        # Log training loss and accuracy
        self.log("train_loss", loss, sync_dist=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            images, labels = batch
            outputs = self(images)
            loss = self.loss_fn(outputs, labels)
            acc = self.accuracy(outputs, labels)

        # Log validation loss and accuracy
        self.log("val_loss", loss, sync_dist=True)
        self.log("val_acc", acc, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        return loss

    def test_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = F.cross_entropy(outputs, labels)
        acc = self.accuracy(outputs, labels)

        # Log test loss and accuracy
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
