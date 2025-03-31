import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import torchmetrics
from transformers import AutoProcessor, RTDetrForObjectDetection, AutoImageProcessor, VitPoseForPoseEstimation
from torchvision.models.video import swin3d_t, Swin3D_T_Weights
from PIL import Image
import numpy as np


class GestureSwin3DWithPose(pl.LightningModule):
    def __init__(self, num_classes, learning_rate=0.0001):
        super(GestureSwin3DWithPose, self).__init__()
        self.save_hyperparameters()
        self.model = swin3d_t(weights=Swin3D_T_Weights.DEFAULT)
        
        for param in self.model.parameters():
            param.requires_grad = False
        
        num_ftrs = self.model.head.in_features
        self.model.head = nn.Sequential(
            nn.Linear(num_ftrs, 512)
        )
        
        self.classifier = nn.Sequential(
                        nn.Linear(512 + 34, 1028),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(1028, 512),
                        nn.ReLU(),
                        nn.Linear(512, num_classes)
        )
        for param in self.model.head.parameters():
            param.requires_grad = True
        
        for param in self.classifier.parameters():
            param.requires_grad = True
        
        self.learning_rate = learning_rate
        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
    
    def forward(self, x):
        x, keypoints = x[0], x[1]
        x = x.permute(0, 2, 1, 3, 4)
        assert x.shape[1] == 3, f"Expected 3 channels, but got {x.shape[1]}"
        video_features = self.model(x)
        keypoints = keypoints.mean(dim=1).flatten(start_dim=1)  # Shape: [8, 34]
        combined_features = torch.cat((video_features, keypoints), dim=1)
        return self.classifier(combined_features)


    def training_step(self, batch, batch_idx):
        videos, keypoints, labels = batch
        outputs = self((videos, keypoints))
        loss = self.loss_fn(outputs, labels)
        acc = self.accuracy(outputs, labels)
        self.log("train_loss", loss, sync_dist=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        videos, keypoints, labels = batch
        outputs = self((videos, keypoints))
        loss = self.loss_fn(outputs, labels)
        acc = self.accuracy(outputs, labels)
        self.log("val_loss", loss, sync_dist=True)
        self.log("val_acc", acc, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        videos, keypoints, labels = batch
        outputs = self((videos, keypoints))
        loss = self.loss_fn(outputs, labels)
        acc = self.accuracy(outputs, labels)
        self.log("test_loss", loss, sync_dist=True)
        self.log("test_acc", acc, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.9, verbose=True)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss", "interval": "epoch", "frequency": 1}}

