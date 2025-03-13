import os
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import h5py

from sklearn.metrics import confusion_matrix

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchmetrics.classification import MulticlassConfusionMatrix
from torchmetrics.classification import MulticlassPrecision, MulticlassRecall, MulticlassF1Score
from utils.DeepSets import DeepSetsClassifier


class ModelTester():

    def __init__(self, train, val, test, y_labels, params, device, quantized=False):
        self.model_name = params['model_name']
        print(quantized)
        if quantized:
            self.model_name = 'quantized_'+self.model_name
        self.train_loader = train
        self.val_loader = val
        self.test_loader = test
        self.labels = y_labels
        self.device = device
        self.early_stopping_patience = params['model']['early_stopping_patience']
        self.lr_tune_patience = params['model']['lr_tune_patience']
        self.epochs = params['model']['epochs']
        self.lr = params['model']['lr']

        for input, target in self.train_loader:
            self.input_dim = input.shape[2]
            self.output_dim = target.shape[1]
            break
        
        self.model = DeepSetsClassifier(self.input_dim, self.output_dim, params['model']['n_constituents'], params['model']['layers'], self.device, params['model']['perm_invar_fun'])
        
        self.model.load_state_dict(torch.load(f'models/{self.model_name}'+'.pth'))
        self.loss_fn = torch.nn.CrossEntropyLoss().to(self.device)

    def test_model(self):
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad) 
        print(f"Total parameters: {total_params}")
        print(f"Trainable parameters: {trainable_params}")
        
        self.compute_loss()
        self.pred_1d = torch.argmax(self.predictions, dim=1).cpu() 
        self.true_1d = torch.argmax(self.true, dim=1).cpu()
        self.precision_recall_f1()
        self.confusion_matrix()

    def confusion_matrix(self):

        cm = confusion_matrix(self.true_1d, self.pred_1d)
        # transform to percentages
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        fig = plt.figure(figsize=(6,5))
        sns.heatmap(cm_percent, annot=True, fmt=".4f", cmap="Blues", xticklabels=self.labels, yticklabels=self.labels)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix")
        plt.savefig(f'plots/{self.model_name}_confusion_matrix.png')


    def precision_recall_f1(self):
        precision_metric = MulticlassPrecision(num_classes=self.output_dim, average=None)
        recall_metric = MulticlassRecall(num_classes=self.output_dim, average=None)
        f1_metric = MulticlassF1Score(num_classes=self.output_dim, average=None)
        
        precision = precision_metric(self.pred_1d, self.true_1d)
        recall = recall_metric(self.pred_1d, self.true_1d)
        f1 = f1_metric(self.pred_1d, self.true_1d)
        
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1-Score: {f1}")

    def compute_loss(self):
        self.model.eval()
        test_loss = 0
        correct_test = 0
        total_test = 0
        
        start = time.time()

        pred_list, true_list = [], []
        with torch.no_grad():
            for X_batch, y_batch in self.test_loader:
                X_batch, y_batch = X_batch.cuda(), y_batch.cuda()
                predictions = self.model(X_batch)
                loss = self.loss_fn(predictions, y_batch)
                test_loss += loss.item()
                correct_test += (predictions.argmax(dim=1) == y_batch.argmax(dim=1)).sum().item()
                total_test += y_batch.size(0)
                pred_list.append(predictions)
                true_list.append(y_batch)
        end = time.time()
        self.predictions = torch.cat(pred_list)
        self.true = torch.cat(true_list)
        print(self.predictions.shape)
        
        test_acc = correct_test / total_test
        
        print(f"Test Loss: {test_loss / len(self.test_loader):.4f}, Test Accuracy: {test_acc:.4f}")
        print(f"Inference in {1000000000*(end-start)/len(self.predictions)} ns")

    def quantize_model(self):
        layers = set(module for module in model.modules() if not isinstance(module, nn.Sequential))

        self.quantized_model = torch.ao.quantization.quantize_dynamic(self.model,
                                                            layers,
                                                            dtype=torch.qint8)
        torch.save(self.quantized_model.state_dict(), f'models/quantized_{model_name}'+'.pth')

