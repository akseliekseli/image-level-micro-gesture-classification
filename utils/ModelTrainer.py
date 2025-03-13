import os
import numpy as np
import matplotlib.pyplot as plt
import h5py

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.utils.tensorboard import SummaryWriter
import optuna

from utils.DeepSets import DeepSetsClassifier
from utils.EarlyStopping import EarlyStopping

class ModelTrainer():

    def __init__(self, train, val, test, params, device):
        self.writer = SummaryWriter(log_dir='runs')
        self.model_name = params['model_name']
        self.train_loader = train
        self.val_loader = val
        self.test_loader = test
        self.device = device
        self.params = params
        self.early_stopping_patience = params['model']['early_stopping_patience']
        self.lr_tune_patience = params['model']['lr_tune_patience']
        self.epochs = params['model']['epochs']
        self.lr = params['model']['lr']

        for input, target in self.train_loader:
            self.input_dim = input.shape[2]
            self.output_dim = target.shape[1]
            break
        
        self.model = DeepSetsClassifier(self.input_dim,
                                        self.output_dim,
                                        params['model']['n_constituents'],
                                        params['model']['layers'],
                                        self.device,
                                        params['model']['perm_invar_fun'])
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss_fn = torch.nn.CrossEntropyLoss().to(self.device)
        #self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', patience=self.lr_tune_patience, factor=0.5)
        self.early_stopping = EarlyStopping(path=f'models/{self.model_name}',
                                            verbose=True,
                                            patience=self.early_stopping_patience)
        

    def train_model(self):
        """
        Trains a DeepSetsClassifier model using the provided datasets.
        """
        self.train_statistics = {
            "train_loss": [],
            "val_loss": [],
            "test_loss": [],
            "train_acc": [],
            "val_acc": [],
            "test_acc": []
        }

        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            correct_train = 0
            total_train = 0
    
            for X_batch, y_batch in self.train_loader:
                X_batch, y_batch = X_batch.cuda(), y_batch.cuda()
                self.optimizer.zero_grad()
                predictions = self.model(X_batch)
                loss = self.loss_fn(predictions, y_batch)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                correct_train += (predictions.argmax(dim=1) == y_batch.argmax(dim=1)).sum().item()
                total_train += y_batch.size(0)
            
            train_acc = correct_train / total_train
            self.train_statistics["train_loss"].append(total_loss / len(self.train_loader))
            self.train_statistics["train_acc"].append(train_acc)
            
            self.model.eval()
            val_loss = 0
            correct_val = 0
            total_val = 0
            with torch.no_grad():
                for X_batch, y_batch in self.val_loader:
                    X_batch, y_batch = X_batch.cuda(), y_batch.cuda()
                    predictions = self.model(X_batch.cuda())
                    loss = self.loss_fn(predictions, y_batch.cuda())
                    val_loss += loss.item()
                    correct_val += (predictions.argmax(dim=1) == y_batch.argmax(dim=1)).sum().item()
                    total_val += y_batch.size(0)
             
            #self.scheduler.step(val_loss)
            
            val_acc = correct_val / total_val
            self.train_statistics["val_loss"].append(val_loss / len(self.val_loader))
            self.train_statistics["val_acc"].append(val_acc)

            self.writer.add_scalar("loss/train", self.train_statistics['train_loss'][-1], epoch)
            self.writer.add_scalar("loss/val", self.train_statistics['val_loss'][-1], epoch)
            self.writer.add_scalar("accuracy/train", self.train_statistics['train_acc'][-1], epoch)
            self.writer.add_scalar("accuracy/val", self.train_statistics['val_acc'][-1], epoch)
                
            print(f"Epoch {epoch+1}/{self.epochs}, Train Loss: {total_loss/len(self.train_loader):.4f}, Val Loss: {val_loss/len(self.val_loader):.4f}")
            # Check Early Stopping
            self.early_stopping(val_loss, self.model)
            if self.early_stopping.early_stop:
                print("Early stopping triggered!")
                break

        return val_acc

    def test_model(self):
        # Test accuracy calculation
        self.model.eval()
        test_loss = 0
        correct_test = 0
        total_test = 0
        
        with torch.no_grad():
            for X_batch, y_batch in self.test_loader:
                X_batch, y_batch = X_batch.cuda(), y_batch.cuda()
                predictions = self.model(X_batch.cuda())
                loss = self.loss_fn(predictions, y_batch.cuda())
                test_loss += loss.item()
                correct_test += (predictions.argmax(dim=1) == y_batch.argmax(dim=1)).sum().item()
                total_test += y_batch.size(0)
        
        test_acc = correct_test / total_test
        self.train_statistics["test_loss"].append(test_loss / len(self.test_loader))
        self.train_statistics["test_acc"].append(test_acc)
        
        print(f"Test Loss: {test_loss / len(self.test_loader):.4f}, Test Accuracy: {test_acc:.4f}")
        
        self.quantize_model()

        self.writer.add_figure('Train Statistics', self.plot_train_statistics())
        self.writer.close()


    def tune_hyperparameters(self):
        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective, n_trials=10)  # Run 10 trials
        print("Best hyperparameters:", study.best_params)


    def objective(self, trial):
        lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)  # Learning rate
        n_constituents = trial.suggest_categorical("n_constituents", [8, 12, 16, 24, 32])  # Batch size
        self.model = DeepSetsClassifier(self.input_dim,
                                        self.output_dim,
                                        n_constituents,
                                        self.params['model']['layers'],
                                        self.device,
                                        self.params['model']['perm_invar_fun'])
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = torch.nn.CrossEntropyLoss().to(self.device)
        self.early_stopping = EarlyStopping(path=f'models/{self.model_name}',
                                            verbose=True,
                                            patience=self.early_stopping_patience)
        val_acc = self.train_model()
        return val_acc

    def plot_train_statistics(self):
        epochs = range(1, len(self.train_statistics["train_loss"]) + 1)
    
        fig = plt.figure(figsize=(12, 5))
            
        # Plot Loss
        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.train_statistics["train_loss"], label="Train Loss")
        plt.plot(epochs, self.train_statistics["val_loss"], label="Val Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
            
        # Plot Accuracy
        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.train_statistics["train_acc"], label="Train Accuracy")
        plt.plot(epochs, self.train_statistics["val_acc"], label="Val Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.title("Training and Validation Accuracy")
        plt.legend()
            
        plt.savefig(f'plots/{self.model_name}_train_statistics.png')
        return fig

    def quantize_model(self):
        layers = set(module for module in self.model.modules() if not isinstance(module, torch.nn.Sequential))

        self.quantized_model = torch.ao.quantization.quantize_dynamic(self.model,
                                                            layers,
                                                            dtype=torch.qint8)
        torch.save(self.quantized_model.state_dict(), f'models/quantized_{self.model_name}'+'.pth')

