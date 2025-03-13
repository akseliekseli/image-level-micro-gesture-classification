import torch
import numpy as np

class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0.0, path='best_model'):
        """
        Args:
            patience (int): How long to wait after last improvement.
            verbose (bool): If True, prints messages for each improvement.
            delta (float): Minimum change to qualify as an improvement.
            path (str): Path to save the best model.
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.path = path
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_loss = np.inf

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves the model when validation loss decreases."""
        if self.verbose:
            print(f"Validation loss improved ({self.best_loss:.6f} → {val_loss:.6f}). Saving model...")
        torch.save(model.state_dict(), self.path+'.pth')
        self.best_loss = val_loss

