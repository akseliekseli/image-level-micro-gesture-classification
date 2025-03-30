import torch
import pytorch_lightning as pl
from .Data import Data


class GestureDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, splits):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.splits = splits
        self.data_obj = Data(data_dir, batch_size, splits)
        self.setup()

    def prepare_data(self):
        pass  

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_loader, self.val_loader, _, self.class_to_idx, self.class_weights = self.data_obj.get_dataloaders()
            print(f'LOADER: {self.train_loader}')
        if stage == "test" or stage is None:
            _, _, self.test_loader, _, _ = self.data_obj.get_dataloaders()

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader

