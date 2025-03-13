import os
import numpy as np
import matplotlib.pyplot as plt
import h5py

import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from utils.DeepSets import DeepSetsClassifier

class Data():

    def __init__(self, params, directories, device):
        '''
        directories: (train, val) path to directory
        device: torch.device
        test: boolean, load only one file
        '''
        self.directories = directories
        self.device = device
        self.test = params['test']
        self.pca_true = params['pca']


    def print_features(self):
        print(f'{self.data.keys()} \n')
        print(f'{self.data["jetFeatureNames"][:]} \n')
        print(f'{self.data["jetConstituentList"][:].shape} \n')
        print(f'{self.data["particleFeatureNames"][:]} \n')
        self.labels = self.data["jetConstituentList"][:]

    def read_and_split_data(self, batch_size=32):
        """
        Splits the dataset into training, validation, and test sets.
        """
        self.train_dataset = self.read_files_from_directory(self.directories[0])
        self.test_dataset = self.read_files_from_directory(self.directories[1])
        self.split_data(batch_size) 
        return self.train_loader, self.val_loader, self.test_loader, self.labels 



    def read_files_from_directory(self, directory):
        y_list = []
        constituents_list = []
        for idx, f in enumerate(os.listdir(directory)):
            with h5py.File(directory + f, "r") as file:
                self.data = file
                if idx == 0:
                    self.print_features()
                    self.labels = self.data["jetFeatureNames"][:][-6:]
                
                y_list.append(torch.tensor(self.data['jets'][:], dtype=torch.float32)[:,-6:])
                constituents_list.append(torch.tensor(self.data['jetConstituentList'][:], dtype=torch.float32))
            if self.test: break
        y = torch.cat(y_list, dim=0)
        constituents = torch.cat(constituents_list, dim=0)
        print(y.shape)
        print(constituents.shape)
        self.data_len = constituents.shape[0]
        self.constituents_len = constituents.shape[1]
        self.input_dim = constituents.shape[2]
        self.output_dim = y.shape[1]
        self.total_size_test = constituents.shape[0]
        
        if self.pca_true:
            constituents, y = self.pca_for_data(constituents, y)
        return TensorDataset(constituents, y) 
    
    def pca_for_data(self, constituents, y):
        constituents_reshaped = constituents.view(-1, 16).cpu().numpy()
        pca_components = 5
        if not hasattr(self, 'pca'):
            self.pca = PCA(n_components=pca_components)
            constituents_pca = self.pca.fit_transform(constituents_reshaped)
        else:
            constituents_pca = self.pca.transform(constituents_reshaped)
        constituents_pca_reshaped = constituents_pca.reshape(self.data_len, self.constituents_len, pca_components)
        constituents = torch.tensor(constituents_pca_reshaped)
        self.input_dim = pca_components 
        return (constituents, y)

    def split_data(self, batch_size):

        val_size = int(self.total_size_test * 0.5)
        test_size = self.total_size_test - val_size
        self.val_dataset, self.test_dataset = random_split(self.test_dataset, [val_size, test_size])
        
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=False)
        self.val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)
