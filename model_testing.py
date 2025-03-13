import sys
import argparse
import yaml
import time
import numpy as np
import torch

from utils.Data import Data
from utils.DeepSets import DeepSetsClassifier
from utils.ModelTester import ModelTester


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trying different quantizations for a trained DeepSets model.")
    parser.add_argument('--config', type=str, required=True, help='config to use')
    parser.add_argument('--name', type=str, required=True, help='config to use')
    parser.add_argument('--quantized', type=bool, required=True, help='config to use')
    args = parser.parse_args()
    config = yaml.safe_load(open(f"{args.config}"))

    params = config[args.name]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    directories = (config[args.name]['train_dir'], config[args.name]['val_dir']) 
    

    data = Data(params, directories, device)
    train, val, test, labels = data.read_and_split_data(batch_size=params['model']['batch_size'])
    print(labels)
    trainer = ModelTester(train, val, test, labels, params, device, quantized=args.quantized)
    trainer.test_model()

    
