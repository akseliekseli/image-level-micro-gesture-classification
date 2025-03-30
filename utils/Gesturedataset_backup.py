
import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from collections import defaultdict
import re

class GestureSequenceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        self.data = defaultdict(lambda: defaultdict(list))
        self.labels = []
        self.label_to_idx = {}
        
        pattern = re.compile(r"(\d+)\.(\d+)\.jpg")
        
        for label in sorted(os.listdir(root_dir)):
            label_path = os.path.join(root_dir, label)
            if not os.path.isdir(label_path):
                continue
            
            self.labels.append(label)
            self.label_to_idx[label] = len(self.labels) - 1
            
            for img_name in sorted(os.listdir(label_path)):
                match = pattern.match(img_name)
                if match:
                    seq_id, frame_id = match.groups()
                    seq_id, frame_id = int(seq_id), int(frame_id)
                    img_path = os.path.join(label_path, img_name)
                    self.data[label][seq_id].append((frame_id, img_path))
        
        self.sequences = []
        for label, seq_dict in self.data.items():
            for seq_id, images in seq_dict.items():
                images.sort()
                self.sequences.append((label, [img[1] for img in images]))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        label, img_paths = self.sequences[idx]
        
        images = [Image.open(img_path).convert("RGB") for img_path in img_paths]
        if self.transform:
            images = [self.transform(img) for img in images]
        
        label_idx = self.label_to_idx[label]
        
        return images, label_idx


def collate_fn(batch):
    images_list, labels = zip(*batch)
    max_len = max(len(seq) for seq in images_list)
    
    padded_sequences = []
    for seq in images_list:
        seq_len = len(seq)
        padding_needed = max_len - seq_len
        if padding_needed > 0:
            pad_image = torch.zeros_like(seq[0])
            seq.extend([pad_image] * padding_needed)
        padded_sequences.append(torch.stack(seq))
    
    images_tensor = torch.stack(padded_sequences)
    images_tensor = images_tensor.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    
    return images_tensor, labels_tensor


def get_dataloaders(root_dir, transform, batch_size=8, train_split=0.7, val_split=0.15, test_split=0.15):
    
    dataset = GestureSequenceDataset(root_dir, transform=transform)
    total_size = len(dataset)
    train_size = int(train_split * total_size)
    val_size = int(val_split * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    return train_loader, val_loader, test_loader

