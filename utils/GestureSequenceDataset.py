import os
import json
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from collections import defaultdict
import re
import numpy as np
from collections import Counter

class GestureSequenceDataset(Dataset):
    def __init__(self, root_dir, keypoints_file=None, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.keypoints_data = None
        self.data = defaultdict(lambda: defaultdict(list))
        self.labels = []
        self.label_to_idx = {} 
        if keypoints_file:
            # Load precomputed keypoints from JSON
            with open(keypoints_file, "r") as f:
                self.keypoints_data = json.load(f)

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
        keypoints = []
        if self.keypoints_data:
            for img_path in img_paths:
                img_name = os.path.basename(img_path)
                label_name = os.path.basename(os.path.dirname(img_path))
    
                # Retrieve keypoints from JSON, or use a zero tensor if missing
                if label_name in self.keypoints_data and img_name in self.keypoints_data[label_name]:
                    keypoint_array = np.array(self.keypoints_data[label_name][img_name], dtype=np.float32)/112
                else:
                    keypoint_array = np.zeros((17, 2), dtype=np.float32)  # Default to zero if missing
    
                keypoints.append(keypoint_array)

        if self.transform:
            images = [self.transform(img) for img in images]

        label_idx = self.label_to_idx[label]
        
        if self.keypoints_data:
            return images, keypoints, label_idx
        else: return images, label_idx


def collate_fn_keypoints(batch):
    images_list, keypoints_list, labels = zip(*batch)
    max_len = max(len(seq) for seq in images_list)

    padded_sequences_images = []
    padded_sequences_keypoints = []

    for seq_images, seq_keypoints in zip(images_list, keypoints_list):
        seq_len = len(seq_images)
        padding_needed = max_len - seq_len

        # Convert keypoints to tensors
        seq_keypoints = [torch.tensor(k, dtype=torch.float32) for k in seq_keypoints]

        if padding_needed > 0:
            pad_image = torch.zeros_like(seq_images[0])  # Shape (C, H, W)
            pad_keypoint = torch.zeros((17, 2), dtype=torch.float32)  # Shape (17, 2)

            seq_images.extend([pad_image] * padding_needed)
            seq_keypoints.extend([pad_keypoint] * padding_needed)

        padded_sequences_images.append(torch.stack(seq_images))  # (T, C, H, W)
        padded_sequences_keypoints.append(torch.stack(seq_keypoints))  # (T, 17, 2)

    images_tensor = torch.stack(padded_sequences_images)  # (B, T, C, H, W)
    keypoints_tensor = torch.stack(padded_sequences_keypoints)  # (B, T, 17, 2)
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    return images_tensor, keypoints_tensor, labels_tensor


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

def get_dataloaders(root_dir, keypoints_file=None, transform=None, batch_size=8, train_split=0.7, val_split=0.15, test_split=0.15):
    dataset = GestureSequenceDataset(root_dir, keypoints_file, transform=transform)
    total_size = len(dataset)
    train_size = int(train_split * total_size)
    val_size = int(val_split * total_size)
    test_size = total_size - train_size - val_size
    
    if keypoints_file: collate_fn_to_use = collate_fn_keypoints
    else: collate_fn_to_use = collate_fn

    class_counts = Counter([label for label in dataset.labels])
    num_classes = len(class_counts)
    
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_to_use)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_to_use)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_to_use)

    return train_loader, val_loader, test_loader, num_classes
