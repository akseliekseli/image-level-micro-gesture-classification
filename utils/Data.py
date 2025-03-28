import os
import random
from collections import defaultdict
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import torch
from collections import Counter

data_dir = "../data/training"


transform = transforms.Compose([
    #transforms.Resize((256, 256)),  # Resize to 256x256
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


def collate_fn(batch):
    event_ids, images_list, labels_list = zip(*batch)
    
    max_images = max(len(images) for images in images_list)
    
    padded_images = []
    padded_labels = []
    
    for images, labels in zip(images_list, labels_list):
        num_images = len(images)
        
        while len(images) < max_images:
            images.append(torch.zeros_like(images[0]))
            labels.append(-1)
        
        padded_images.append(torch.stack(images))
        padded_labels.append(torch.tensor(labels))

    return event_ids, torch.stack(padded_images), torch.stack(padded_labels)


class GestureDataset(Dataset):
    def __init__(self, event_dict, transform=None):
        self.event_dict = list(event_dict.items())
        self.transform = transform

    def __len__(self):
        return len(self.event_dict)

    def __getitem__(self, idx):
        event_id, image_label_pairs = self.event_dict[idx]
        images, labels = [], []
        
        for img_path, label in image_label_pairs:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            images.append(image)
            labels.append(label)
        
        return event_id, images, labels


class Data():
    def __init__(self, data_dir, batch_size, splits):
        self.data_dir = data_dir
        self.BATCH_SIZE = batch_size
        self.TRAIN_SPLIT = splits[0]
        self.VAL_SPLIT = splits[1]
        self.TEST_SPLIT = splits[2]

    def load_dataset(self):
        event_dict = defaultdict(list)  # {event_id: [(img_path, class_label)]}
        class_to_idx = {}  # {class_name (str): class_idx (int)}
        class_counter = 0
    
        for class_name in os.listdir(self.data_dir):
            class_path = os.path.join(self.data_dir, class_name)
            if os.path.isdir(class_path):
                if class_name not in class_to_idx:
                    class_to_idx[class_name] = class_counter
                    class_counter += 1
    
                for img_name in os.listdir(class_path):
                    if img_name.endswith(".jpg"):
                        event_id = img_name.split('.')[0]  # Extract event ID
                        img_path = os.path.join(class_path, img_name)
                        event_dict[event_id].append((img_path, class_to_idx[class_name]))
        return event_dict, class_to_idx

    
    def split_dataset(self, event_dict, undersampling_threshold=None):
        events = list(event_dict.keys())
        random.shuffle(events)

        # Count class occurrences
        class_counts = Counter()
        for event_id in events:
            for _, label in event_dict[event_id]:
                class_counts[label] += 1

        # Define undersampling threshold
        if undersampling_threshold is None:
            undersampling_threshold = np.percentile(list(class_counts.values()), 75)  # Upper quartile

        # Undersample only overrepresented classes
        undersampled_event_dict = defaultdict(list)
        class_sample_limit = {cls: min(count, undersampling_threshold) for cls, count in class_counts.items()}
        class_current_count = {cls: 0 for cls in class_counts}

        for event_id in events:
            for img_path, label in event_dict[event_id]:
                if class_current_count[label] < class_sample_limit[label]:
                    undersampled_event_dict[event_id].append((img_path, label))
                    class_current_count[label] += 1

        # Get updated event list
        events = list(undersampled_event_dict.keys())

        # Shuffle and split dataset
        random.shuffle(events)
        total_events = len(events)
        train_size = int(total_events * self.TRAIN_SPLIT)
        val_size = int(total_events * self.VAL_SPLIT)

        self.train_events = events[:train_size]
        self.val_events = events[train_size:train_size + val_size]
        self.test_events = events[train_size + val_size:]

        def get_images(event_list):
            images, labels = [], []
            for event_id in event_list:
                for img_path, label in undersampled_event_dict[event_id]:  
                    images.append(img_path)
                    labels.append(label)
            return images, labels

        return get_images(self.train_events), get_images(self.val_events), get_images(self.test_events)
     
    def get_dataloaders(self):
        event_dict, class_to_idx = self.load_dataset()
        self.split_dataset(event_dict)
        
        train_dataset = GestureDataset({k: event_dict[k] for k in self.train_events}, transform)
        val_dataset = GestureDataset({k: event_dict[k] for k in self.val_events}, transform)
        test_dataset = GestureDataset({k: event_dict[k] for k in self.test_events}, transform)
    
        train_labels = []
        for event_id in self.train_events:
            for _, label in event_dict[event_id]:
                train_labels.append(label)
        
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(train_labels),
            y=train_labels
        )
        
        class_weights = torch.tensor(class_weights, dtype=torch.float)
    
        train_loader = DataLoader(train_dataset, batch_size=self.BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=self.BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=self.BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
        print(f"Classes: {class_to_idx}")
        print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")
        print(f"Class Weights: {class_weights}")
    
        return train_loader, val_loader, test_loader, class_to_idx, class_weights


