import os
import random
from collections import defaultdict
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image

# Define dataset path
data_dir = "../data/training"

# Parameters

# Define image transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Custom Dataset Class
class GestureDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


class Data():
    # Load dataset
    
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
    
        # Iterate over class folders
        for class_name in os.listdir(self.data_dir):
            class_path = os.path.join(self.data_dir, class_name)
            if os.path.isdir(class_path):
                if class_name not in class_to_idx:
                    class_to_idx[class_name] = class_counter
                    class_counter += 1
    
                # Get all images and group them by event
                for img_name in os.listdir(class_path):
                    if img_name.endswith(".jpg"):
                        event_id = img_name.split('.')[0]  # Extract event ID
                        img_path = os.path.join(class_path, img_name)
                        event_dict[event_id].append((img_path, class_to_idx[class_name]))
    
        return event_dict, class_to_idx
    
    # Split dataset into train, val, test sets
    def split_dataset(self, event_dict):
        events = list(event_dict.keys())
    
        # Compute split sizes
        total_events = len(events)
        train_size = int(total_events * self.TRAIN_SPLIT)
        val_size = int(total_events * self.VAL_SPLIT)
    
        # Assign events to sets
        train_events = events[:train_size]
        val_events = events[train_size:train_size + val_size]
        test_events = events[train_size + val_size:]
    
        def get_images(event_list):
            images, labels = [], []
            for event_id in event_list:
                for img_path, label in event_dict[event_id]:
                    images.append(img_path)
                    labels.append(label)
            return images, labels
    
        return get_images(train_events), get_images(val_events), get_images(test_events)
    
    # Main Function
    def get_dataloaders(self):
        event_dict, class_to_idx = self.load_dataset()
        print(event_dict)
        (train_imgs, train_labels), (val_imgs, val_labels), (test_imgs, test_labels) = self.split_dataset(event_dict)
    
        # Create datasets
        train_dataset = GestureDataset(train_imgs, train_labels, transform)
        val_dataset = GestureDataset(val_imgs, val_labels, transform)
        test_dataset = GestureDataset(test_imgs, test_labels, transform)
    
        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=self.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.BATCH_SIZE, shuffle=False)
        print(f"Classes: {class_to_idx}")
        print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")
    
        return train_loader, val_loader, test_loader, class_to_idx

