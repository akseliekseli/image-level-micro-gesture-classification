import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from collections import defaultdict
import re
from transformers import AutoProcessor, VitPoseForPoseEstimation, RTDetrForObjectDetection
import numpy as np


class VitPoseKeypoints:
    def __init__(self, model_name="usyd-community/vitpose-base-simple", device=None):
        """
        Initializes the ViTPose model for keypoint extraction.
        :param model_name: Hugging Face model name.
        :param device: Device to run the model on ('cuda' or 'cpu').
        """
        self.device = device if device else ('cuda:1' if torch.cuda.is_available() else 'cpu')
        self.processor = AutoProcessor.from_pretrained(model_name, use_fast=True)
        self.model = VitPoseForPoseEstimation.from_pretrained(model_name).to(self.device)
        self.person_image_processor = AutoProcessor.from_pretrained("PekingU/rtdetr_r50vd_coco_o365", use_fast=True)
        self.person_model = RTDetrForObjectDetection.from_pretrained("PekingU/rtdetr_r50vd_coco_o365", device_map=device).to(self.device)

        self.model.eval()  # Set model to evaluation mode

    def extract_keypoints(self, image, boxes=None):
        if image.min() < 0 or image.max() > 1:
            image = (image - image.min()) / (image.max() - image.min())
            image = image.clamp(0, 1)
        else:
            image = image.convert("RGB")
        inputs = self.person_image_processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = self.person_model(**inputs)
        
        results = self.person_image_processor.post_process_object_detection(
        outputs, target_sizes=torch.tensor([(image.shape[0], image.shape[1])]), threshold=0.1)
        result = results[0]
        person_boxes = result["boxes"][result["labels"] == 0].cpu()

        person_boxes[:, 2] = person_boxes[:, 2] - person_boxes[:, 0]
        person_boxes[:, 3] = person_boxes[:, 3] - person_boxes[:, 1]
        inputs = self.processor(image.cpu(), boxes=[person_boxes], return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs.to('cuda'))
        pose_results = self.processor.post_process_pose_estimation(outputs, boxes=[person_boxes])
        image_pose_result = pose_results[0]
        return image_pose_result[0]['keypoints']


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

        # Initialize the keypoint extractor (ViTPose)
        self.keypoint_extractor = VitPoseKeypoints()

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        label, img_paths = self.sequences[idx]
        
        images = [Image.open(img_path).convert("RGB") for img_path in img_paths]
        keypoints = [self.keypoint_extractor.extract_keypoints(self.transform(img).to(self.keypoint_extractor.device) if self.transform else img.to(self.keypoint_extractor.device)) for img in images]
        if self.transform:
            images = [self.transform(img) for img in images]
        
        label_idx = self.label_to_idx[label]
        
        return images, keypoints, label_idx


def collate_fn(batch):
    images_list, keypoints_list, labels = zip(*batch)
    max_len = max(len(seq) for seq in images_list)

    padded_sequences_images = []
    padded_sequences_keypoints = []

    for seq_images, seq_keypoints in zip(images_list, keypoints_list):
        seq_len = len(seq_images)
        padding_needed = max_len - seq_len

        # Ensure every keypoint sequence has shape (17, 2)
        seq_keypoints = [np.array(k, dtype=np.float32) if isinstance(k, (list, np.ndarray)) else np.zeros((17, 2), dtype=np.float32) for k in seq_keypoints]

        if padding_needed > 0:
            pad_image = torch.zeros_like(seq_images[0])  # Shape (C, H, W)
            pad_keypoint = np.zeros((17, 2), dtype=np.float32)  # Shape (17, 2)

            seq_images.extend([pad_image] * padding_needed)
            seq_keypoints.extend([pad_keypoint] * padding_needed)

        # Convert to tensors after ensuring consistent shape
        padded_sequences_images.append(torch.stack(seq_images))  # (T, C, H, W)
        seq_keypoints = np.array(seq_keypoints, dtype=np.float32)
        padded_sequences_keypoints.append(torch.tensor(seq_keypoints))

    images_tensor = torch.stack(padded_sequences_images)  # (B, T, C, H, W)
    keypoints_tensor = torch.stack(padded_sequences_keypoints)  # (B, T, 17, 2)
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    return images_tensor, keypoints_tensor, labels_tensor


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
