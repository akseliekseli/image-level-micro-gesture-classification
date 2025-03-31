import os
import torch
import json
from torchvision import transforms
from PIL import Image
from transformers import AutoProcessor, VitPoseForPoseEstimation, RTDetrForObjectDetection
import numpy as np
from tqdm import tqdm
from torchvision.transforms.functional import to_pil_image

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class VitPoseKeypoints:
    def __init__(self, model_name="usyd-community/vitpose-base-simple", device=None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.processor = AutoProcessor.from_pretrained(model_name, use_fast=True)
        self.model = VitPoseForPoseEstimation.from_pretrained(model_name).to(self.device)
        self.person_image_processor = AutoProcessor.from_pretrained("PekingU/rtdetr_r50vd_coco_o365", use_fast=True)
        self.person_model = RTDetrForObjectDetection.from_pretrained("PekingU/rtdetr_r50vd_coco_o365").to(self.device)
        self.model.eval()

    def extract_keypoints(self, image, boxes=None):
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





def generate_keypoints(dataset_path, output_file):
    extractor = VitPoseKeypoints()
    keypoints_data = {}
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor()
    ])
    
    for label in tqdm(os.listdir(dataset_path), desc="Processing labels"):
        label_path = os.path.join(dataset_path, label)
        if not os.path.isdir(label_path):
            continue
        
        keypoints_data[label] = {}
        for img_name in tqdm(sorted(os.listdir(label_path)), desc=f"Processing {label}"):
            img_path = os.path.join(label_path, img_name)
            image = Image.open(img_path)
            image = transform(image).to("cuda")
            keypoints = extractor.extract_keypoints(image)
            keypoints_data[label][img_name] = keypoints.cpu().numpy().tolist()
    with open(output_file, "w") as f:
        json.dump(keypoints_data, f)
    print(f"Keypoints saved to {output_file}")
