import os
import shutil
import random

def prune_images(source_root, dest_root, max_images_per_class=1000):
    if not os.path.exists(dest_root):
        os.makedirs(dest_root)
    
    for class_folder in os.listdir(source_root):
        class_path = os.path.join(source_root, class_folder)
        dest_class_path = os.path.join(dest_root, class_folder)
        
        if os.path.isdir(class_path):
            os.makedirs(dest_class_path, exist_ok=True)
            images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.png'))]
            
            unique_prefixes = {}
            for img in images:
                prefix = img.split('.')[0]  # Extract the unique identifier before the first dot
                if prefix not in unique_prefixes:
                    unique_prefixes[prefix] = img
            
            selected_images = list(unique_prefixes.values())
            
            if len(selected_images) > max_images_per_class:
                selected_images = random.sample(selected_images, max_images_per_class)
            
            for img in selected_images:
                shutil.copy(os.path.join(class_path, img), os.path.join(dest_class_path, img))

source_directory = "data/training"
destination_directory = "data_pruned"
prune_images(source_directory, destination_directory)

