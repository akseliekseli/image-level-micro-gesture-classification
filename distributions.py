import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import confusion_matrix


def plot_class_distribution(data_dir):
    class_counts = {}
    for class_folder in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_folder)
        if os.path.isdir(class_path):
            num_images = len([f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.png'))])
            class_counts[class_folder] = num_images
    
    plt.figure(figsize=(12, 6))
    plt.bar(class_counts.keys(), class_counts.values(), color='skyblue')
    plt.xlabel('Class', fontsize=14)
    plt.ylabel('Number of Images', fontsize=14)
    plt.title('Image Distribution Across Classes', fontsize=16)
    plt.xticks(rotation=45, fontsize=12)
    plt.savefig("class_distribution.png", dpi=300)
    plt.close()

# Example usage:
plot_class_distribution("data_pruned")

