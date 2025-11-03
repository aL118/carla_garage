"""
Image segmentation
Grayscale
Randomized textures
Randomized lighting
Outlines
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random
from skimage import segmentation, color
from skimage.segmentation import slic

def apply_segmentation(image_path):
    """Apply superpixel segmentation using SLIC"""
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    segments = slic(img_rgb, n_segments=100, compactness=10, sigma=1)
    segmented = color.label2rgb(segments, img_rgb, kind='avg', bg_label=0)
    
    return segmented

def apply_grayscale(image_path):
    """Convert image to grayscale and normalize to 0-255 range"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    
    # Ensure values are in 0-255 range
    gray_rgb = np.clip(gray_rgb, 0, 255).astype(np.uint8)
    
    return gray_rgb

def apply_random_lighting(image_path, brightness_range=(-50, 50), contrast_range=(0.5, 1.5)):
    """Apply randomized lighting changes"""
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    brightness = random.randint(brightness_range[0], brightness_range[1])
    contrast = random.uniform(contrast_range[0], contrast_range[1])
    
    adjusted = cv2.convertScaleAbs(img_rgb, alpha=contrast, beta=brightness)
    
    return adjusted

def augment_all(img_array, brightness_range=(-50, 50), contrast_range=(0.5, 1.5)):
    """img_array expects image converted to numpy array"""
    # Convert to BGR for OpenCV processing
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Segmentation
    segments = slic(img_array, n_segments=100, compactness=10, sigma=1)
    segmented = color.label2rgb(segments, img_array, kind='avg', bg_label=0)
    
    # Grayscale
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    grayscale = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    grayscale = np.clip(grayscale, 0, 255).astype(np.uint8)
    
    # Random lighting
    brightness = random.randint(brightness_range[0], brightness_range[1])
    contrast = random.uniform(contrast_range[0], contrast_range[1])
    lighting = cv2.convertScaleAbs(img_array, alpha=contrast, beta=brightness)
    
    return [segmented, grayscale, lighting]

def save_augmented_images(image_path):
    """Apply all augmentations and save using matplotlib"""
    original = cv2.imread(image_path)
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    
    segmented = apply_segmentation(image_path)
    grayscale = apply_grayscale(image_path)
    lighting = apply_random_lighting(image_path)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].imshow(original_rgb)
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(segmented)
    axes[0, 1].set_title('Segmentation')
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(grayscale)
    axes[1, 0].set_title('Grayscale')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(lighting)
    axes[1, 1].set_title('Random Lighting')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('augmentation/augmented_images.png', dpi=150, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    virtual = '/fs/nexus-scratch/aliu1237/carla_splits/left/CAM_F/routes_10mshortroutes_Town01_Scenario9_route0_11_23_20_36_48__0070.png'
    save_augmented_images(virtual)

