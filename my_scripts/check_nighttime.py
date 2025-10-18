#!/usr/bin/env python3
"""
Script to check for nighttime data in sensor_blobs dataset.
Samples one image from each of 100 vehicle folders.
"""

import os
import random
from pathlib import Path
import numpy as np
from PIL import Image

def is_nighttime(image_path, threshold=60):
    """
    Determine if an image is from nighttime based on average brightness.

    Args:
        image_path: Path to the image file
        threshold: Brightness threshold (0-255). Below this is considered night.

    Returns:
        bool: True if nighttime, False otherwise
    """
    try:
        img = Image.open(image_path)
        # Convert to grayscale and get average brightness
        img_array = np.array(img.convert('L'))
        avg_brightness = np.mean(img_array)
        return avg_brightness < threshold
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def main():
    dataset_path = Path("/fs/nexus-projects/sim2real/aliu/carla_garage_data")

    # Get all 1st level folders (e.g., Accident, EmergencyVehicles, etc.)
    first_level_folders = [f for f in dataset_path.iterdir() if f.is_dir()]
    print(f"Found {len(first_level_folders)} 1st level folders: {[f.name for f in first_level_folders]}\n")

    # Sample 10 vehicle folders from each 1st level folder
    sampled_folders = []
    for first_level_folder in first_level_folders:
        veh_folders = [f for f in first_level_folder.iterdir() if f.is_dir()]
        print(f"{first_level_folder.name}: {len(veh_folders)} vehicle folders")

        # Sample up to 10 folders from this category
        sample_count = min(10, len(veh_folders))
        if sample_count > 0:
            sampled = random.sample(veh_folders, sample_count)
            sampled_folders.extend(sampled)

    sample_size = len(sampled_folders)
    print(f"\nTotal sampled folders: {sample_size}\n")

    nighttime_count = 0
    daytime_count = 0
    error_count = 0

    for i, veh_folder in enumerate(sampled_folders, 1):
        # Use CAM_F0 (front camera) for consistency
        cam_folder = veh_folder / "rgb"

        if not cam_folder.exists():
            print(f"{i}. {veh_folder.name}: CAM_F0 not found")
            error_count += 1
            continue

        # Get first image from the camera folder
        image_files = sorted([f for f in cam_folder.iterdir() if f.suffix in ['.jpg', '.png', '.jpeg']])

        if not image_files:
            print(f"{i}. {veh_folder.name}: No images found")
            error_count += 1
            continue

        # Check first image
        i = random.randint(0, len(image_files)-1)
        image_path = image_files[i]
        is_night = is_nighttime(image_path)

        if is_night is None:
            error_count += 1
            continue

        if is_night:
            nighttime_count += 1
            time_label = "NIGHT"
        else:
            daytime_count += 1
            time_label = "DAY"

        # Print every 10th sample
        if i % (sample_size/10) == 0:
            print(f"{i}. {veh_folder.name}: {time_label}")

    # Calculate statistics
    total_valid = nighttime_count + daytime_count

    print("\n" + "="*60)
    print("RESULTS:")
    print("="*60)
    print(f"Total samples analyzed: {sample_size}")
    print(f"Valid samples: {total_valid}")
    print(f"Errors: {error_count}")
    print(f"\nNighttime samples: {nighttime_count}")
    print(f"Daytime samples: {daytime_count}")

    if total_valid > 0:
        nighttime_pct = (nighttime_count / total_valid) * 100
        daytime_pct = (daytime_count / total_valid) * 100
        print(f"\nNighttime percentage: {nighttime_pct:.1f}%")
        print(f"Daytime percentage: {daytime_pct:.1f}%")

if __name__ == "__main__":
    main()
    # print(is_nighttime("/fs/nexus-projects/sim2real/aliu/carla_garage_data/Accident/Town12_Rep0_10_0_route0_11_08_23_53_07/rgb/0008.jpg"))
