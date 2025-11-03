"""
Inference script to run predictions with trained model without CARLA server.
Usage: python inference.py --checkpoint /path/to/model.pth --num_samples 5
"""

import argparse
import json
import sys
import os

# Add paths
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(parent_dir, 'team_code'))
sys.path.insert(0, os.path.join(parent_dir, 'team_code_action'))

# Add CARLA Python API to path
carla_root = os.path.join(parent_dir, 'carla')
sys.path.insert(0, os.path.join(carla_root, 'PythonAPI'))
sys.path.insert(0, os.path.join(carla_root, 'PythonAPI', 'carla'))

import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader

from config import GlobalConfig
from model import LidarCenterNet
from data import CARLA_Data


def load_model(checkpoint_path, config):
    """Load trained model from checkpoint."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create model
    model = LidarCenterNet(config)

    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)

    model.to(device)
    model.eval()

    return model, device


def run_inference(model, dataloader, device, num_samples=5):
    """Run inference on dataset samples."""

    results = []

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            if i >= num_samples:
                break

            # Prepare inputs
            rgb = data['rgb'].to(device, dtype=torch.float32)
            lidar = data['lidar'].to(device, dtype=torch.float32)
            target_point = data['target_point'].to(device, dtype=torch.float32)
            ego_vel = data['speed'].to(device, dtype=torch.float32).unsqueeze(1)
            command = data['command'].to(device, dtype=torch.float32)

            # Run model
            pred_wp, pred_target_speed, pred_checkpoint, pred_semantic, \
            pred_bev_semantic, pred_depth, pred_bounding_box, attention_weights, \
            pred_wp_1, selected_path, pred_steer, pred_throttle = model(
                rgb=rgb,
                lidar_bev=lidar,
                target_point=target_point,
                ego_vel=ego_vel,
                command=command
            )

            # Collect predictions
            sample_result = {
                'sample_idx': i,
                'predictions': {
                    'steering': pred_steer[0].item() if pred_steer is not None else None,
                    'throttle': pred_throttle[0].item() if pred_throttle is not None else None,
                    'waypoints': pred_wp[0].cpu().numpy().tolist() if pred_wp is not None else None,
                    'checkpoints': pred_checkpoint[0].cpu().numpy().tolist() if pred_checkpoint is not None else None,
                    'target_speed_logits': pred_target_speed[0].cpu().numpy().tolist() if pred_target_speed is not None else None,
                },
                'ground_truth': {
                    'steering': data['steer'][0].item() if 'steer' in data else None,
                    'throttle': data['throttle'][0].item() if 'throttle' in data else None,
                    'speed': data['speed'][0].item(),
                }
            }

            results.append(sample_result)

            # Print sample result
            print(f"\n{'='*60}")
            print(f"Sample {i+1}/{num_samples}")
            print(f"{'='*60}")
            if pred_steer is not None:
                print(f"Predicted Steering: {pred_steer[0].item():.4f} | GT: {sample_result['ground_truth']['steering']:.4f}")
            if pred_throttle is not None:
                print(f"Predicted Throttle: {pred_throttle[0].item():.4f} | GT: {sample_result['ground_truth']['throttle']:.4f}")
            print(f"Current Speed: {sample_result['ground_truth']['speed']:.2f} m/s")

            if pred_target_speed is not None:
                predicted_speed_class = torch.argmax(pred_target_speed[0]).item()
                print(f"Predicted Speed Class: {predicted_speed_class}")

            if pred_wp is not None:
                print(f"Predicted Waypoints (first 3): {pred_wp[0][:3].cpu().numpy()}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Run inference with trained model')
    parser.add_argument('--checkpoint', type=str,
                       default='/fs/nexus-scratch/aliu1237/carla_garage/logs/bash/model_0019.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--config_dir', type=str,
                       default='/fs/nexus-scratch/aliu1237/carla_garage/logs/bash',
                       help='Directory containing config.json and args.txt')
    parser.add_argument('--num_samples', type=int, default=5,
                       help='Number of samples to run inference on')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size for inference')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save inference results (JSON)')

    args = parser.parse_args()

    # Load config from saved args
    config_path = Path(args.config_dir) / 'args.txt'
    print(f"Loading config from {config_path}")

    with open(config_path, 'r') as f:
        saved_args = json.load(f)

    # Create config object
    config = GlobalConfig()

    # Update config with saved args
    for key, value in saved_args.items():
        if hasattr(config, key):
            setattr(config, key, value)

    print(f"\nConfig loaded:")
    print(f"  Backbone: {config.backbone}")
    print(f"  Setting: {config.setting}")
    print(f"  Dataset: {config.root_dir}")

    # Load model
    model, device = load_model(args.checkpoint, config)
    print(f"Model loaded successfully on {device}")

    # Load dataset (training data - all towns except val_towns)
    print(f"\nLoading training dataset (Town 12, excluding Town 13)...")

    # Build data_roots list (same as train.py)
    config.data_roots = []
    for td_path in config.root_dir:
        config.data_roots = config.data_roots + [os.path.join(td_path, name) for name in os.listdir(td_path)]

    full_dataset = CARLA_Data(root=config.data_roots, config=config, shared_dict=None, validation=False, rgb_real=False)

    # Apply slicing factor if specified (to match training)
    slicing_factor = 500
    if slicing_factor > 1:
        from torch.utils.data import Subset
        subset_indices = range(0, len(full_dataset), slicing_factor)
        dataset = Subset(full_dataset, subset_indices)
        print(f"Full dataset: {len(full_dataset)} samples")
        print(f"After slicing by {slicing_factor}: {len(dataset)} samples")
    else:
        dataset = full_dataset
        print(f"Dataset loaded: {len(dataset)} samples")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    # Run inference
    print(f"\nRunning inference on {args.num_samples} samples...")
    results = run_inference(model, dataloader, device, num_samples=args.num_samples)

    # Save results if output path specified
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to {args.output}")

    print(f"\nInference complete! Processed {len(results)} samples.")


if __name__ == '__main__':
    main()
