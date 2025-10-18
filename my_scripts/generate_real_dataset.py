# from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
# from nuplan.common.actor_state.state_representation import TimePoint
import numpy as np
import pickle
from PIL import Image
import os
import json
import gzip  

def quaternion_to_yaw(quat):
    """Convert quaternion [w, x, y, z] to yaw angle."""
    w, x, y, z = quat
    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return yaw

def extract_waypoints_from_frames(
    frames: list,
    start_idx: int = 0,
    num_waypoints: int = 8,
    frame_skip: int = 5,  # Sample every Nth frame
) -> np.ndarray:
    """
    Extract waypoints from consecutive frames.

    Args:
        frames: List of frame dictionaries
        start_idx: Starting frame index
        num_waypoints: Number of waypoints to extract
        frame_skip: Sample every Nth frame (5 means 0.5 seconds at 10Hz)

    Returns:
        Array of shape (num_waypoints, 3) with [x, y, heading] in global coordinates
    """
    waypoints = []

    for i in range(num_waypoints):
        frame_idx = start_idx + i * frame_skip
        if frame_idx >= len(frames):
            # If we run out of frames, use the last frame
            frame_idx = len(frames) - 1

        frame = frames[frame_idx]

        # Get position
        pos = frame['ego2global_translation']

        # Get orientation (convert quaternion to yaw)
        quat = frame['ego2global_rotation']
        yaw = quaternion_to_yaw(quat)

        waypoints.append([pos[0], pos[1], yaw])

    return np.array(waypoints)

def create_trajectory_gif(
    frames: list,
    output_path: str,
    sensor_base_path: str = "/fs/nexus-projects/sim2real/aliu/navsim/dataset/sensor_blobs/trainval"
):
    """
    Create a GIF from front camera images along the trajectory.

    Args:
        frames: List of frame dictionaries
        start_idx: Starting frame index
        num_frames: Number of frames to include in GIF
        frame_skip: Sample every Nth frame
        output_path: Path to save the GIF
        sensor_base_path: Base path to sensor data
    """
    images = []

    # Build a mapping from filename to frame index using metadata
    log_name = frames[0]['log_name']
    cam_f0_dir = os.path.join(sensor_base_path, log_name, 'CAM_F0')

    if not os.path.exists(cam_f0_dir):
        print(f"Error: Camera directory not found at {cam_f0_dir}")
        return

    # Get set of available image files
    available_files = set(os.listdir(cam_f0_dir))

    # Find frames that have images available, in temporal order
    frames_with_images = []
    for frame_idx, frame in enumerate(frames):
        cam_relative_path = frame['cams']['CAM_F0']['data_path']
        filename = os.path.basename(cam_relative_path)
        if filename in available_files:
            frames_with_images.append((frame_idx, filename))

    print(f"Found {len(frames_with_images)} frames with available images out of {len(frames)} total frames")

    # Load images in temporal order, limited by num_frames
    for frame_idx, filename in frames_with_images:
        cam_path = os.path.join(cam_f0_dir, filename)
        try:
            img = Image.open(cam_path)
            images.append(img)
        except Exception as e:
            print(f"Warning: Could not load image {cam_path}: {e}")

    if images and output_path is not None:
        # Save as GIF with 200ms per frame (5 fps)
        images[0].save(
            output_path,
            save_all=True,
            append_images=images[1:],
            duration=100,
            loop=0
        )
        print(f"\nGIF saved to: {output_path}")
        print(f"Number of frames: {len(images)}")
    else:
        print("Warning: No images found to create GIF")

def save_scenario(
    frames: list,
    output_path: str,
    trajectory: str,
    sensor_base_path: str = "/fs/nexus-projects/sim2real/aliu/navsim/dataset/sensor_blobs/trainval"
):
    """
    Create a GIF from front camera images along the trajectory.

    Args:
        frames: List of frame dictionaries
        start_idx: Starting frame index
        num_frames: Number of frames to include in GIF
        frame_skip: Sample every Nth frame
        output_path: Path to save the GIF
        sensor_base_path: Base path to sensor data
    """
    # Build a mapping from filename to frame index using metadata
    log_name = frames[0]['log_name']
    cam_f0_dir = os.path.join(sensor_base_path, log_name, 'CAM_F0')

    if not os.path.exists(cam_f0_dir):
        print(f"Error: Camera directory not found at {cam_f0_dir}")
        return

    # Get set of available image files
    available_files = set(os.listdir(cam_f0_dir))

    # Find frames that have images available, in temporal order
    frames_with_images = []
    for frame_idx, frame in enumerate(frames):
        cam_relative_path = frame['cams']['CAM_F0']['data_path']
        filename = os.path.basename(cam_relative_path)
        if filename in available_files:
            frames_with_images.append((frame_idx, filename))

    print(f"Found {len(frames_with_images)} frames with available images out of {len(frames)} total frames")

    # Create output directory structure
    rgb_output_dir = os.path.join(output_path, f'navsim_{trajectory}', log_name, 'rgb')
    os.makedirs(rgb_output_dir, exist_ok=True)

    # Load and save images in temporal order with sequential numbering
    saved_count = 0
    for frame_idx, filename in frames_with_images:
        cam_path = os.path.join(cam_f0_dir, filename)
        try:
            img = Image.open(cam_path)
            # Save with sequential numbering: 0000.jpg, 0001.jpg, etc.
            output_filename = f"{saved_count:04d}.jpg"
            output_img_path = os.path.join(rgb_output_dir, output_filename)
            img.save(output_img_path)
            saved_count += 1
        except Exception as e:
            print(f"Warning: Could not load/save image {cam_path}: {e}")

    print(f"Saved {saved_count} images to {rgb_output_dir}")

def save_metadata(
    frames: list,
    output_path: str,
    start_idx: int = 0,
    num_waypoints: int = 8,
    frame_skip: int = 5
):
    """
    Calculate and save metadata files in CARLA format (results.json.gz and records.json.gz).

    Args:
        frames: List of frame dictionaries
        output_path: Base output path
        start_idx: Starting frame index for waypoint extraction
        num_waypoints: Number of waypoints to extract
        frame_skip: Sample every Nth frame for waypoints

    Returns:
        Dictionary containing trajectory type and other metadata
    """
    log_name = frames[0]['log_name']

    # Extract waypoints
    waypoints = extract_waypoints_from_frames(
        frames=frames,
        start_idx=start_idx,
        num_waypoints=num_waypoints,
        frame_skip=frame_skip
    )

    # Convert to ego-relative coordinates
    ego_pos = waypoints[0, :2]
    ego_yaw = waypoints[0, 2]

    # Rotation matrix to convert from global to ego frame
    cos_yaw = np.cos(-ego_yaw)
    sin_yaw = np.sin(-ego_yaw)
    rotation_matrix = np.array([[cos_yaw, -sin_yaw],
                                [sin_yaw, cos_yaw]])

    waypoints_ego = []
    for wp in waypoints:
        # Translate to ego origin
        pos_translated = wp[:2] - ego_pos
        # Rotate to ego frame
        pos_ego = rotation_matrix @ pos_translated
        yaw_ego = wp[2] - ego_yaw
        waypoints_ego.append([pos_ego[0], pos_ego[1], yaw_ego])

    waypoints_ego = np.array(waypoints_ego)

    # Classify trajectory direction
    final_yaw_deg = np.degrees(waypoints_ego[-1, 2])
    final_yaw_deg = ((final_yaw_deg + 180) % 360) - 180

    straight_threshold = 5.0  # degrees
    if abs(final_yaw_deg) <= straight_threshold:
        trajectory = "STRAIGHT"
    elif final_yaw_deg > straight_threshold:
        trajectory = "LEFT"
    else:
        trajectory = "RIGHT"

    # Create output directory
    scenario_dir = os.path.join(output_path, f'navsim_{trajectory}', log_name)
    os.makedirs(scenario_dir, exist_ok=True)

    # 1. Create results.json.gz (high-level summary)
    results_data = {
        "timestamp": log_name,
        "trajectory": trajectory,
        "route_id": log_name,
        "meta": {
            "trajectory_type": trajectory,
            "final_yaw_deg": float(final_yaw_deg),
            "num_frames": len(frames),
            "num_waypoints": len(waypoints)
        }
    }

    results_path = os.path.join(scenario_dir, 'results.json.gz')
    with gzip.open(results_path, 'wt', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2)

    # 2. Create records.json.gz (route and waypoint information)
    # Convert waypoints to list format
    waypoints_global = waypoints.tolist()
    waypoints_ego_list = waypoints_ego.tolist()

    records_data = {
        "meta_data": {
            "index": log_name,
            "town": "Real/NavSim",
            "trajectory_type": trajectory
        },
        "states": [],  # Could be populated with per-frame states if needed
        "lights": [],
        "route": waypoints_global,  # Global waypoints
        "route_ego": waypoints_ego_list,  # Ego-relative waypoints
        "ego_actions": [],
        "adv_actions": []
    }

    records_path = os.path.join(scenario_dir, 'records.json.gz')
    with gzip.open(records_path, 'wt', encoding='utf-8') as f:
        json.dump(records_data, f, indent=2)

    return {
        "trajectory": trajectory,
        "waypoints": waypoints,
        "waypoints_ego": waypoints_ego,
        "final_yaw_deg": final_yaw_deg
    }

def main(load_folder: str, save_folder: str):
    i = 0
    for pkl_file in os.listdir(load_folder):
        with open(os.path.join(load_folder, pkl_file), 'rb') as f:
            data = pickle.load(f)

        metadata = save_metadata(
            frames=data,
            output_path=save_folder
        )
        save_scenario(
            frames=data,
            output_path=save_folder,
            trajectory=metadata["trajectory"]
        )
        i += 1
        if i % 100 == 0:
            print(f"Processed {i} scenarios")

if __name__ == "__main__":
    main(load_folder = "/fs/nexus-projects/sim2real/aliu/navsim/dataset/navsim_logs/trainval", save_folder = "/fs/nexus-projects/sim2real/aliu/navsim_data")

    