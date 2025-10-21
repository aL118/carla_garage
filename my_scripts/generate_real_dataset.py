# from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
# from nuplan.common.actor_state.state_representation import TimePoint
import numpy as np
import pickle
from PIL import Image
import os
import json
import gzip  
from tqdm import tqdm

def quaternion_to_yaw(quat):
    """Convert quaternion [w, x, y, z] to yaw angle."""
    w, x, y, z = quat
    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return yaw

def quaternion_to_rotation_matrix(quat):
    """Convert quaternion [w, x, y, z] to 3x3 rotation matrix."""
    w, x, y, z = quat

    # Compute rotation matrix elements
    r00 = 1 - 2*(y**2 + z**2)
    r01 = 2*(x*y - w*z)
    r02 = 2*(x*z + w*y)

    r10 = 2*(x*y + w*z)
    r11 = 1 - 2*(x**2 + z**2)
    r12 = 2*(y*z - w*x)

    r20 = 2*(x*z - w*y)
    r21 = 2*(y*z + w*x)
    r22 = 1 - 2*(x**2 + y**2)

    return np.array([[r00, r01, r02],
                     [r10, r11, r12],
                     [r20, r21, r22]])

def frame_to_carla_measurement(frame, route_ego=None, frame_idx_in_route=0, trajectory_type=None):
    """
    Convert NavSim frame data to CARLA measurement format.

    Args:
        frame: NavSim frame dictionary
        route_ego: List of ego-frame waypoints [x, y, z] (optional)
        frame_idx_in_route: Index of this frame in the route (for route extraction)
        trajectory_type: Trajectory type string ("STRAIGHT", "LEFT", "RIGHT")

    Returns:
        Dictionary in CARLA measurement format
    """
    # Extract position and rotation
    pos = frame['ego2global_translation']  # [x, y, z] in global coordinates
    quat = frame['ego2global_rotation']   # [w, x, y, z]

    # Convert quaternion to rotation matrix
    rotation_matrix = quaternion_to_rotation_matrix(quat)

    # Create 4x4 ego_matrix (homogeneous transformation matrix)
    ego_matrix = np.eye(4)
    ego_matrix[:3, :3] = rotation_matrix
    ego_matrix[:3, 3] = pos

    # Extract velocity and acceleration from ego_dynamic_state
    # ego_dynamic_state = [velocity_x, velocity_y, velocity_z, acceleration_norm]
    ego_dynamic = frame.get('ego_dynamic_state', [0, 0, 0, 0])
    velocity_x = ego_dynamic[0] if len(ego_dynamic) > 0 else 0
    velocity_y = ego_dynamic[1] if len(ego_dynamic) > 1 else 0
    speed = np.sqrt(velocity_x**2 + velocity_y**2)

    # Extract yaw angle and steering angle
    theta = quaternion_to_yaw(quat)

    # Extract steering angle from can_bus
    # can_bus format: [x, y, z, qw, qx, qy, qz, ax, ay, az, vx, vy, angular_rates..., steering, ...]
    can_bus = frame.get('can_bus', None)
    if can_bus is not None and len(can_bus) > 16:
        steering_angle = float(can_bus[16])  # Steering angle in radians
    else:
        steering_angle = 0.0

    # Calculate angle (lateral error angle) - for now use steering as proxy
    # In CARLA, angle is the steering angle
    angle = steering_angle

    # Map trajectory_type to CARLA command
    # CARLA command encoding: 1=LEFT, 2=RIGHT, 3=STRAIGHT, 4=LANEFOLLOW
    if trajectory_type == "LEFT":
        command = 1
    elif trajectory_type == "RIGHT":
        command = 2
    elif trajectory_type == "STRAIGHT":
        command = 3
    else:
        # Default to LANEFOLLOW if trajectory_type not provided
        command = 4

    # Create route from route_ego if available
    route = None
    route_original = None
    target_point = None
    target_point_next = None

    if route_ego is not None and len(route_ego) > frame_idx_in_route:
        # Extract future waypoints in ego frame
        future_waypoints = route_ego[frame_idx_in_route:]
        route = [[wp[0], wp[1]] for wp in future_waypoints[:20]]  # Take next 20 waypoints
        route_original = route.copy()

        # Set target_point to a waypoint far ahead (e.g., ~20-30 meters)
        # Look for waypoint at roughly 25 meters ahead
        for i, wp in enumerate(future_waypoints):
            dist = np.sqrt(wp[0]**2 + wp[1]**2)
            if dist >= 25.0 or i == len(future_waypoints) - 1:
                target_point = [float(wp[0]), float(wp[1])]
                # Next target point slightly further
                if i + 1 < len(future_waypoints):
                    next_wp = future_waypoints[i + 1]
                    target_point_next = [float(next_wp[0]), float(next_wp[1])]
                else:
                    target_point_next = target_point
                break

        # If no waypoint found far enough, use the last one
        if target_point is None and len(future_waypoints) > 0:
            last_wp = future_waypoints[-1]
            target_point = [float(last_wp[0]), float(last_wp[1])]
            target_point_next = target_point

    # Build measurement dictionary in CARLA format
    measurement = {
        # Position and orientation
        "pos_global": [float(pos[0]), float(pos[1])],
        "theta": float(theta),
        "ego_matrix": ego_matrix.tolist(),

        # Velocity
        "speed": float(speed),
        "target_speed": None,  # Not available in NavSim
        "speed_limit": None,   # Not available in NavSim

        # Navigation
        "target_point": target_point,
        "target_point_next": target_point_next,
        "command": int(command),
        "next_command": int(command),

        # Route information
        "aim_wp": route[0] if route and len(route) > 0 else None,  # First waypoint in route
        "route": route,
        "route_original": route_original,
        "changed_route": False,

        # Hazards and obstacles (not available in NavSim)
        "speed_reduced_by_obj_type": None,
        "speed_reduced_by_obj_id": None,
        "speed_reduced_by_obj_distance": None,

        # Control inputs
        "steer": steering_angle,
        "throttle": None,  # Not available in NavSim
        "brake": None,     # Not available in NavSim
        "control_brake": None,

        # Traffic conditions (not available in NavSim)
        "junction": None,
        "vehicle_hazard": None,
        "vehicle_affecting_id": None,
        "light_hazard": None,
        "walker_hazard": None,
        "walker_affecting_id": None,
        "stop_sign_hazard": None,
        "stop_sign_close": None,
        "walker_close": None,
        "walker_close_id": None,

        # Steering angle
        "angle": angle,

        # Augmentation (set to 0 for real data)
        "augmentation_translation": 0.0,
        "augmentation_rotation": 0.0,
    }
    return measurement

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


def save_scenario(
    frames: list,
    output_path: str,
    trajectory: str,
    sensor_base_path: str = "/fs/nexus-projects/sim2real/aliu/navsim/dataset/sensor_blobs/trainval"
):
    """
    Save RGB images for frames that have available camera data.

    Args:
        frames: List of frame dictionaries
        output_path: Path to save the images
        trajectory: Trajectory type for directory naming
        sensor_base_path: Base path to sensor data

    Returns:
        List of (original_frame_idx, sequential_idx) tuples for frames with images
    """
    # Build a mapping from filename to frame index using metadata
    log_name = frames[0]['log_name']
    cam_f0_dir = os.path.join(sensor_base_path, log_name, 'CAM_F0')

    if not os.path.exists(cam_f0_dir):
        print(f"Error: Camera directory not found at {cam_f0_dir}")
        return []

    # Get set of available image files
    available_files = set(os.listdir(cam_f0_dir))

    # Find frames that have images available, in temporal order
    frames_with_images = []
    for frame_idx, frame in enumerate(frames):
        cam_relative_path = frame['cams']['CAM_F0']['data_path']
        filename = os.path.basename(cam_relative_path)
        if filename in available_files:
            frames_with_images.append((frame_idx, filename))

    # Create output directory structure
    rgb_output_dir = os.path.join(output_path, f'navsim_{trajectory}', log_name, 'rgb')
    os.makedirs(rgb_output_dir, exist_ok=True)

    # Load and save images in temporal order with sequential numbering
    saved_indices = []
    saved_count = 0
    for frame_idx, filename in frames_with_images:
        cam_path = os.path.join(cam_f0_dir, filename)
        try:
            img = Image.open(cam_path)
            # Save with sequential numbering: 0000.jpg, 0001.jpg, etc.
            output_filename = f"{saved_count:04d}.jpg"
            output_img_path = os.path.join(rgb_output_dir, output_filename)
            img.save(output_img_path)
            saved_indices.append((frame_idx, saved_count))
            saved_count += 1
        except Exception as e:
            print(f"Warning: Could not load/save image {cam_path}: {e}")
    return saved_indices

def get_metadata(
    frames: list,
    output_path: str,
    start_idx: int = 0,
    num_waypoints: int = 8,
    frame_skip: int = 5
):
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

    waypoints_global = waypoints.tolist()
    waypoints_ego_list = waypoints_ego.tolist()

    return {
        "trajectory_type": trajectory,
        "timestamp": log_name,
        "route": waypoints_global,  # Global waypoints
        "route_ego": waypoints_ego_list,  # Ego-relative waypoints
        "ego_actions": [],
        "adv_actions": [],
        "trajectory": trajectory,
        "waypoints": waypoints,
        "waypoints_ego": waypoints_ego,
        "final_yaw_deg": final_yaw_deg
    }

def save_metadata(results_data, records_data, output_path: str, trajectory: str):
    log_name = results_data["timestamp"]
    scenario_dir = os.path.join(output_path, f'navsim_{trajectory}', log_name)
    os.makedirs(scenario_dir, exist_ok=True)
    results_path = os.path.join(scenario_dir, 'results.json.gz')
    with gzip.open(results_path, 'wt', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2)
    records_path = os.path.join(scenario_dir, 'records.json.gz')
    with gzip.open(records_path, 'wt', encoding='utf-8') as f:
        json.dump(records_data, f, indent=2)

def save_measurement(measurement: dict, output_path: str, frame_idx: int):
    os.makedirs(output_path, exist_ok=True)
    measurement_file = os.path.join(output_path, f'{frame_idx:04d}.json.gz')
    with gzip.open(measurement_file, 'wt', encoding='utf-8') as f:
        json.dump(measurement, f, indent=2)

def main(load_folder: str, save_folder: str):
    pkl_files = os.listdir(load_folder)
    for pkl_file in tqdm(pkl_files, desc="Processing scenarios"):
        with open(os.path.join(load_folder, pkl_file), 'rb') as f:
            data = pickle.load(f)

        general_metadata = get_metadata(
            frames=data,
            output_path=save_folder,
            start_idx=0
        )

        # Save RGB images and get mapping of frame indices
        saved_indices = save_scenario(
            frames=data,
            output_path=save_folder,
            trajectory=general_metadata["trajectory"]
        )

        # Create measurements only for frames with images
        measurements_dir = os.path.join(
            save_folder,
            f'navsim_{general_metadata["trajectory"]}',
            general_metadata['timestamp'],
            'measurements'
        )

        for original_frame_idx, sequential_idx in saved_indices:
            frame = data[original_frame_idx]
            measurement = frame_to_carla_measurement(
                frame,
                route_ego=general_metadata['route_ego'],
                frame_idx_in_route=original_frame_idx,
                trajectory_type=general_metadata['trajectory']
            )
            # Use sequential_idx to match RGB numbering (0000.jpg -> 0000.json.gz)
            save_measurement(measurement, measurements_dir, sequential_idx)

if __name__ == "__main__":
    main(load_folder = "/fs/nexus-projects/sim2real/aliu/navsim/dataset/navsim_logs/trainval", save_folder = "/fs/nexus-projects/sim2real/aliu/navsim_data")

def create_trajectory_gif(
    frames: list,
    start_idx: int,
    num_frames: int,
    frame_skip: int,
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
    for frame_idx, filename in frames_with_images[:num_frames]:
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