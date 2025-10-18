# from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
# from nuplan.common.actor_state.state_representation import TimePoint
import numpy as np
import pickle
from PIL import Image
import os
import json
import gzip

class InterpolatedTrajectory:
    pass
class TimePoint:
    pass
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

def trajectory_to_waypoints(
    trajectory: InterpolatedTrajectory,
    num_waypoints: int = 8,
    time_horizon: float = 4.0,  # seconds
) -> np.ndarray:
    """
    Convert InterpolatedTrajectory to discrete waypoints.

    Args:
        trajectory: The interpolated trajectory
        num_waypoints: Number of waypoints to sample
        time_horizon: Total time span to sample over

    Returns:
        Array of shape (num_waypoints, 3) with [x, y, heading]
    """
    # Get start time
    start_time_s = trajectory.start_time.time_s

    # Create time points at regular intervals
    time_intervals = np.linspace(0, time_horizon, num_waypoints)
    time_points = [TimePoint(int((start_time_s + dt) * 1e6)) for dt in time_intervals]

    # Sample ego states at those time points
    ego_states = trajectory.get_state_at_times(time_points)

    # Extract waypoints as [x, y, heading]
    waypoints = np.array([
        [state.rear_axle.x, state.rear_axle.y, state.rear_axle.heading]
        for state in ego_states
    ])

    return waypoints

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

def load_navsim_records(records_path: str):
    """
    Load data from a NavSim records.json.gz file.

    Args:
        records_path: Path to the records.json.gz file

    Returns:
        Dictionary containing the records data with keys:
            - meta_data: dict with index, town, trajectory_type
            - route: list of global waypoints [x, y, yaw]
            - route_ego: list of ego-relative waypoints [x, y, yaw]
            - states, lights, ego_actions, adv_actions: lists (usually empty)
    """
    with gzip.open(records_path, 'rt', encoding='utf-8') as f:
        data = json.load(f)

    return data

def load_metadata():                                                                                                                                                                                                                                                                                                                                                                  
    # Load a sample metadata file                                                                                                                                                                      
    pkl_file = "/fs/nexus-projects/sim2real/aliu/navsim/dataset/navsim_logs/meta_datas/trainval/2021.06.14.18.42.45_veh-12_03913_04017.pkl"                                          
                                                                                                                                                                                                        
    with open(pkl_file, 'rb') as f:                                                                                                                                                                    
        data = pickle.load(f)                                                                                                                                                                          
                                                                                                                                                                                                        
    print(f"Type of data: {type(data)}")                                                                                                                                                               
    print(f"Number of frames in this log: {len(data)}")                                                                                                                                                
    print("\n" + "="*80)                                                                                                                                                                               
    print("SAMPLE FRAME METADATA (first frame):")                                                                                                                                                      
    print("="*80)                                                                                                                                                                                      
                                                                                                                                                                                                        
    # Show first frame                                                                                                                                                                                 
    frame = data[0]                                                                                                                                                                                    
    print(f"\nFrame keys: {frame.keys()}")                                                                                                                                                             
    print("\n--- Core Metadata ---")                                                                                                                                                                   
    print(f"Token: {frame['token']}")                                                                                                                                                                  
    print(f"Timestamp: {frame['timestamp']}")                                                                                                                                                          
    print(f"Log name: {frame['log_name']}")                                                                                                                                                            
                                                                                                                                                                                                        
    print("\n--- Ego State ---")
    # Extract ego position from ego2global_translation
    ego_translation = frame['ego2global_translation']
    print(f"Ego translation (x, y, z): ({ego_translation[0]:.2f}, {ego_translation[1]:.2f}, {ego_translation[2]:.2f})")

    # Extract ego rotation (quaternion: w, x, y, z)
    ego_rotation = frame['ego2global_rotation']
    print(f"Ego rotation (quaternion): [{ego_rotation[0]:.4f}, {ego_rotation[1]:.4f}, {ego_rotation[2]:.4f}, {ego_rotation[3]:.4f}]")

    # CAN bus data structure:
    # [0-2]: translation (x, y, z)
    # [3-6]: rotation quaternion (w, x, y, z)
    # [7-9]: angular velocity
    # [10-12]: linear acceleration
    # [13-15]: angular acceleration
    # [16-17]: steering/other
    can_bus = frame['can_bus']
    print(f"\nCAN Bus Data:")
    print(f"  Linear acceleration: [{can_bus[10]:.4f}, {can_bus[11]:.4f}, {can_bus[12]:.4f}]")
    print(f"  Angular velocity: [{can_bus[7]:.4f}, {can_bus[8]:.4f}, {can_bus[9]:.4f}]")

    # Ego dynamic state (appears to be velocity-related)
    ego_dynamic = frame['ego_dynamic_state']
    print(f"\nEgo dynamic state: {ego_dynamic}")                                                                                                                                            
                                                                                                                                                                                                        
    print("\n--- Route Information ---")                                                                                                                                                               
    print(f"Roadblock IDs (route): {frame['roadblock_ids'][:3]}..." if len(frame['roadblock_ids']) > 3 else f"Roadblock IDs: {frame['roadblock_ids']}")                                                
    print(f"Number of roadblocks in route: {len(frame['roadblock_ids'])}")                                                                                                                             
                                                                                                                                                                                                        
    print("\n--- Annotations (other vehicles/objects) ---")
    # Check if 'annotations' or 'anns' key exists
    ann_key = 'annotations' if 'annotations' in frame else 'anns' if 'anns' in frame else None
    if ann_key and frame[ann_key]:
        anns = frame[ann_key]
        if isinstance(anns, dict):
            if 'instance_tokens' in anns:
                print(f"Number of annotated objects: {len(anns['instance_tokens'])}")
                if len(anns['instance_tokens']) > 0 and 'category_names' in anns and 'boxes' in anns:
                    print(f"First object category: {anns['category_names'][0]}")
                    print(f"First object box (x,y,z,l,w,h,heading): {anns['boxes'][0]}")
            else:
                print(f"Annotations dict keys: {list(anns.keys())}")
        else:
            print(f"Number of annotations: {len(anns)}")                                                                                                       
                                                                                                                                                                                                        
    print("\n--- Traffic Lights ---")                                                                                                                                                                  
    print(f"Traffic lights: {frame['traffic_lights'][:3]}..." if len(frame['traffic_lights']) > 3 else f"Traffic lights: {frame['traffic_lights']}")                                                   
                                                                                                                                                                                                        
    print("\n--- Sensor Data References ---")                                                                                                                                                          
    print(f"Camera keys available: {list(frame['cams'].keys())}")                                                                                                                                      
    print(f"LiDAR available: {'lidar_pc' in frame}")                                                                                                                                                   
                                                                                                                                                                                                        
    if 'cam_f0' in frame['cams']:                                                                                                                                                                      
        print(f"\nFront camera filename: {frame['cams']['cam_f0']['filename']}")                                                                                                                       
        print(f"Front camera intrinsics shape: {frame['cams']['cam_f0']['cam_intrinsic']}")

def log(pkl_file: str):
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)

    print("="*80)
    print("CHECKING VEHICLE MOTION")
    print("="*80)

    # Check when the vehicle starts moving
    print("\nAnalyzing ego positions across frames...")
    start_frame = 0

    print("\n" + "="*80)
    print("WAYPOINT EXTRACTION DEMO")
    print("="*80)

    # Extract waypoints from when vehicle is moving
    # At 10Hz data, frame_skip=5 means 0.5 second intervals
    # 8 waypoints x 0.5 seconds = 4 seconds total
    waypoints = extract_waypoints_from_frames(
        frames=data,
        start_idx=start_frame,
        num_waypoints=8,
        frame_skip=5
    )

    print(f"\nExtracting from frame {start_frame} (vehicle is moving)")

    print(f"\nExtracted {len(waypoints)} waypoints:")
    print(f"Shape: {waypoints.shape}")
    print("\nWaypoints (x, y, yaw):")
    for i, wp in enumerate(waypoints):
        print(f"  WP {i}: x={wp[0]:.2f}, y={wp[1]:.2f}, yaw={wp[2]:.4f} rad ({np.degrees(wp[2]):.2f}°)")

    # Convert to ego-relative coordinates (relative to first waypoint)
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

    print("\n\nEgo-relative waypoints (relative to first frame):")
    for i, wp in enumerate(waypoints_ego):
        print(f"  WP {i}: x={wp[0]:.2f}m, y={wp[1]:.2f}m, yaw={wp[2]:.4f} rad ({np.degrees(wp[2]):.2f}°)")

    print("\n" + "="*80)
    print("\nNow showing full metadata for reference:")
    print("="*80)
    load_metadata()

    # Classify trajectory direction
    print("\n" + "="*80)
    print("TRAJECTORY CLASSIFICATION")
    print("="*80)

    # Use the final waypoint's heading to determine direction
    final_yaw_deg = np.degrees(waypoints_ego[-1, 2])

    # Normalize angle to [-180, 180] range
    final_yaw_deg = ((final_yaw_deg + 180) % 360) - 180

    straight_threshold = 5.0  # degrees

    if abs(final_yaw_deg) <= straight_threshold:
        direction = "STRAIGHT"
    elif final_yaw_deg > straight_threshold:
        direction = "LEFT"
    else:
        direction = "RIGHT"

    print(f"\nFinal heading change: {final_yaw_deg:.2f}°")
    print(f"Threshold for straight: ±{straight_threshold}°")
    print(f"Trajectory direction: {direction}")

    # Extract log name from the pickle file path to name the output
    pkl_basename = os.path.basename(pkl_file).replace('.pkl', '')
    output_gif_path = f"/fs/nexus-scratch/aliu1237/carla_garage/trajectory_gifs/{pkl_basename}_{direction}.gif"

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_gif_path), exist_ok=True)

    create_trajectory_gif(
        frames=data,
        start_idx=start_frame,
        num_frames=8,
        frame_skip=5,
        output_path=output_gif_path
    )

    print(f"\nTrajectory classification: {direction}")
    print(f"GIF saved to: {output_gif_path}")

if __name__ == "__main__":
    data = load_navsim_records("/fs/nexus-projects/sim2real/aliu/carla_garage_data/Accident/Town12_Rep0_10_0_route0_11_08_23_53_07/records.json.gz")
    print(json.dumps(data, indent=2))

"""
/fs/nexus-projects/sim2real/aliu/navsim_data/navsim_LEFT/2021.05.12.22.28.35_veh-35_02138_02481/records.json.gz
{
  "meta_data": {
    "index": "2021.05.12.22.28.35_veh-35_02138_02481",
    "town": "Real/NavSim",
    "trajectory_type": "LEFT"
  },
  "states": [],
  "lights": [],
  "route": [
    [
      664453.6977853155,
      3998777.779693727,
      -1.7747300870672018
    ],...
  ],
  "route_ego": [
    [
      0.0,
      0.0,
      0.0
    ],...
  ],
  "ego_actions": [],
  "adv_actions": []
}
/fs/nexus-projects/sim2real/aliu/navsim_data/navsim_LEFT/2021.05.12.22.28.35_veh-35_02138_02481/results.json.gz
{
  "timestamp": "2021.05.12.22.28.35_veh-35_02138_02481",
  "trajectory": "LEFT",
  "route_id": "2021.05.12.22.28.35_veh-35_02138_02481",
  "meta": {
    "trajectory_type": "LEFT",
    "final_yaw_deg": 8.613190244486788,
    "num_frames": 686,
    "num_waypoints": 8
  }
}
"""