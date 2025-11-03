import os
import sys

# Add directories to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
# Add parent dir for package imports (from team_code import ...)
sys.path.insert(0, parent_dir)
# Also add team_code for internal imports within team_code (import transfuser_utils)
sys.path.insert(0, os.path.join(parent_dir, 'team_code'))

# Add CARLA Python API to path
carla_root = os.path.join(parent_dir, 'carla')
sys.path.insert(0, os.path.join(carla_root, 'PythonAPI'))
sys.path.insert(0, os.path.join(carla_root, 'PythonAPI', 'carla'))

from sim2drive.data import CARLA_Navsim_Data
from team_code.config import GlobalConfig
config = GlobalConfig()
navsim_path = '/fs/nexus-projects/sim2real/aliu/navsim_data/navsim_STRAIGHT'
# Pass the scenario folder directly, not individual routes
# The data loader expects: root -> scenario folders -> route folders
navsim_roots = [navsim_path]
navsim_train_set = CARLA_Navsim_Data(root=navsim_roots,
                                  config=config,
                                  estimate_class_distributions=False,
                                  estimate_sem_distribution=False,
                                  shared_dict=None,
                                  rank=0,
                                  validation=False,
                                  rgb_real=True)
print("NavSim training set size:", len(navsim_train_set))

# def move_results_into_rgb():
#     route = '/fs/nexus-projects/sim2real/aliu/navsim_data'
#     for trajectory in os.listdir(route):
#         for folder in os.listdir(os.path.join(route, trajectory)):
#             full_path = os.path.join(route, trajectory, folder)
#             if os.path.isdir(full_path):
#                 from_dir =os.path.join(full_path, 'rgb')
#                 to_dir = full_path
#                 if not os.path.exists(to_dir):
#                     os.makedirs(to_dir)
#                 src_file = os.path.join(from_dir, 'results.json.gz')
#                 dst_file = os.path.join(to_dir, 'results.json.gz')
#                 if os.path.isfile(src_file):
#                     os.rename(src_file, dst_file)
#         print(f"Finished processing trajectory: {trajectory}")
# move_results_into_rgb()