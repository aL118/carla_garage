import os
import sys

# Add team_code directory to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
team_code_dir = os.path.join(os.path.dirname(script_dir), 'team_code')
sys.path.insert(0, team_code_dir)

from data import CARLA_Data
from config import GlobalConfig
config = GlobalConfig()
navsim_path = '/fs/nexus-projects/sim2real/aliu/navsim_data/navsim_STRAIGHT'
# Pass the scenario folder directly, not individual routes
# The data loader expects: root -> scenario folders -> route folders
navsim_roots = [navsim_path]
navsim_train_set = CARLA_Data(root=navsim_roots,
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