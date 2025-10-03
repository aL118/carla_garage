#!/bin/bash

# Define a base name for the job, experiment, and output files

#SBATCH --job-name=test                            
#SBATCH --output=/fs/nexus-scratch/aliu1237/carla_garage/my_dump/slurm_output/%x.out.%j       # indicates a file to redirect STDOUT to; %j is the jobid
#SBATCH --error=/fs/nexus-scratch/aliu1237/carla_garage/my_dump/slurm_output/%x.out.%j        # indicates a file to redirect STDERR to; %j is the jobid

## Scale ntasks with gpus
#SBATCH --mem=120gb                                               # memory required by job; if unit is not specified MB will be assumed
#SBATCH --gres=gpu:rtxa5000:8
#SBATCH --ntasks=32

# set up notification settings for failures
##SBATCH --mail-user=angelaliu9805@gmail.com
##SBATCH --mail-type=ALL

## Scavenger training config (low priority, unlimited resources)
##SBATCH --time=48:00:00
##SBATCH --qos=scavenger
##SBATCH --account=scavenger
##SBATCH --partition=scavenger

## GAMMA training config
#SBATCH --time=48:00:00     
#SBATCH --qos=huge-long                                    
#SBATCH --account=gamma
#SBATCH --partition=gamma

eval "$(conda shell.bash hook)"
conda activate garage_2

NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)

echo "Number of GPUS: $NUM_GPUS"

export CARLA_ROOT="/fs/nexus-scratch/aliu1237/carla_garage/carla"
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
# export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla/":${PYTHONPATH}
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/fs/nexus-scratch/aliu1237/miniconda3/envs/garage_2/lib

export OMP_NUM_THREADS=16  # Limits pytorch to spawn at most num cpus cores threads
export OPENBLAS_NUM_THREADS=1  # Shuts off numpy multithreading, to avoid threads spawning other threads.
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nnodes=1 --nproc_per_node=8 --max_restarts=1 --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d \
    team_code/train.py --id train_10 --crop_image 1 --seed 2 --epochs 10 --batch_size 10 --lr 1.875e-4 --setting all \
    --root_dir /fs/nexus-scratch/aliu1237/carla_garage/datasubset \
    --logdir /fs/nexus-scratch/aliu1237/carla_garage/logs \
    --use_controller_input_prediction 1 --use_wp_gru 0 --use_discrete_command 1 --use_tp 1 --tp_attention 0 --continue_epoch 1 --cpu_cores 64 --num_repetitions 1 \
    --max_x 32 --crop_bev_height_only_from_behind 1 --lidar_resolution_height 256 --dataset_cache_name dataset_cache_384 

## sbatch -J test run_train.sh
## /fs/nexus-projects/sim2real/