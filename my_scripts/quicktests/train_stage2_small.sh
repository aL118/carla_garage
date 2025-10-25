#!/bin/bash

# Define a base name for the job, experiment, and output files

#SBATCH --job-name=test                            
#SBATCH --output=/fs/nexus-scratch/aliu1237/carla_garage/my_dump/slurm_output/%x.out.%j       # indicates a file to redirect STDOUT to; %j is the jobid
#SBATCH --error=/fs/nexus-scratch/aliu1237/carla_garage/my_dump/slurm_output/%x.out.%j        # indicates a file to redirect STDERR to; %j is the jobid

## Scale ntasks with gpus
#SBATCH --mem=120gb
#SBATCH --gres=gpu:rtxa6000:4
#SBATCH --ntasks=4

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
echo "Job Name: $SLURM_JOB_NAME"

export CARLA_ROOT="/fs/nexus-scratch/aliu1237/carla_garage/carla"
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
# export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla/":${PYTHONPATH}
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/fs/nexus-scratch/aliu1237/miniconda3/envs/garage_2/lib
export HF_ENDPOINT=https://huggingface.co

export OMP_NUM_THREADS=16  # Limits pytorch to spawn at most num cpus cores threads
export OPENBLAS_NUM_THREADS=1  # Shuts off numpy multithreading, to avoid threads spawning other threads.

# Stage 2 - FAST MODE for quick testing
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes=1 --nproc_per_node=4 --max_restarts=1 --rdzv_id=${SLURM_JOB_ID:-$$} --rdzv_backend=c10d \
    /fs/nexus-scratch/aliu1237/carla_garage/team_code/train.py --id "${SLURM_JOB_NAME:-test}" --crop_image 1 --seed 2 --epochs 10 --batch_size 6 \
    --lr 0.01 --setting 13_withheld  --num_repetitions 1 \
    --root_dir /fs/nexus-projects/sim2real/aliu/carla_garage_data \
    --navsim_path /fs/nexus-projects/sim2real/aliu/navsim_data/navsim_STRAIGHT \
    --logdir /fs/nexus-scratch/aliu1237/carla_garage/logs \
    --use_controller_input_prediction 1 --use_wp_gru 0 --use_discrete_command 1 --use_tp 1 --tp_attention 0 --continue_epoch 0 --cpu_cores 64 \
    --max_x 32 --crop_bev_height_only_from_behind 1 --lidar_resolution_height 256  --use_plant 0 --dataset_cache_name dataset_cache_384 \
    --dropout 0.01 --backbone transFuser_dinov2 --slicing_factor 1000

## sbatch -J test run_train_stage2_sim2drive.sh
## /fs/nexus-projects/sim2real/