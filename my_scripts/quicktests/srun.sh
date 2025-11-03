#!/bin/bash

# Script to run training with srun
# Usage: srun -J job_name [srun options] run_train_stage2_sim2drive_srun.sh
#
# IMPORTANT: Use --ntasks=1 with srun, NOT --ntasks=4
# torchrun will handle spawning multiple GPU processes internally
#
# Example: srun -J test --mem=120gb --gres=gpu:rtxa6000:4 --ntasks=1 --time=48:00:00 --qos=huge-long --account=gamma --partition=gamma run_train_stage2_sim2drive_srun.sh

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate garage_2

# Count available GPUs
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
echo "Number of GPUS: $NUM_GPUS"
echo "Job Name: ${SLURM_JOB_NAME:-unknown}"
echo "Job ID: ${SLURM_JOB_ID:-unknown}"

# Set up CARLA environment
export CARLA_ROOT="/fs/nexus-scratch/aliu1237/carla_garage/carla"
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla/":${PYTHONPATH}
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/fs/nexus-scratch/aliu1237/miniconda3/envs/garage_2/lib
export HF_ENDPOINT=https://huggingface.co

# Threading configuration
export OMP_NUM_THREADS=16  # Limits pytorch to spawn at most num cpus cores threads
export OPENBLAS_NUM_THREADS=1  # Shuts off numpy multithreading, to avoid threads spawning other threads.

# Memory optimization settings
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # Reduce memory fragmentation

# Get job name from environment or use default
JOB_NAME=${SLURM_JOB_NAME:-test}

# Create output directory if it doesn't exist
OUTPUT_DIR=/fs/nexus-scratch/aliu1237/carla_garage/my_dump/slurm_output
mkdir -p $OUTPUT_DIR

# Stage 2 Training
# torchrun handles multi-GPU training internally, so we only need one srun task
# Reduced batch_size from 10 to 6 to fit in GPU memory (RTX A6000 has ~48GB, but models are large)
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes=1 --nproc_per_node=4 --max_restarts=1 --rdzv_id=${SLURM_JOB_ID:-$$} --rdzv_backend=c10d \
    /fs/nexus-scratch/aliu1237/carla_garage/team_code_action/train.py --id "${SLURM_JOB_NAME:-test}" --crop_image 1 --seed 2 --epochs 20 --batch_size 4 \
    --lr 1.875e-4 --setting 13_withheld  --num_repetitions 1 \
    --root_dir /fs/nexus-projects/sim2real/aliu/carla_garage_data \
    --logdir /fs/nexus-scratch/aliu1237/carla_garage/logs \
    --use_controller_input_prediction 1 --use_wp_gru 0 --use_discrete_command 1 --use_tp 1 --tp_attention 0 --continue_epoch 0 --cpu_cores 64 \
    --max_x 32 --crop_bev_height_only_from_behind 1 --lidar_resolution_height 256  --use_plant 0 --dataset_cache_name dataset_cache_384 \
    --dropout 0.001 --backbone transFuser --slicing_factor 500 --detect_boxes 0

# Training time: 2355.5188913345337 seconds
# No navsim data, 10 epochs
# Train loss: 0.7
# Val loss: 0.77
