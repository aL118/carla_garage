#!/bin/bash

# Define a base name for the job, experiment, and output files

#SBATCH --job-name=test
#SBATCH --output=/fs/nexus-scratch/aliu1237/carla_garage/my_dump/slurm_output/%x.out.%j       # indicates a file to redirect STDOUT to; %j is the jobid
#SBATCH --error=/fs/nexus-scratch/aliu1237/carla_garage/my_dump/slurm_output/%x.out.%j        # indicates a file to redirect STDERR to; %j is the jobid

## Scale ntasks with gpus
#SBATCH --mem=120gb                                               # memory required by job; if unit is not specified MB will be assumed
#SBATCH --gres=gpu:rtxa6000:4
#SBATCH --ntasks=16

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
export WORK_DIR="/fs/nexus-scratch/aliu1237/carla_garage"

export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export SCENARIO_RUNNER_ROOT=${WORK_DIR}/scenario_runner
export LEADERBOARD_ROOT=${WORK_DIR}/leaderboard
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
# export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla/":"${SCENARIO_RUNNER_ROOT}":"${LEADERBOARD_ROOT}":${PYTHONPATH}

export ROUTES=${WORK_DIR}/leaderboard/data/longest6.xml
export REPETITIONS=1

export CHALLENGE_TRACK_CODENAME=SENSORS
export RUN_NAME="pretrained"
export CHECKPOINT_ENDPOINT=${WORK_DIR}/results/${RUN_NAME}.json

export TEAM_AGENT=${WORK_DIR}/team_code/sensor_agent.py
export TEAM_CONFIG=${WORK_DIR}/pretrained_models/all_towns
# export TEAM_CONFIG=${WORK_DIR}/logs/${RUN_NAME}

export DEBUG_CHALLENGE=1 # set to 1 to save debug images and measurements
export RESUME=0
export DATAGEN=0
export PORT=2000

export SAVE_PATH="$WORK_DIR/${RUN_NAME}_output" # uncomment for debug output

# Cleanup function with progress
cleanup() {
    echo ""
    echo "🛑 Script interrupted! Performing cleanup..."
    
    # Kill evaluator if running
    if [[ -n $EVALUATOR_PID ]]; then
        echo "📊 Stopping evaluator (PID: $EVALUATOR_PID)..."
        kill -TERM $EVALUATOR_PID 2>/dev/null
    fi
    
    # Kill CARLA
    if [[ -n $CARLA_PID ]]; then
        echo "🚗 Stopping CARLA server (PID: $CARLA_PID)..."
        kill -TERM $CARLA_PID 2>/dev/null
        sleep 3
        kill -KILL $CARLA_PID 2>/dev/null
    fi
    
    # Nuclear option - kill all CARLA processes
    echo "🔥 Killing all CARLA processes..."
    pkill -f -KILL CarlaUE4 2>/dev/null
    
    # Verify cleanup
    if pgrep -f CarlaUE4 > /dev/null; then
        echo "⚠️  Some CARLA processes may still be running"
    else
        echo "✅ All CARLA processes stopped"
    fi
    
    exit 0
}

# Set trap
trap cleanup SIGINT SIGTERM

echo "🚀 Starting CARLA evaluation (Press Ctrl+C to stop cleanly)"
echo "Loading models in $TEAM_CONFIG"

mkdir -p $WORK_DIR/logs 

# Start CARLA
# carla/CarlaUE4.sh -vulkan -RenderOffscreen -nosound
DISPLAY= ${CARLA_ROOT}/CarlaUE4.sh -carla-port=${PORT} -fps=20 -vulkan -RenderOffscreen -nosound > $WORK_DIR/logs/carla.log 2>&1 & 
CARLA_PID=$!
echo "CARLA started with PID: $CARLA_PID"

sleep 15

# Run evaluator in background to capture its PID
python -u ${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator_local.py \
--routes=${ROUTES} \
--repetitions=${REPETITIONS} \
--track=${CHALLENGE_TRACK_CODENAME} \
--checkpoint=${CHECKPOINT_ENDPOINT} \
--agent=${TEAM_AGENT} \
--agent-config=${TEAM_CONFIG} \
--debug=${DEBUG_CHALLENGE} \
--port=${PORT} \
--scenario-limit=10 \
--resume=${RESUME} > $WORK_DIR/logs/evaluation.log 2>&1 &

EVALUATOR_PID=$!
echo "Evaluator started with PID: $EVALUATOR_PID"

# Wait for evaluator to complete
wait $EVALUATOR_PID

# Normal cleanup
cleanup

# bash /fs/nexus-scratch/aliu1237/transfuser/leaderboard/scripts/local_evaluation.sh