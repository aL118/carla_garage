export WORK_DIR="/fs/nexus-scratch/aliu1237/carla_garage"
# export ROUTES=${WORK_DIR}/leaderboard/data/bench2drive220.xml
export ROUTES=${WORK_DIR}/leaderboard/data/longest6.xml
export JSON=${WORK_DIR}/results/pretrained_longest6

python /fs/nexus-scratch/aliu1237/carla_garage/tools/result_parser.py --xml $ROUTES --results $JSON --subset