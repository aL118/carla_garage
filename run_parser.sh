export WORK_DIR="/fs/nexus-scratch/aliu1237/carla_garage"
export ROUTES=${WORK_DIR}/leaderboard/data/bench2drive220.xml
export JSON=${WORK_DIR}/results

python tools/result_parser.py --xml $ROUTES --results $JSON