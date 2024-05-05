#!/bin/bash

set -x

PROJECT_DIR=$1
SCRIPT_DIR="${PROJECT_DIR}/catkin_ws/src/rmpcpp/rmpcpp_torch/misc/euler_scripts"

set +x
source ${SCRIPT_DIR}/setup.sh ${PROJECT_DIR}
set -x

SCRIPT_DIR=$(dirname "$0")
cmd=(python "${SCRIPT_DIR}/../../python/rmp_dl/testing/random_world_tester.py" \
 "--test_type" "$2" \
 "--num_workers" "$3" \
 "--wandb_id" "$4" \
 "--version" "$5" \
 "--world_type" "$6" \
 "--decoder_type" "$7" \
 "--partial_ray_observation" "$8" \
 "--enable_robot_size" "$9" \
 "--robot_size" "${10}")

echo "${cmd[@]}"


set +x

"${cmd[@]}"

