#!/bin/bash

SCRIPT_DIR=${1}
FULL_PROJECT_DIR="${SCRIPT_DIR}/../../../../../.."
source ${SCRIPT_DIR}/../setup.sh ${FULL_PROJECT_DIR}

export WANDB__SERVICE_WAIT=300

shift 
python ${SCRIPT_DIR}/../../../python/rmp_dl/testing/real_world_tester/real_world_tester.py $@

