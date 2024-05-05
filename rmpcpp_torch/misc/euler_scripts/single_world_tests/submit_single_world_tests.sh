#!/bin/bash



set -e

FULL_SCRIPT_DIR=$(pwd)

ARGS=(
    "--world-names"
    "sun3d_home_at1"
    "bundlefusion-apt2"
    "bundlefusion-apt0"
    "--break"

    "rrt"

    "--time"
    "0.1"
    "--margin-to-obstacles"
    "0.1"
)

cmd="${FULL_SCRIPT_DIR}/run_single_world_tests.sh ${FULL_SCRIPT_DIR} ${ARGS[@]}"

echo "${cmd}"

# scontrol show node 
# to get node information to figure out which nodes we want to run on
# eu-g3-[016-048] are RTX_2080_TI with AMD EPYC_7742
# We need to specifically select these as there are also other RTX2080TI nodes with different CPUs (xeon gold something)

sbatch  --ntasks=1 \
        --cpus-per-task=1 \
        --nodes=1 \
        --gpus-per-node=1 \
        --mem-per-cpu=8000 \
	    --gres=gpumem:8g \
        --time=2:00:00 \
        --nodelist=eu-g3-[016-048] \ 
        --wrap="${cmd}"

# # FOR TESTING ON NON-SLURM SYSTEMS:
# echo "Executing command:"
# echo "${cmd}"

# ${cmd}


