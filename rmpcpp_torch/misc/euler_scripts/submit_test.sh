#!/bin/bash

#SBATCH --ntasks=13
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-cpu=3500
#SBATCH --gres=gpumem:23g
#SBATCH --time=8:00:00
#SBATCH --tmp=2G
#//inactive --dependency=afterok:24736254

world_type="sphere_box"  # sphere_box, planes
test_type="learned"  # learned, expert or baseline
wandb_id="g2j8uxxd" # MAKE SURE TO SET IT TO "" IF STARTING EXPERT/BASELINE RUN FROM SCRATCH

# Only used for learned, and if we use the ray output decoder. Otherwise it is ignored
#decoder_type="max_decoder"
decoder_type="max_sum50_decoder"

# Only used for learned
partial_ray_observation="" # vel_zeros, vel_avg, vel_max

enable_robot_size=true
robot_size=0.1

workers=12

# These only matters if test_type is learned
version="last"

set -x

# Use -D . when calling this script to set this correctly
SCRIPT_DIR=$(pwd)
FULL_PROJECT_DIR="${SCRIPT_DIR}/../../../../../.."

"${SCRIPT_DIR}/run_test.sh" "${FULL_PROJECT_DIR}"  "${test_type}" "${workers}" "${wandb_id}" \
    "${version}" "${world_type}" "${decoder_type}" "${partial_ray_observation}" "${enable_robot_size}" "${robot_size}"

