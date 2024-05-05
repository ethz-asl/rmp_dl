#!/bin/bash

#SBATCH -n 16
#SBATCH --time=1:00:00
#SBATCH --mem-per-cpu=2048
#SBATCH --job-name=build

module load gcc/8.2.0 git python_gpu/3.8.5 cmake/3.25.0 glog/0.4.0 gflags/2.2.2 googletest/1.8.1 openmpi/4.1.4 boost/1.73.0 graphviz/2.46.0 open3d/0.9.0
module load eth_proxy

# From https://stackoverflow.com/questions/56962129/how-to-get-original-location-of-script-used-for-slurm-job
if [ -n "${SLURM_JOB_ID:-}" ] ; then
  SCRIPT_DIR=$(scontrol show job "$SLURM_JOB_ID" | awk -F= '/Command=/{print $2}')
else
  SCRIPT_DIR=$(realpath "$0")
fi

SCRIPT_DIR=$(dirname "${SCRIPT_DIR}")

source ${SCRIPT_DIR}/../../../../../../venv/rmp_dl/bin/activate

cd ${SCRIPT_DIR}

catkin build rmpcpp_torch -j 16


