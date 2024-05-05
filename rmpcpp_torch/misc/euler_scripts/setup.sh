module load gcc/8.2.0 git python_gpu/3.8.5 cmake/3.25.0 glog/0.4.0 gflags/2.2.2 googletest/1.8.1 openmpi/4.1.4  boost/1.73.0 graphviz/2.46.0 open3d/0.9.0

module load eth_proxy

# Wandb has a bug that it leaks file descriptors for artifacts that it is logging, so we increase the limit.
ulimit -n 4096

PROJECT_DIR=${1:-"./../../../../../.."}

source "${PROJECT_DIR}/venv/rmp_dl_385_cu111/bin/activate"
source "${PROJECT_DIR}/catkin_ws/devel/setup.sh"

# Somehow catkin does not correctly set the python path, so we add to it manually
export PYTHONPATH="${PROJECT_DIR}/catkin_ws/devel/lib/python3.8/site-packages:${PYTHONPATH}"

# https://github.com/isl-org/Open3D/issues/897 and https://stackoverflow.com/questions/66497147/cant-run-opengl-on-wsl2/66506098#66506098
export LIBGL_ALWAYS_INDIRECT=0

export WANDB_CACHE_DIR="${SCRATCH}/wandb_cache"

echo $(which python)
echo ${PYTHONPATH}
echo $(nvidia-smi)

