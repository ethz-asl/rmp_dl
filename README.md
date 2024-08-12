# RMP DL

This codebase is an iteration on the one released [here](https://github.com/ethz-asl/reactive_avoidance). This codebase contains learned navigation policies and the infrastructure to train networks for these policies. We provide installation instructions below. For more information on how to run things after installation, refer to the readme in [rmpcpp_torch](rmpcpp_torch/).

# Installation Instructions

## Prerequisites
- Nvblox, install according to their instructions [here](https://github.com/nvidia-isaac/nvblox), and link against the installation when building (see instructions below). 
- Ros is optional for the rmpcpp_torch package, however for the rmpcpp_ros and rmpcpp_lidar_sim packages ROS is required (specifically we used ROS Noetic). Training and running learned policies only uses the rmpcpp_torch package, which does not require ROS. However, catkin is required to build. 
- git lfs if you want to download the pretrained models and datasets defined in [rmpcpp_torch/lfs/](rmpcpp_torch/lfs/).
- OMPL. I built from source and used the main branch at commit id: `763fc2f1e5591a42b0a95316cdf9dd10a6ae20cd`, as the apt release I tried threw segfaults when using the RRT* example, and also iirc the newest ompl version changes how you can link against it with cmake, which made it much easier to use. So if you use some earlier release I think you might run into some issues there. 
- Eigen. This is probably where most your compile errors will come from. OMPL requires a cmake eigen configuration to exist, nvblox links against its own internally downloaded eigen and puts it in its interface, the rmpcpp packages also need eigen, and uses eigen_catkin, which looks for a system eigen, or downloads one if it does not find it. All in all, this is a giant mess, but I recommend installing a system eigen version (nvblox uses 3.4.0, so try that) and hoping for the best. Ideally everything would use nvblox' eigen, but getting ompl to use that one if it is looking for a cmake eigen configuration while that does not exist will probably make things weird.  


What we have used is:
- Cuda 11.1
- Python 3.8

With that we will install these torch versions (installation instructions using pip are given below).
- Torch 1.9.1 (Cuda 11.1) (see https://pytorch.org/get-started/previous-versions/#v191, installed with pip), together with pytorch lightning version 1.7. If you need different python/cuda/torch/lightning versions, use these pages to figure out what might work for you **(and change the corresponding torch version required in [rmpcpp_torch/CMakeLists.txt](rmpcpp_torch/CMakeLists.txt))**
  - https://pytorch.org/get-started/previous-versions
  - https://lightning.ai/docs/pytorch/stable/versioning.html#compatibility-matrix 
  - And note that nvblox has only been tested with CUDA 11.0 - 11.8

Below are full install instructions using these exact versions:

Create a catkin workspace and a Python virtual environment.
```bash
mkdir rmp_dl
cd rmp_dl
python3.8 -m venv venv/rmp_dl
. ./venv/rmp_dl/bin/activate
mkdir catkin_ws
cd catkin_ws
mkdir src
```

At this point we need catkin. Either you install ROS which comes with catkin, or you install it with pip `pip install catkin-tools`. The latter is much more lightweight, but you won't be able to use ROS functionality; so the `rmpcpp_ros` and `rmpcpp_lidar_sim` packages won't work (but we don't need it for `rmpcpp_torch`)

Initialize catkin and clone the required repositories. 
```bash
catkin init

cd src
git clone git@github.com:ethz-asl/eigen_catkin.git
git clone git@github.com:catkin/catkin_simple.git

git clone git@github.com:ethz-asl/rmp_dl.git --recurse-submodules
```

At this point we tell catkin where we installed nvblox. Follow their installation instructions [here](https://github.com/nvidia-isaac/nvblox) to set that up. **Note that we use nvblox version 0.0.4**; there are a few lines of code in the bindings that would require changes if using the newest version. The newest version also is not compatible with CUDA < 11.2.
```bash
catkin config -a --cmake-args -Dnvblox_DIR=/path/to/nvblox/install
```
Next we also tell catkin where we installed OMPL. 
```bash
catkin config -a --cmake-args -Dompl_DIR=/path/to/ompl/install
```

If you installed catkin via pip, you will also need to clone the catkin repo into the source space here.

```bash
git clone https://github.com/ros/catkin.git
```

We install some more pip packages and the pytorch prerequisites. Again, make sure that your CUDA version is compatible with this, or choose other versions otherwise. 
```bash
pip install empy 
pip install catkin_pkg
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
```

Make sure we are building in release mode
```bash
catkin config -a --cmake-args -DCMAKE_BUILD_TYPE=Release
```

Also it is useful to export the compile commands for if you're using a language server like e.g. clangd.
```bash
catkin config -a --cmake-args -DCMAKE_EXPORT_COMPILE_COMMANDS=ON 
```

And we build. If you used the catkin python package, you really have to specify `rmpcpp_torch` here, as the `rmpcpp_ros` and `rmpcpp_lidar_sim` won't build. We don't need them for anything in this repo though, so I specify it anyway.
```bash
catkin build rmpcpp_torch
```

Again, if you're using a language server like clangd, you can combine all compile commands into a single json and configure the language server to find this file (command below taken from: https://github.com/catkin/catkin_tools/issues/551#issuecomment-732989966)
```bash
cd ..
jq -s 'map(.[])' build/**/compile_commands.json > build/compile_commands.json
cd src
```

Source the setup script (change to .bash, .sh, .zsh depending on your shell). Remember to do this anytime you run any code. 
```bash
. ../devel/setup.bash
```

We go to the rmpcpp_torch directory, install the rmp_dl and comparison_planners packages as an editable package, and install some more dependencies with pip. If you installed different pytorch versions, make sure that the pytorch-lightning version is compatible here!
```bash
cd rmp_dl/rmpcpp_torch
pip install -e ./python
pip install -e ./python/comparison_planners

pip install attrs open3d==0.13.0 scikit-fmm pytorch-lightning==1.7

# See: https://github.com/AUTOMATIC1111/stable-diffusion-webui/issues/11648
pip install torchmetrics==0.11.4

# Just make sure you're not on 16, as that means wandb will start a new run when downloading an artifact: 
# https://github.com/wandb/wandb/issues/6604
pip install wandb==0.15.4

pip install graphviz
pip install lxml

# There is some code that can use dash for some information
pip install dash
```

Finally, for doing visualizations in open3d during training, we use a docker image that has to be built:
```bash
docker build docker/open3d_vis/ --tag open3d-v16-image-renderer-final
```
This takes quite a while, and is only necessary for the simple wall visualizations. Training works fine without it, it will just skip that step if the docker image is not found. 




# Getting things running on a slurm system

## Installation
Installation is quite similar to described in the readme above, with the change that we use singularity instead of Docker for headless open3d rendering, and we use the pip catkin package by default. Furtermore, we use some python packages in the form of modules that we load (this might be outdated still below). 

## Singularity

We start with this, as this step takes ages to run. We list the docker images, and find the `open3d-v16-image-renderer-final` that we created previously. 
```bash
docker images
```

Take note of the id, and put it in the command below: Save image in tarball and copy to cluster. Make sure to create the correct directories first, or adjust the line accordingly. 
```bash
docker save DOCKER_ID -o open3d-v16-image-renderer-final.tar
scp open3d-v16-image-renderer-final.tar USERNAME@euler.ethz.ch:/cluster/scratch/USERNAME/open3d_image_renderer
```

SSH into euler in the folder above where we put the tarball and create the singularity image. We request an interactive node with a 4 hour timelimit as this step can take quite long.
```bash
srun --ntasks=1 --time=04:00:00 --pty bash
singularity build open3d-v16-image-renderer-final.sif docker-archive:open3d-v16-image-renderer-final.tar
```

## Installation on Euler
In the meantime in a new login node, we set up the repo. 
```bash
module load gcc/8.2.0 git python_gpu/3.8.5 cmake/3.25.0 glog/0.4.0 gflags/2.2.2 googletest/1.8.1 boost/1.73.0 eigen/3.3.9

mkdir rmp_dl
cd rmp_dl
```

We create a virtual environment with the system packages, as they are loaded as system-wide modules. 
```bash
python3.8 -m venv venv/rmp_dl --system-site-packages
. ./venv/rmp_dl/bin/activate
mkdir catkin_ws
cd catkin_ws

mkdir src
cd src
```

We use pip catkin on the cluster as we can't install ROS.
```bash

git clone git@github.com:ros/catkin.git
git clone git@github.com:ethz-asl/eigen_catkin.git
git clone git@github.com:catkin/catkin_simple.git
TODO: git clone git@github.com:ethz-asl/rmp_dl.git --recurse-submodules

pip install catkin-tools
catkin init --workspace ../

pip install empy
```

Make sure to set the paths to nvblox and ompl
```bash
catkin config -a --cmake-args -Dnvblox_DIR=path/to/nvblox/install -Dompl_DIR=/path/to/ompl/install
```

Request an interactive node to start building
```bash
srun --ntasks=4 --time=02:00:00 --pty bash
```

And then build. 
```bash
module load gcc/8.2.0 git python_gpu/3.8.5 cmake/3.25.0 glog/0.4.0 gflags/2.2.2 googletest/1.8.1 boost/1.73.0 eigen/3.3.9
catkin build rmpcpp_torch -j 4
```

Exit the compute node
```bash
exit
```

Source the environment
```bash
. ../devel/setup.bash
```

Install as editable package
```bash
cd rmp_dl/rmpcpp_torch
pip install -e ./python
pip install -e ./python/comparison_planners

pip install scikit-fmm

# See: https://github.com/AUTOMATIC1111/stable-diffusion-webui/issues/11648
pip install torchmetrics==0.11.4

# Just make sure you're not on 16, as that means wandb will start a new run when downloading an artifact: 
# https://github.com/wandb/wandb/issues/6604
pip install wandb==0.15.4

pip install huggingface_hub graphviz lxml dash
```

At this point everything should be set (once the singularity build has finished running). Training runs are started via the scripts in [misc/euler_scripts](misc/euler_scripts). Simply run `./submit.sh` and it will submit a run. In the script a title is given to the run, which is what will show up in the wandb interface. To change parameters of the run, edit the configuration files in [configs/](configs/). Once a run is started, it will log in `$SCRATCH/logging/$wandb_id/`. Doing a `tail -f -n 100 debug.log` is a good way to see if everything is running correctly (as it takes a while before training actually starts due to initial setup of the datapipeline). 

