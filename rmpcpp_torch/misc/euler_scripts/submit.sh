#!/bin/bash

# We exit if any of these commands fail
set -e


run_title="Dense - No Inflations - Planes" 
#run_title="U100-512k - Planes SB Mix - 4 x 256 - NEA - wd5e-9"
num_workers=10

train_dataset_config_name="dense"
validation_dataset_config_name="default"
model_config_name="ffn"

dataset_long_term_storage_path="/cluster/scratch/${USER}/dataset_saves"
temporary_storage_path="/scratch/${USER}/temporary_storage"
wandb_metadata_path="/cluster/scratch/${USER}/wandb_metadata"
logging_path="/cluster/scratch/${USER}/logging"
open3d_image_renderer_path="/cluster/scratch/${USER}/open3d_image_renderer/open3d-v16-image-renderer-final.sif"

mkdir -p "${dataset_long_term_storage_path}"
mkdir -p "${wandb_metadata_path}"
mkdir -p "${logging_path}"

# We copy the parameters file to a new directory, as it can happen that a batch job is stuck in the queue,
# and then the parameters file might be overwritten when setting up the next job. 

SCRIPT_DIR=$(dirname "$0")
FULL_PROJECT_DIR="$(pwd)/../../../../../.."

# Okay if this exists already (-p)
mkdir -p "${SCRIPT_DIR}/../../configs/copies/"

# But we need a unique directory for the file, so this will throw an error if it already exists and we exit.
dir="${SCRIPT_DIR}/../../configs/copies/${run_title}"
mkdir "${dir}"

# Copy the files
cp -r "${SCRIPT_DIR}/../../configs/training/"* "${dir}"
config_path=$(realpath "${dir}")

# Now we can submit the job
cmd=(${SCRIPT_DIR}/run.sh \
        \""${run_title}"\" \
        \""${config_path}"\" \
        \""${train_dataset_config_name}"\" \
        \""${validation_dataset_config_name}"\" \
        \""${model_config_name}"\" \
        ${num_workers} \
        \""${dataset_long_term_storage_path}"\" \
        \""${wandb_metadata_path}"\" \
        \""${logging_path}"\" \
        \""${open3d_image_renderer_path}"\" \
	\""${FULL_PROJECT_DIR}"\")

cmd="${cmd[@]}"

echo "${cmd[@]}"

sbatch  --ntasks=1 \
 	--cpus-per-task=$((num_workers)) \
        --nodes=1 \
        --mem-per-cpu=15000 \
        --gres=gpumem:23g \
        --gpus-per-node=1 \
        --time=120:00:00 \
	--tmp=100G \
        --wrap="${cmd}" 
	
