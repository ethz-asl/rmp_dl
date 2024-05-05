#!/bin/bash

PROJECT_DIR=${11}
SCRIPT_DIR="${PROJECT_DIR}/catkin_ws/src/rmpcpp/rmpcpp_torch/misc/euler_scripts"

source ${SCRIPT_DIR}/setup.sh ${PROJECT_DIR}
cmd=(python "${SCRIPT_DIR}/../../python/rmp_dl/learning/train.py" \
	 --run_title "${1}" \
	 --config_path "${2}" \
	 --train_dataset_config_name "${3}" \
	 --validation_dataset_config_name "${4}" \
	 --model_config_name "${5}" \
	 --num_workers "${6}" \
	 --dataset_long_term_storage_path "${7}" \
	 --dataset_short_term_caching_path "${TMPDIR}/dataset_cache" \
	 --wandb_metadata_path "${8}" \
	 --logging_path "${9}" \
	 --open3d_renderer_container_path_or_name "${10}" \
	 --temporary_storage_path "${TMPDIR}/temporary_storage" \
)

echo "${cmd[@]}"
"${cmd[@]}"



