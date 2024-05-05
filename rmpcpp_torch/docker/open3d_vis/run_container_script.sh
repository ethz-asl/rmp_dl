#!/bin/bash

# Check for input parameters
if [ $# -lt 2 ]; then
  echo "Usage: $0 <container_name_or_path> <mounting_directory> [additional_args...]"
  exit 1
fi


container="$1"
mount_dir="$2"
additional_args="${@:3}"

SCRIPT_DIR=$(dirname "$0")

# Check if Docker is available
if command -v docker &> /dev/null; then
  echo "Docker detected. Running with Docker..."
  docker run -v "${mount_dir}:/opt/data" \
             -v "${SCRIPT_DIR}"/entrypoint:/opt/entry \
              "${container}" ${additional_args}
  exit_code=$?
  if [ $exit_code -ne 0 ]; then
    echo "Docker command failed with exit code $exit_code"
  fi
  exit $exit_code
fi

# Check if Singularity is available
if command -v singularity &> /dev/null; then
  echo "Singularity detected. Running with Singularity..."
  singularity run --bind "${mount_dir}:/opt/data" \
                  --bind "${SCRIPT_DIR}"/entrypoint:/opt/entry \
                    "${container}" ${additional_args}
  exit_code=$?
  if [ $exit_code -ne 0 ]; then
    echo "Singularity command failed with exit code $exit_code"
  fi
  exit $exit_code
fi

echo "Neither Docker nor Singularity detected. Please install one of these to proceed."
exit 1
