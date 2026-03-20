#!/bin/bash -l
#SBATCH --job-name=run_experiment
#SBATCH --output=results_%A.out
#SBATCH --time=00:59:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a40:1
#SBATCH --nodes=1

set -euo pipefail

module purge
module load python

CONTAINER_PATH="${CONTAINER_PATH:-A1_prefix_tuning/container/a1_prefix_tuning.sif}"
CONFIG_PATH="${CONFIG_PATH:-/workspace/A1_prefix_tuning/config_hpc.yaml}"
ENV_FILE="${ENV_FILE:-A1_prefix_tuning/cluster/hpc_env.sh}"
WORKSPACE_MOUNT="${WORKSPACE_MOUNT:-$PWD}"

source "${ENV_FILE}"

apptainer run \
  --nv \
  --bind "${WORKSPACE_MOUNT}:/workspace" \
  "${CONTAINER_PATH}" \
  python3 /workspace/A1_prefix_tuning/scripts/preprocess.py --config "${CONFIG_PATH}"

apptainer run \
  --nv \
  --bind "${WORKSPACE_MOUNT}:/workspace" \
  "${CONTAINER_PATH}" \
  accelerate launch --num_processes 1 /workspace/A1_prefix_tuning/scripts/train.py --config "${CONFIG_PATH}"
