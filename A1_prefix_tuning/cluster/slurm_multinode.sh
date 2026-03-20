#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${CONFIG_PATH:-A1_prefix_tuning/config_hpc.yaml}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
MASTER_PORT="${MASTER_PORT:-29500}"
ENV_FILE="${ENV_FILE:-A1_prefix_tuning/cluster/hpc_env.sh}"

MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)"
WORLD_SIZE="$((SLURM_NNODES * GPUS_PER_NODE))"

cd "${SLURM_SUBMIT_DIR}"

srun --nodes="${SLURM_NNODES}" --ntasks="${SLURM_NNODES}" --ntasks-per-node=1 bash -lc "
set -euo pipefail
if [ -f ${ENV_FILE} ]; then
  source ${ENV_FILE}
fi
source .venv/bin/activate
accelerate launch \
  --num_machines ${SLURM_NNODES} \
  --machine_rank \${SLURM_NODEID} \
  --main_process_ip ${MASTER_ADDR} \
  --main_process_port ${MASTER_PORT} \
  --mixed_precision bf16 \
  --num_processes ${WORLD_SIZE} \
  A1_prefix_tuning/scripts/train.py --config ${CONFIG_PATH}
"
