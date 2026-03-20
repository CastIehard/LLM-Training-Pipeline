# A1 Prefix Tuning

Single-GPU prefix tuning for `Qwen/Qwen3-0.6B` using Hugging Face Transformers, PEFT, Accelerate, and Apptainer.

This module is now scoped to one GPU on one node. There is no multi-node setup in `A1_prefix_tuning`.

## What It Does

The pipeline has four scripts:

1. `scripts/preprocess.py`
   - reads the source Q&A file
   - groups by hash to avoid train/valid/test leakage
   - writes chat-formatted `train.jsonl`, `valid.jsonl`, and `test.jsonl`
2. `scripts/train.py`
   - loads `Qwen/Qwen3-0.6B`
   - applies a PEFT `PrefixTuningConfig`
   - trains only the prefix parameters
   - requires CUDA and fails fast if PyTorch cannot see a GPU
3. `scripts/evaluate.py`
   - reloads the base model plus the saved adapter
   - computes loss and perplexity
   - writes sample generations
4. `scripts/infer.py`
   - runs prompt-based inference with the saved adapter

## Files

Main files in this module:

- [config.yaml](/home/reese/llms/final_project/UTN-3-LLM-Final-Project/A1_prefix_tuning/config.yaml)
- [config_hpc.yaml](/home/reese/llms/final_project/UTN-3-LLM-Final-Project/A1_prefix_tuning/config_hpc.yaml)
- [run_pipeline.py](/home/reese/llms/final_project/UTN-3-LLM-Final-Project/A1_prefix_tuning/run_pipeline.py)
- [scripts/preprocess.py](/home/reese/llms/final_project/UTN-3-LLM-Final-Project/A1_prefix_tuning/scripts/preprocess.py)
- [scripts/train.py](/home/reese/llms/final_project/UTN-3-LLM-Final-Project/A1_prefix_tuning/scripts/train.py)
- [scripts/evaluate.py](/home/reese/llms/final_project/UTN-3-LLM-Final-Project/A1_prefix_tuning/scripts/evaluate.py)
- [scripts/infer.py](/home/reese/llms/final_project/UTN-3-LLM-Final-Project/A1_prefix_tuning/scripts/infer.py)
- [cluster/hpc_env.sh](/home/reese/llms/final_project/UTN-3-LLM-Final-Project/A1_prefix_tuning/cluster/hpc_env.sh)
- [cluster/slurm_apptainer_single_node.sh](/home/reese/llms/final_project/UTN-3-LLM-Final-Project/A1_prefix_tuning/cluster/slurm_apptainer_single_node.sh)
- [container/apptainer.def](/home/reese/llms/final_project/UTN-3-LLM-Final-Project/A1_prefix_tuning/container/apptainer.def)
- [container/requirements.txt](/home/reese/llms/final_project/UTN-3-LLM-Final-Project/A1_prefix_tuning/container/requirements.txt)

## Recommended Workflow

Use the Apptainer container plus the single-node SLURM script.

The supported path is:

1. build the container
2. set the storage environment
3. submit the one-GPU SLURM job

## Build The Container

Build the `.sif` image from [container/apptainer.def](/home/reese/llms/final_project/UTN-3-LLM-Final-Project/A1_prefix_tuning/container/apptainer.def#L1):

```bash
apptainer build A1_prefix_tuning/container/a1_prefix_tuning.sif A1_prefix_tuning/container/apptainer.def
```

The container installs the Python stack from [container/requirements.txt](/home/reese/llms/final_project/UTN-3-LLM-Final-Project/A1_prefix_tuning/container/requirements.txt#L1) and uses a generic runscript, so you can pass normal commands after `apptainer run`.

## Storage Layout On The HPC

This module assumes the following storage policy:

- code stays in `/home/hpc` or your repo checkout
- large model downloads, Hugging Face cache, dataset cache, split files, and checkpoints go to a workspace or `$WORK`
- node-local temporary files go to `$TMPDIR`
- optional exported final artifacts can go to `/home/hpc` or `/home/vault`

The storage variables are set by [cluster/hpc_env.sh](/home/reese/llms/final_project/UTN-3-LLM-Final-Project/A1_prefix_tuning/cluster/hpc_env.sh#L1).

It resolves:

- `PREFIX_TUNING_PERSISTENT_ROOT`
- `PREFIX_TUNING_SCRATCH_ROOT`
- `PREFIX_TUNING_HF_HOME`
- `PREFIX_TUNING_MODEL_CACHE_DIR`
- `PREFIX_TUNING_DATASET_CACHE_DIR`
- `PREFIX_TUNING_DATA_DIR`
- `PREFIX_TUNING_OUTPUT_DIR`

Typical usage:

```bash
# Option 1: use a workspace name if ws_find is available
export PREFIX_TUNING_WORKSPACE_NAME=prefix-tuning

# Option 2: set the path directly
# export PREFIX_TUNING_PERSISTENT_ROOT="$(ws_find prefix-tuning)"

source A1_prefix_tuning/cluster/hpc_env.sh
```

If you want long-term retention outside the workspace lifetime:

```bash
export PREFIX_TUNING_RESULTS_EXPORT_DIR="/home/vault/$USER/prefix_tuning_results"
source A1_prefix_tuning/cluster/hpc_env.sh
```

## Configs

There are two configs:

- [config.yaml](/home/reese/llms/final_project/UTN-3-LLM-Final-Project/A1_prefix_tuning/config.yaml): local/default paths
- [config_hpc.yaml](/home/reese/llms/final_project/UTN-3-LLM-Final-Project/A1_prefix_tuning/config_hpc.yaml): env-driven HPC paths

For the cluster and container flow, use `config_hpc.yaml`.

Important defaults in [config_hpc.yaml](/home/reese/llms/final_project/UTN-3-LLM-Final-Project/A1_prefix_tuning/config_hpc.yaml#L1):

- model: `Qwen/Qwen3-0.6B`
- dtype: `bfloat16`
- max sequence length: `1024`
- train batch size per device: `4`
- eval batch size per device: `4`
- gradient accumulation steps: `8`
- learning rate: `0.002`
- output dir: `${PREFIX_TUNING_OUTPUT_DIR}`

## Manual Commands

If you want to run steps manually inside the container:

```bash
source A1_prefix_tuning/cluster/hpc_env.sh

apptainer run --nv --bind "$PWD:/workspace" A1_prefix_tuning/container/a1_prefix_tuning.sif \
  python3 /workspace/A1_prefix_tuning/scripts/preprocess.py --config /workspace/A1_prefix_tuning/config_hpc.yaml

apptainer run --nv --bind "$PWD:/workspace" A1_prefix_tuning/container/a1_prefix_tuning.sif \
  accelerate launch --num_processes 1 /workspace/A1_prefix_tuning/scripts/train.py --config /workspace/A1_prefix_tuning/config_hpc.yaml

apptainer run --nv --bind "$PWD:/workspace" A1_prefix_tuning/container/a1_prefix_tuning.sif \
  python3 /workspace/A1_prefix_tuning/scripts/evaluate.py --config /workspace/A1_prefix_tuning/config_hpc.yaml
```

Inference:

```bash
apptainer run --nv --bind "$PWD:/workspace" A1_prefix_tuning/container/a1_prefix_tuning.sif \
  python3 /workspace/A1_prefix_tuning/scripts/infer.py \
    --config /workspace/A1_prefix_tuning/config_hpc.yaml \
    --prompt "What support exists for international students at UTN?"
```

## SLURM Job

Use [cluster/slurm_apptainer_single_node.sh](/home/reese/llms/final_project/UTN-3-LLM-Final-Project/A1_prefix_tuning/cluster/slurm_apptainer_single_node.sh#L1).

Submit:

```bash
sbatch A1_prefix_tuning/cluster/slurm_apptainer_single_node.sh
```

The script is intentionally aligned with your requested job shape:

- `#SBATCH --job-name=run_experiment`
- `#SBATCH --output=results_%A.out`
- `#SBATCH --time=00:59:00`
- `#SBATCH --ntasks=1`
- `#SBATCH --gres=gpu:a40:1`
- `#SBATCH --nodes=1`
- `module purge`
- `module load python`

What it does:

1. sources [cluster/hpc_env.sh](/home/reese/llms/final_project/UTN-3-LLM-Final-Project/A1_prefix_tuning/cluster/hpc_env.sh#L1)
2. binds the repo to `/workspace`
3. runs preprocessing inside the container
4. runs single-GPU training inside the container with `accelerate launch --num_processes 1`

## Where Things Are Saved

With the HPC config:

- downloaded model/tokenizer cache: `${PREFIX_TUNING_MODEL_CACHE_DIR}`
- dataset cache: `${PREFIX_TUNING_DATASET_CACHE_DIR}`
- generated `train.jsonl`, `valid.jsonl`, `test.jsonl`: `${PREFIX_TUNING_DATA_DIR}`
- checkpoints and adapters: `${PREFIX_TUNING_OUTPUT_DIR}`
- optional exported final adapter copy: `${PREFIX_TUNING_RESULTS_EXPORT_DIR}`

The trained adapter is not merged into the base model. The main saved outputs are:

- `best_adapter/`
- `final_adapter/`
- `checkpoints/step-*`

under `${PREFIX_TUNING_OUTPUT_DIR}`.

## Local Non-HPC Usage

If you are not on the cluster, you can still run the module locally with [config.yaml](/home/reese/llms/final_project/UTN-3-LLM-Final-Project/A1_prefix_tuning/config.yaml#L1):

```bash
python A1_prefix_tuning/scripts/preprocess.py --config A1_prefix_tuning/config.yaml
accelerate launch --num_processes 1 A1_prefix_tuning/scripts/train.py --config A1_prefix_tuning/config.yaml
python A1_prefix_tuning/scripts/evaluate.py --config A1_prefix_tuning/config.yaml
```

Or run the convenience wrapper:

```bash
python A1_prefix_tuning/run_pipeline.py --config A1_prefix_tuning/config.yaml
```

## Notes

- `scripts/train.py` requires CUDA and exits immediately if PyTorch cannot see a GPU.
- The training code uses `tokenizer.apply_chat_template(...)` so formatting matches Qwen chat inference.
- Prefix tuning keeps the Qwen base weights frozen and trains only the prefix parameters.
- If you already have a local snapshot of Qwen, set `model.name_or_path` accordingly and optionally set `model.local_files_only: true`.
