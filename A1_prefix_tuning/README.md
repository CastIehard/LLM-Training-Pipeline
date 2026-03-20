# A1 Prefix Tuning

This module trains a prefix-tuned adapter for `Qwen/Qwen3-0.6B` with Hugging Face Transformers, PEFT, and Accelerate.

## Pipeline

1. `scripts/preprocess.py`
   - Reads `../data/llm_qna.jsonl`
   - Groups rows by hash to avoid split leakage
   - Writes chat-formatted `train.jsonl`, `valid.jsonl`, and `test.jsonl`
2. `scripts/train.py`
   - Loads the base Qwen3 model and tokenizer
   - Wraps the frozen base model with `PrefixTuningConfig`
   - Runs distributed training with `Accelerator`
   - Saves `best_adapter`, `final_adapter`, and rolling checkpoints
3. `scripts/evaluate.py`
   - Reloads the base model plus the saved prefix adapter
   - Computes loss and perplexity on the configured split
   - Writes sample generations to `outputs/.../evaluation.json`
4. `scripts/infer.py`
   - Loads the saved adapter for an interactive prompt

## Install

Use the repo environment, then add the missing training packages if needed:

```bash
pip install datasets peft
```

## Local Run

Preprocess, train, and evaluate on a single machine:

```bash
python A1_prefix_tuning/run_pipeline.py --config A1_prefix_tuning/config.yaml
```

You can also run the stages directly:

```bash
python A1_prefix_tuning/scripts/preprocess.py --config A1_prefix_tuning/config.yaml
accelerate launch --num_processes 1 A1_prefix_tuning/scripts/train.py --config A1_prefix_tuning/config.yaml
python A1_prefix_tuning/scripts/evaluate.py --config A1_prefix_tuning/config.yaml
python A1_prefix_tuning/scripts/infer.py --config A1_prefix_tuning/config.yaml --prompt "What support exists for international students at UTN?"
```

## Multi-Node Cluster

### Preconditions

- Every node must see the same code and dataset paths, either through a shared filesystem or by copying the repo to identical paths.
- The same Python environment must exist on every node.
- `num_processes` must match the total GPU count across all nodes.
- `main_process_ip` should be the private IP of rank 0.

These launch requirements follow the current Accelerate multi-node docs:
- https://huggingface.co/docs/accelerate/basic_tutorials/launch
- https://huggingface.co/docs/accelerate/package_reference/launchers

### Manual launch on 2 nodes with 4 GPUs each

Run this on node 0:

```bash
accelerate launch \
  --num_machines 2 \
  --machine_rank 0 \
  --main_process_ip 10.0.0.1 \
  --main_process_port 29500 \
  --mixed_precision bf16 \
  --num_processes 8 \
  A1_prefix_tuning/scripts/train.py \
  --config A1_prefix_tuning/config.yaml
```

Run the same command on node 1, changing only `--machine_rank 1`.

### Launch with an Accelerate config file

Start from [`cluster/accelerate_multinode.yaml`](/home/reese/llms/final_project/UTN-3-LLM-Final-Project/A1_prefix_tuning/cluster/accelerate_multinode.yaml), copy it once per node, and set:

- `num_machines`
- `num_processes`
- `main_process_ip`
- `main_process_port`
- `machine_rank`

Then run on each node:

```bash
accelerate launch \
  --config_file A1_prefix_tuning/cluster/accelerate_multinode.yaml \
  A1_prefix_tuning/scripts/train.py \
  --config A1_prefix_tuning/config.yaml
```

### Launch on SLURM

Use [`cluster/slurm_multinode.sh`](/home/reese/llms/final_project/UTN-3-LLM-Final-Project/A1_prefix_tuning/cluster/slurm_multinode.sh) inside a SLURM allocation:

```bash
sbatch --nodes=2 --gres=gpu:4 your_job_script.sh
```

Inside `your_job_script.sh`, call:

```bash
bash A1_prefix_tuning/cluster/slurm_multinode.sh
```

The helper script derives:

- `MASTER_ADDR` from the first host in `SLURM_JOB_NODELIST`
- `machine_rank` from `SLURM_NODEID`
- total `num_processes` from `SLURM_NNODES * GPUS_PER_NODE`

## Key Config Knobs

- `model.name_or_path`: use `Qwen/Qwen3-0.6B` or a local snapshot path
- `data.max_seq_length`: truncation length for prompt + answer
- `training.learning_rate`: prefix tuning usually tolerates a higher LR than full fine-tuning
- `training.gradient_accumulation_steps`: increase this before shrinking sequence length
- `peft.num_virtual_tokens`: controls the learned prefix size
- `peft.prefix_projection`: adds a projection MLP for the prefix encoder

## Notes

- PEFT prefix tuning keeps the base Qwen weights frozen and only trains the prefix parameters.
- The implementation uses `tokenizer.apply_chat_template(...)` so the training format matches Qwen chat inference.
- If you already have a local model snapshot, set `model.name_or_path` to that path and optionally set `model.local_files_only: true`.

