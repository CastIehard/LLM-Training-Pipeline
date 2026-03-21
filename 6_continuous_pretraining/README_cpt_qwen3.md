# Qwen3 continued pretraining on raw markdown/text files

This package contains a standalone script for continued pretraining (CPT) of `Qwen/Qwen3-0.6B` on a folder of raw `.md`, `.markdown`, or `.txt` files.

## Files

- `cpt_qwen3_raw_text.py`: main training script

## What it does

- recursively scans a corpus directory for text files
- splits by file into train and validation sets to reduce leakage
- applies conservative cleanup for obvious markdown/navigation boilerplate
- tokenizes documents and appends EOS between documents
- packs tokens into fixed-size blocks for causal language modeling
- trains with Hugging Face `Trainer`
- evaluates once per epoch
- logs to TensorBoard
- saves checkpoints and a final model folder

## Recommended environment

Python 3.11+ on Linux with CUDA.

Suggested packages:

```bash
pip install -U \
  "torch>=2.6" \
  "transformers>=4.51.0" \
  "datasets>=3.0.0" \
  "accelerate>=1.0.0" \
  "tensorboard>=2.14.0" \
  "sentencepiece>=0.2.0"
```

Optional for better attention performance:

```bash
pip install flash-attn --no-build-isolation
```

Optional if you want an 8-bit optimizer instead of fused AdamW:

```bash
pip install bitsandbytes>=0.45.0
```

## Example run

```bash
python cpt_qwen3_raw_text.py \
  --corpus_dir /path/to/corpus \
  --output_dir /path/to/runs/qwen3_0_6b_cpt \
  --model_name_or_path Qwen/Qwen3-0.6B \
  --block_size 2048 \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 2 \
  --gradient_accumulation_steps 16 \
  --num_train_epochs 1 \
  --learning_rate 5e-5 \
  --gradient_checkpointing
```

## Where logs and outputs go

Inside `output_dir` the script writes:

- `tb_logs/`: TensorBoard logs
- `checkpoint-*`: trainer checkpoints
- `final_model/`: final model and tokenizer
- `split_manifest.json`: exact train/validation file split
- `run_summary.json`: block counts and final metrics

## TensorBoard

```bash
tensorboard --logdir /path/to/runs/qwen3_0_6b_cpt/tb_logs
```

## Notes on the defaults

The defaults are intentionally conservative for a single 16 GB GPU.

If you hit OOM:

- reduce `--per_device_train_batch_size` to `1`
- reduce `--block_size` to `1024`
- keep `--gradient_checkpointing` enabled
- use `--attn_implementation sdpa` if `flash_attention_2` is not available

If the run is very stable and memory headroom is available:

- try `--per_device_train_batch_size 4`
- reduce `--gradient_accumulation_steps`
- keep `--block_size 2048`

## About `/no_think`

Do not inject `/no_think` into raw-text continued pretraining data.

That control is useful later for chat or SFT-style data formatting and inference behavior. For raw corpus CPT you generally want plain text, not synthetic prompt suffixes.
