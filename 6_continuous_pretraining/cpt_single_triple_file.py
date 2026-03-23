#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import logging
import math
import os
import random
import re
import sys
import unicodedata
from collections import defaultdict
from dataclasses import asdict, dataclass
from itertools import chain
from pathlib import Path
from typing import Sequence

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DefaultDataCollator,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
    set_seed,
)

LOG = logging.getLogger("single_file_cpt_triples")
TRIPLE_SEPARATOR_RE = re.compile(r"\s*\|\s*")
MULTISPACE_RE = re.compile(r"[ \t]{2,}")


@dataclass(frozen=True)
class Config:
    input_file: str
    output_dir: str
    model_name_or_path: str
    cache_dir: str | None
    block_size: int
    validation_ratio: float
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    num_train_epochs: float
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    logging_steps: int
    save_total_limit: int
    seed: int
    min_line_chars: int
    max_lines: int | None
    tokenize_batch_size: int
    dataloader_num_workers: int
    attn_implementation: str
    gradient_checkpointing: bool
    trust_remote_code: bool
    resume_from_checkpoint: str | None
    torch_compile: bool
    lr_scheduler_type: str
    keep_exact_duplicates_together: bool
    require_two_pipes: bool


class EvalPerplexityCallback(TrainerCallback):
    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: dict[str, float] | None = None,
        **kwargs,
    ) -> None:
        if not metrics:
            return
        eval_loss = metrics.get("eval_loss")
        if eval_loss is None:
            return
        try:
            metrics["eval_perplexity"] = math.exp(eval_loss)
        except OverflowError:
            metrics["eval_perplexity"] = float("inf")


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Continued pretraining from a single file containing one triple per line, e.g. 'subject | relation | object'."
    )
    parser.add_argument("--input_file", type=str, required=True, help="Single .md/.txt file with one triple per line.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for checkpoints, logs, and the final model.")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="HF model id or local model path. Use the base model, not an instruct checkpoint.",
    )
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--block_size", type=int, default=1024)
    parser.add_argument("--validation_ratio", type=float, default=0.1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--num_train_epochs", type=float, default=5.0)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.0)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_line_chars", type=int, default=3, help="Drop lines shorter than this after normalization.")
    parser.add_argument("--max_lines", type=int, default=None, help="Optional cap for quick debugging.")
    parser.add_argument("--tokenize_batch_size", type=int, default=4096)
    parser.add_argument("--dataloader_num_workers", type=int, default=min(8, os.cpu_count() or 2))
    parser.add_argument("--attn_implementation", type=str, default="sdpa", choices=["flash_attention_2", "sdpa", "eager"],)
    parser.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--torch_compile", action="store_true")
    parser.add_argument("--lr_scheduler_type", type=str, default="constant", choices=["linear", "cosine", "constant", "constant_with_warmup"],)
    parser.add_argument("--keep_exact_duplicates_together", action=argparse.BooleanOptionalAction, default=True, help="Keep identical normalized lines in the same split to reduce train/val leakage.",)
    parser.add_argument("--require_two_pipes", action=argparse.BooleanOptionalAction, default=True, help="Drop non-empty lines that do not look like 'subject | relation | object'.",)

    args = parser.parse_args()

    if not 0.0 < args.validation_ratio < 0.5:
        parser.error("--validation_ratio must be between 0 and 0.5")
    if args.block_size < 128:
        parser.error("--block_size must be at least 128")
    if args.min_line_chars < 1:
        parser.error("--min_line_chars must be positive")
    if args.tokenize_batch_size < 1:
        parser.error("--tokenize_batch_size must be positive")
    if not 0.0 <= args.warmup_ratio < 1.0:
        parser.error("--warmup_ratio must be in [0, 1)")

    return Config(**vars(args))


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def normalize_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\ufeff", "")
    return text


def clean_triple_line(line: str, *, min_line_chars: int, require_two_pipes: bool) -> str | None:
    line = normalize_text(line).strip()
    if not line:
        return None
    if line.startswith("#"):
        return None

    line = TRIPLE_SEPARATOR_RE.sub(" | ", line)
    line = MULTISPACE_RE.sub(" ", line).strip()

    if require_two_pipes and line.count("|") < 2:
        return None
    if len(line) < min_line_chars:
        return None
    return line


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def load_cleaned_lines(path: Path, config: Config) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")
    if not path.is_file():
        raise NotADirectoryError(f"Input path is not a file: {path}")

    try:
        raw_text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError as exc:
        raise RuntimeError(f"Could not read input file {path}: {exc}") from exc

    cleaned: list[str] = []
    for raw_line in normalize_text(raw_text).split("\n"):
        line = clean_triple_line(
            raw_line,
            min_line_chars=config.min_line_chars,
            require_two_pipes=config.require_two_pipes,
        )
        if line is None:
            continue
        cleaned.append(line)
        if config.max_lines is not None and len(cleaned) >= config.max_lines:
            break

    if not cleaned:
        raise RuntimeError("No usable triple lines were found. Check the file format or relax the filters.")
    return cleaned


def split_lines(lines: Sequence[str], validation_ratio: float, seed: int, keep_exact_duplicates_together: bool) -> tuple[list[str], list[str], dict[str, int]]:
    rng = random.Random(seed)

    if keep_exact_duplicates_together:
        groups: dict[str, list[int]] = defaultdict(list)
        for idx, line in enumerate(lines):
            groups[line].append(idx)

        unique_lines = list(groups.keys())
        rng.shuffle(unique_lines)

        val_group_count = max(1, int(len(unique_lines) * validation_ratio))
        if val_group_count >= len(unique_lines):
            val_group_count = max(1, len(unique_lines) - 1)

        val_groups = set(unique_lines[:val_group_count])
        train_lines = [line for line in lines if line not in val_groups]
        val_lines = [line for line in lines if line in val_groups]
        split_stats = {
            "num_total_lines": len(lines),
            "num_unique_lines": len(unique_lines),
            "num_train_lines": len(train_lines),
            "num_val_lines": len(val_lines),
            "num_train_unique_lines": len({line for line in train_lines}),
            "num_val_unique_lines": len({line for line in val_lines}),
        }
    else:
        indices = list(range(len(lines)))
        rng.shuffle(indices)

        val_count = max(1, int(len(indices) * validation_ratio))
        if val_count >= len(indices):
            val_count = max(1, len(indices) - 1)

        val_idx = set(indices[:val_count])
        train_lines = [line for i, line in enumerate(lines) if i not in val_idx]
        val_lines = [line for i, line in enumerate(lines) if i in val_idx]
        split_stats = {
            "num_total_lines": len(lines),
            "num_unique_lines": len(set(lines)),
            "num_train_lines": len(train_lines),
            "num_val_lines": len(val_lines),
            "num_train_unique_lines": len(set(train_lines)),
            "num_val_unique_lines": len(set(val_lines)),
        }

    if not train_lines:
        raise RuntimeError("Training split is empty after applying validation_ratio")
    if not val_lines:
        raise RuntimeError("Validation split is empty after applying validation_ratio")

    rng.shuffle(train_lines)
    rng.shuffle(val_lines)
    return train_lines, val_lines, split_stats


def save_line_split(output_dir: Path, train_lines: Sequence[str], val_lines: Sequence[str], config: Config, input_file: Path, split_stats: dict[str, int]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "train_lines.txt").write_text("\n".join(train_lines) + "\n", encoding="utf-8")
    (output_dir / "validation_lines.txt").write_text("\n".join(val_lines) + "\n", encoding="utf-8")

    manifest = {
        "config": asdict(config),
        "input_file": str(input_file),
        "input_file_sha256": file_sha256(input_file),
        "split_stats": split_stats,
    }
    (output_dir / "split_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def tokenize_and_pack(lines: Sequence[str], tokenizer: AutoTokenizer, block_size: int, tokenize_batch_size: int, split_name: str) -> tuple[Dataset, dict[str, int]]:
    eos = tokenizer.eos_token or ""
    formatted_lines = [f"{line}\n{eos}" if eos else f"{line}\n" for line in lines]

    all_input_ids: list[int] = []
    for start in range(0, len(formatted_lines), tokenize_batch_size):
        batch_texts = formatted_lines[start: start + tokenize_batch_size]
        tokenized = tokenizer(
            batch_texts,
            add_special_tokens=False,
            truncation=False,
            padding=False,
        )
        all_input_ids.extend(chain.from_iterable(tokenized["input_ids"]))

    total_tokens_before_packing = len(all_input_ids)
    total_tokens_after_packing = (total_tokens_before_packing // block_size) * block_size
    if total_tokens_after_packing == 0:
        raise RuntimeError(
            f"No {split_name} blocks were created. Lower --block_size or provide more lines for that split."
        )

    all_input_ids = all_input_ids[:total_tokens_after_packing]
    input_ids = [all_input_ids[i: i + block_size] for i in range(0, total_tokens_after_packing, block_size)]
    attention_mask = [[1] * block_size for _ in range(len(input_ids))]
    labels = [ids.copy() for ids in input_ids]

    dataset = Dataset.from_dict(
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
    )
    stats = {
        "num_lines": len(lines),
        "num_tokens_before_packing": total_tokens_before_packing,
        "num_tokens_after_packing": total_tokens_after_packing,
        "num_blocks": len(input_ids),
    }
    return dataset, stats


def estimate_total_update_steps(config: Config, num_train_examples: int) -> int:
    world_size = max(1, int(os.environ.get("WORLD_SIZE", "1")))
    micro_batches_per_epoch = math.ceil(num_train_examples / max(1, config.per_device_train_batch_size * world_size))
    updates_per_epoch = math.ceil(micro_batches_per_epoch / max(1, config.gradient_accumulation_steps))
    total_update_steps = math.ceil(updates_per_epoch * config.num_train_epochs)
    return max(1, total_update_steps)


def compute_warmup_steps(config: Config, total_update_steps: int) -> int:
    if config.warmup_ratio <= 0.0:
        return 0
    warmup_steps = int(round(total_update_steps * config.warmup_ratio))
    if warmup_steps == 0:
        warmup_steps = 1
    return min(warmup_steps, total_update_steps - 1) if total_update_steps > 1 else warmup_steps


def configure_tensorboard_logging(output_dir: Path) -> Path:
    tb_dir = output_dir / "tb_logs"
    tb_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TENSORBOARD_LOGGING_DIR"] = str(tb_dir)
    return tb_dir


def build_training_args(config: Config, output_dir: Path, warmup_steps: int, bf16: bool, fp16: bool) -> TrainingArguments:
    params = inspect.signature(TrainingArguments.__init__).parameters
    kwargs: dict[str, object] = {
        "output_dir": str(output_dir),
        "num_train_epochs": config.num_train_epochs,
        "per_device_train_batch_size": config.per_device_train_batch_size,
        "per_device_eval_batch_size": config.per_device_eval_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "warmup_steps": warmup_steps,
        "logging_steps": config.logging_steps,
        "save_total_limit": config.save_total_limit,
        "seed": config.seed,
        "data_seed": config.seed,
        "report_to": ["tensorboard"],
        "remove_unused_columns": False,
        "dataloader_num_workers": config.dataloader_num_workers,
        "dataloader_pin_memory": True,
        "load_best_model_at_end": False,
        "lr_scheduler_type": config.lr_scheduler_type,
        "save_strategy": "no",
        "logging_strategy": "steps",
        "bf16": bf16,
        "fp16": fp16,
    }

    if "evaluation_strategy" in params:
        kwargs["evaluation_strategy"] = "epoch"
    if "eval_strategy" in params:
        kwargs["eval_strategy"] = "epoch"
    if "tf32" in params:
        kwargs["tf32"] = True
    if "torch_compile" in params:
        kwargs["torch_compile"] = config.torch_compile

    return TrainingArguments(**kwargs)


def main() -> None:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    setup_logging()
    config = parse_args()
    set_seed(config.seed)

    input_file = Path(config.input_file)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_tensorboard_logging(output_dir)

    if config.cache_dir:
        Path(config.cache_dir).mkdir(parents=True, exist_ok=True)

    lines = load_cleaned_lines(input_file, config)
    train_lines, val_lines, split_stats = split_lines(
        lines,
        validation_ratio=config.validation_ratio,
        seed=config.seed,
        keep_exact_duplicates_together=config.keep_exact_duplicates_together,
    )
    save_line_split(output_dir, train_lines, val_lines, config, input_file, split_stats)

    LOG.info("Loaded %s usable lines from %s", len(lines), input_file)
    LOG.info("Split into %s training lines and %s validation lines", len(train_lines), len(val_lines))

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name_or_path,
        cache_dir=config.cache_dir,
        trust_remote_code=config.trust_remote_code,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    lm_train, train_stats = tokenize_and_pack(
        train_lines,
        tokenizer,
        block_size=config.block_size,
        tokenize_batch_size=config.tokenize_batch_size,
        split_name="training",
    )
    lm_val, val_stats = tokenize_and_pack(
        val_lines,
        tokenizer,
        block_size=config.block_size,
        tokenize_batch_size=config.tokenize_batch_size,
        split_name="validation",
    )

    LOG.info("Packed training split into %s blocks", train_stats["num_blocks"])
    LOG.info("Packed validation split into %s blocks", val_stats["num_blocks"])

    total_update_steps = estimate_total_update_steps(config=config, num_train_examples=len(lm_train))
    warmup_steps = compute_warmup_steps(config=config, total_update_steps=total_update_steps)
    LOG.info("Estimated %s optimizer update steps with %s warmup steps", total_update_steps, warmup_steps)

    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        runtime_dtype = torch.bfloat16
        bf16 = True
        fp16 = False
    elif torch.cuda.is_available():
        runtime_dtype = torch.float16
        bf16 = False
        fp16 = True
    else:
        runtime_dtype = torch.float32
        bf16 = False
        fp16 = False

    attn_impl = config.attn_implementation
    LOG.info("Using torch dtype %s and attention implementation %s", runtime_dtype, attn_impl or "default")

    model_init_params = inspect.signature(AutoModelForCausalLM.from_pretrained).parameters
    model_kwargs: dict[str, object] = {
        "cache_dir": config.cache_dir,
        "trust_remote_code": config.trust_remote_code,
    }
    if "dtype" in model_init_params:
        model_kwargs["dtype"] = runtime_dtype
    elif "torch_dtype" in model_init_params:
        model_kwargs["torch_dtype"] = runtime_dtype
    if attn_impl is not None and "attn_implementation" in model_init_params:
        model_kwargs["attn_implementation"] = attn_impl

    try:
        model = AutoModelForCausalLM.from_pretrained(config.model_name_or_path, **model_kwargs)
    except Exception as exc:
        if attn_impl == "flash_attention_2" and "attn_implementation" in model_kwargs:
            LOG.warning("flash_attention_2 load failed, retrying with sdpa: %s", exc)
            model_kwargs["attn_implementation"] = "sdpa"
            model = AutoModelForCausalLM.from_pretrained(config.model_name_or_path, **model_kwargs)
        else:
            raise

    model.config.use_cache = False
    if config.gradient_checkpointing:
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()

    training_args = build_training_args(
        config=config,
        output_dir=output_dir,
        warmup_steps=warmup_steps,
        bf16=bf16,
        fp16=fp16,
    )

    trainer_init_params = inspect.signature(Trainer.__init__).parameters
    trainer_kwargs: dict[str, object] = {
        "model": model,
        "args": training_args,
        "train_dataset": lm_train,
        "eval_dataset": lm_val,
        "data_collator": DefaultDataCollator(),
        "callbacks": [EvalPerplexityCallback()],
    }
    if "processing_class" in trainer_init_params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_init_params:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = Trainer(**trainer_kwargs)

    train_result = trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)
    final_model_dir = output_dir / "final_model"
    trainer.save_model(str(final_model_dir))
    tokenizer.save_pretrained(str(final_model_dir))

    train_metrics = dict(train_result.metrics)
    eval_metrics = trainer.evaluate()
    eval_loss = eval_metrics.get("eval_loss")
    if eval_loss is not None:
        try:
            eval_metrics["eval_perplexity"] = math.exp(eval_loss)
        except OverflowError:
            eval_metrics["eval_perplexity"] = float("inf")

    trainer.log_metrics("train", train_metrics)
    trainer.save_metrics("train", train_metrics)
    trainer.save_state()

    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    run_summary = {
        "model_name_or_path": config.model_name_or_path,
        "input_file": str(input_file),
        "input_file_sha256": file_sha256(input_file),
        "split_stats": split_stats,
        "train_pack_stats": train_stats,
        "val_pack_stats": val_stats,
        "block_size": config.block_size,
        "dtype": str(runtime_dtype),
        "attn_implementation": getattr(model.config, "_attn_implementation", attn_impl),
        "estimated_total_update_steps": total_update_steps,
        "warmup_steps": warmup_steps,
        "train_metrics": train_metrics,
        "eval_metrics": eval_metrics,
    }
    (output_dir / "run_summary.json").write_text(json.dumps(run_summary, indent=2, ensure_ascii=False), encoding="utf-8")

    LOG.info("Finished. Final model saved to %s", final_model_dir)


if __name__ == "__main__":
    main()
