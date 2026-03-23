#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import json
import logging
import math
import os
import random
import re
import sys
import unicodedata
from dataclasses import asdict, dataclass
from itertools import chain
from pathlib import Path
from typing import Iterator, Sequence

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

LOG = logging.getLogger("qwen3_cpt")

SUPPORTED_EXTENSIONS = {".md", ".markdown", ".txt"}
DEFAULT_DROP_LINE_PATTERNS = (
    r"^\s*Menü\s*$",
    r"^\s*Suche\s*$",
    r"^\s*weiter\s*$",
    r"^\s*Sitemap\s*$",
    r"^\s*[×✕✖]\s*$",
    r"^\s*\*\[\]:.*$",
    r"^\s*\[\]\([^)]*\)\s*$",
)
DEFAULT_DROP_LINE_REGEX = [re.compile(p, re.IGNORECASE) for p in DEFAULT_DROP_LINE_PATTERNS]
ADJACENT_DUPLICATE_HEADING_RE = re.compile(r"^(#{1,6}\s+.+)$")
NUMERIC_NAV_LINE_RE = re.compile(r"^\s*\d+[.)]?\s+.+$")


@dataclass(frozen=True)
class Config:
    corpus_dir: str
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
    min_chars: int
    max_files: int | None
    preprocessing_num_workers: int
    dataloader_num_workers: int
    attn_implementation: str
    gradient_checkpointing: bool
    trust_remote_code: bool
    resume_from_checkpoint: str | None
    overwrite_output_dir: bool
    disable_default_boilerplate_filter: bool
    extra_drop_line_regex: tuple[str, ...]
    torch_compile: bool
    lr_scheduler_type: str


class EvalPerplexityCallback(TrainerCallback):
    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, metrics: dict[str, float] | None = None, **kwargs, ) -> None:
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
    parser = argparse.ArgumentParser(description="Continued pretraining for Qwen/Qwen3-0.6B on a folder of raw markdown/text files.")
    parser.add_argument("--corpus_dir", type=str, required=True, help="Folder containing .md/.markdown/.txt files.")
    parser.add_argument("--output_dir", type=str, required=True, help="Where checkpoints, logs and the final model are written.")
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen3-0.6B", help="HF model id or local model path. Use the BASE model, not an instruct checkpoint.", )
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--block_size", type=int, default=1024)
    parser.add_argument("--validation_ratio", type=float, default=0.1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--learning_rate", type=float, default=2e-6)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.0)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_chars", type=int, default=200)
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--preprocessing_num_workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument("--dataloader_num_workers", type=int, default=min(8, os.cpu_count() or 2))
    parser.add_argument("--attn_implementation", type=str, default="sdpa", choices=["flash_attention_2", "sdpa", "eager"],)
    parser.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--overwrite_output_dir", action="store_true")
    parser.add_argument("--disable_default_boilerplate_filter", action="store_true")
    parser.add_argument("--extra_drop_line_regex", type=str, nargs="*", default=(), help="Additional regex patterns for dropping obvious boilerplate lines.", )
    parser.add_argument("--torch_compile", action="store_true")
    parser.add_argument("--lr_scheduler_type", type=str, default="constant", choices=["linear", "cosine", "constant", "constant_with_warmup"])

    args = parser.parse_args()
    if not 0.0 < args.validation_ratio < 0.5:
        parser.error("--validation_ratio must be between 0 and 0.5")
    if args.block_size < 128:
        parser.error("--block_size must be at least 128")
    if args.min_chars < 1:
        parser.error("--min_chars must be positive")
    if not 0.0 <= args.warmup_ratio < 1.0:
        parser.error("--warmup_ratio must be in [0, 1)")

    return Config(**vars(args))


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s", handlers=[logging.StreamHandler(sys.stdout)], )


def list_text_files(corpus_dir: Path, max_files: int | None) -> list[Path]:
    if not corpus_dir.exists():
        raise FileNotFoundError(f"Corpus directory does not exist: {corpus_dir}")
    if not corpus_dir.is_dir():
        raise NotADirectoryError(f"Corpus path is not a directory: {corpus_dir}")

    files = [path for path in corpus_dir.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS]
    files.sort()
    if max_files is not None:
        files = files[:max_files]
    if not files:
        raise RuntimeError(f"No supported text files found under {corpus_dir}")
    return files


def normalize_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\ufeff", "")
    return text


def clean_document(text: str, drop_regexes: Sequence[re.Pattern[str]], min_chars: int) -> str | None:
    text = normalize_text(text)
    raw_lines = text.split("\n")

    cleaned_lines: list[str] = []
    prev_heading: str | None = None
    numeric_nav_streak = 0

    for raw_line in raw_lines:
        line = raw_line.strip()

        if any(regex.match(line) for regex in drop_regexes):
            continue

        if line.startswith("http://") or line.startswith("https://"):
            continue

        if line.startswith("[](") and line.endswith(")"):
            continue

        if not line:
            cleaned_lines.append("")
            numeric_nav_streak = 0
            prev_heading = None
            continue

        if line.count("#") > 8:
            continue

        if NUMERIC_NAV_LINE_RE.match(line):
            numeric_nav_streak += 1
        else:
            numeric_nav_streak = 0

        if numeric_nav_streak >= 12:
            continue

        heading_match = ADJACENT_DUPLICATE_HEADING_RE.match(line)
        if heading_match:
            heading = heading_match.group(1)
            if heading == prev_heading:
                continue
            prev_heading = heading
        else:
            prev_heading = None

        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = text.strip()

    if len(text) < min_chars:
        return None

    alpha_chars = sum(ch.isalpha() for ch in text)
    if alpha_chars < max(40, len(text) // 20):
        return None

    return text


def read_cleaned_document(path: str, drop_regexes: Sequence[re.Pattern[str]], min_chars: int) -> dict[str, str] | None:
    try:
        text = Path(path).read_text(encoding="utf-8", errors="ignore")
    except OSError as exc:
        LOG.warning("Skipping unreadable file %s: %s", path, exc)
        return None

    cleaned = clean_document(text=text, drop_regexes=drop_regexes, min_chars=min_chars)
    if cleaned is None:
        return None
    return {"text": cleaned, "source": path}


def dataset_generator(paths: Sequence[str], drop_patterns: Sequence[str], min_chars: int) -> Iterator[dict[str, str]]:
    drop_regexes = [re.compile(p, re.IGNORECASE) for p in drop_patterns]
    for path in paths:
        item = read_cleaned_document(path, drop_regexes=drop_regexes, min_chars=min_chars)
        if item is not None:
            yield item


def make_dataset(paths: Sequence[str], config: Config) -> Dataset:
    patterns = tuple(DEFAULT_DROP_LINE_PATTERNS if not config.disable_default_boilerplate_filter else ())
    patterns = patterns + tuple(config.extra_drop_line_regex)
    return Dataset.from_generator(
        dataset_generator,
        gen_kwargs={
            "paths": list(paths),
            "drop_patterns": patterns,
            "min_chars": config.min_chars,
        },
        cache_dir=config.cache_dir,
    )


def split_files(files: Sequence[Path], validation_ratio: float, seed: int) -> tuple[list[str], list[str]]:
    file_strings = [str(path) for path in files]
    rng = random.Random(seed)
    rng.shuffle(file_strings)

    val_count = max(1, int(len(file_strings) * validation_ratio))
    if val_count >= len(file_strings):
        val_count = max(1, len(file_strings) - 1)
    val_files = file_strings[:val_count]
    train_files = file_strings[val_count:]

    if not train_files:
        raise RuntimeError("Training split is empty after applying validation_ratio")
    return train_files, val_files


def tokenize_documents(batch: dict[str, list[str]], tokenizer: AutoTokenizer) -> dict[str, list[list[int]]]:
    texts: list[str] = []
    eos = tokenizer.eos_token or ""
    for text in batch["text"]:
        if eos and not text.endswith(eos):
            texts.append(text + eos)
        else:
            texts.append(text)
    return tokenizer(
        texts,
        add_special_tokens=False,
        truncation=False,
        padding=False,
    )


def group_texts(examples: dict[str, list[list[int]]], block_size: int) -> dict[str, list[list[int]]]:
    concatenated_examples = {key: list(chain.from_iterable(examples[key])) for key in examples.keys()}
    total_length = len(concatenated_examples["input_ids"])
    total_length = (total_length // block_size) * block_size
    if total_length == 0:
        return {"input_ids": [], "attention_mask": [], "labels": []}

    result = {
        key: [tokens[i: i + block_size] for i in range(0, total_length, block_size)]
        for key, tokens in concatenated_examples.items()
    }
    result["labels"] = [ids.copy() for ids in result["input_ids"]]
    return result


def save_split_manifest(output_dir: Path, train_files: Sequence[str], val_files: Sequence[str], config: Config) -> None:
    manifest = {
        "config": asdict(config),
        "train_files": list(train_files),
        "validation_files": list(val_files),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "split_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


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


def main() -> None:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    setup_logging()
    config = parse_args()
    set_seed(config.seed)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_tensorboard_logging(output_dir)

    files = list_text_files(Path(config.corpus_dir), max_files=config.max_files)
    train_files, val_files = split_files(files=files, validation_ratio=config.validation_ratio, seed=config.seed)
    save_split_manifest(output_dir=output_dir, train_files=train_files, val_files=val_files, config=config)

    LOG.info("Found %s candidate files", len(files))
    LOG.info("Split into %s training files and %s validation files", len(train_files), len(val_files))

    if config.cache_dir:
        Path(config.cache_dir).mkdir(parents=True, exist_ok=True)

    train_raw = make_dataset(train_files, config=config)
    val_raw = make_dataset(val_files, config=config)

    if len(train_raw) == 0:
        raise RuntimeError("No training documents survived cleaning/filtering. Lower --min_chars or relax filters.")
    if len(val_raw) == 0:
        raise RuntimeError("No validation documents survived cleaning/filtering. Lower --min_chars or relax filters.")

    LOG.info("After cleaning: %s training documents, %s validation documents", len(train_raw), len(val_raw))

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name_or_path,
        cache_dir=config.cache_dir,
        trust_remote_code=config.trust_remote_code,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenized_train = train_raw.map(
        lambda batch: tokenize_documents(batch, tokenizer),
        batched=True,
        num_proc=max(1, config.preprocessing_num_workers),
        remove_columns=train_raw.column_names,
        desc="Tokenizing training documents",
    )
    tokenized_val = val_raw.map(
        lambda batch: tokenize_documents(batch, tokenizer),
        batched=True,
        num_proc=max(1, config.preprocessing_num_workers),
        remove_columns=val_raw.column_names,
        desc="Tokenizing validation documents",
    )

    lm_train = tokenized_train.map(
        lambda batch: group_texts(batch, config.block_size),
        batched=True,
        num_proc=max(1, config.preprocessing_num_workers),
        desc="Packing training tokens into fixed blocks",
    )
    lm_val = tokenized_val.map(
        lambda batch: group_texts(batch, config.block_size),
        batched=True,
        num_proc=max(1, config.preprocessing_num_workers),
        desc="Packing validation tokens into fixed blocks",
    )

    lm_train = lm_train.filter(lambda row: len(row["input_ids"]) == config.block_size, desc="Filtering short train blocks")
    lm_val = lm_val.filter(lambda row: len(row["input_ids"]) == config.block_size, desc="Filtering short val blocks")

    if len(lm_train) == 0 or len(lm_val) == 0:
        raise RuntimeError("No packed training blocks were created. Lower --block_size or add more data.")

    LOG.info("Packed into %s training blocks and %s validation blocks", len(lm_train), len(lm_val))

    total_update_steps = estimate_total_update_steps(config=config, num_train_examples=len(lm_train))
    warmup_steps = compute_warmup_steps(config=config, total_update_steps=total_update_steps)
    LOG.info("Estimated %s optimizer update steps with %s warmup steps", total_update_steps, warmup_steps)

    torch_dtype = torch.bfloat16
    attn_impl = config.attn_implementation
    LOG.info("Using torch dtype %s and attention implementation %s", torch_dtype, attn_impl or "default")

    model_init_params = inspect.signature(AutoModelForCausalLM.from_pretrained).parameters
    model_kwargs: dict[str, object] = {
        "cache_dir": config.cache_dir,
        "trust_remote_code": config.trust_remote_code,
    }

    if "dtype" in model_init_params:
        model_kwargs["dtype"] = torch_dtype
    elif "torch_dtype" in model_init_params:
        model_kwargs["torch_dtype"] = torch_dtype

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

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_steps=warmup_steps,
        logging_steps=config.logging_steps,
        save_total_limit=config.save_total_limit,
        seed=config.seed,
        data_seed=config.seed,
        report_to=["tensorboard"],
        remove_unused_columns=False,
        dataloader_num_workers=config.dataloader_num_workers,
        dataloader_pin_memory=True,
        load_best_model_at_end=False,
        lr_scheduler_type=config.lr_scheduler_type,
        save_strategy="epoch",
        logging_strategy="steps",
        bf16=True,
        tf32=True,
        torch_compile=config.torch_compile,
    )

    data_collator = DefaultDataCollator()

    trainer_init_params = inspect.signature(Trainer.__init__).parameters
    trainer_kwargs: dict[str, object] = {
        "model": model,
        "args": training_args,
        "train_dataset": lm_train,
        "eval_dataset": lm_val,
        "data_collator": data_collator,
        "callbacks": [EvalPerplexityCallback()],
    }
    if "processing_class" in trainer_init_params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_init_params:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = Trainer(**trainer_kwargs)

    train_result = trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)
    trainer.save_model(str(output_dir / "final_model"))
    tokenizer.save_pretrained(str(output_dir / "final_model"))

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
        "num_input_files": len(files),
        "num_train_files": len(train_files),
        "num_val_files": len(val_files),
        "num_train_documents": len(train_raw),
        "num_val_documents": len(val_raw),
        "num_train_blocks": len(lm_train),
        "num_val_blocks": len(lm_val),
        "block_size": config.block_size,
        "dtype": str(torch_dtype),
        "attn_implementation": getattr(model.config, "_attn_implementation", attn_impl),
        "estimated_total_update_steps": total_update_steps,
        "warmup_steps": warmup_steps,
        "train_metrics": train_metrics,
        "eval_metrics": eval_metrics,
    }
    (output_dir / "run_summary.json").write_text(json.dumps(run_summary, indent=2, ensure_ascii=False), encoding="utf-8")

    LOG.info("Finished. Final model saved to %s", output_dir / "final_model")


if __name__ == "__main__":
    main()
