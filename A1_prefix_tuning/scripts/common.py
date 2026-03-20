from __future__ import annotations

import json
import math
import random
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
MODULE_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = MODULE_DIR.parent


def load_config(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_path(path_str: str | Path) -> Path:
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path
    return (MODULE_DIR / path).resolve()


def resolve_model_name_or_path(value: str) -> str:
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return str(candidate)

    module_relative = (MODULE_DIR / candidate).resolve()
    if module_relative.exists():
        return str(module_relative)

    project_relative = (PROJECT_ROOT / candidate).resolve()
    if project_relative.exists():
        return str(project_relative)

    return value


def get_split_paths(config: dict[str, Any]) -> dict[str, Path]:
    output_dir = resolve_path(config["data"]["output_dir"])
    return {
        "train": output_dir / "train.jsonl",
        "validation": output_dir / "valid.jsonl",
        "test": output_dir / "test.jsonl",
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def grouped_split(rows: list[dict[str, Any]], group_key: str, train_ratio: float, valid_ratio: float, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, row in enumerate(rows):
        key = str(row.get(group_key, index))
        groups[key].append(row)

    keys = list(groups)
    rng = random.Random(seed)
    shuffled_keys = list(keys)
    rng.shuffle(shuffled_keys)

    train_end = int(len(shuffled_keys) * train_ratio)
    valid_end = int(len(shuffled_keys) * (train_ratio + valid_ratio))

    train_keys = set(shuffled_keys[:train_end])
    valid_keys = set(shuffled_keys[train_end:valid_end])

    train_rows: list[dict[str, Any]] = []
    valid_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []

    for key, group_rows in groups.items():
        if key in train_keys:
            train_rows.extend(group_rows)
        elif key in valid_keys:
            valid_rows.extend(group_rows)
        else:
            test_rows.extend(group_rows)

    return train_rows, valid_rows, test_rows


def format_chat_example(row: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    data_config = config["data"]
    question = row.get(data_config["question_field"], row.get("user", ""))
    answer = row.get(data_config["answer_field"], row.get("assistant", ""))

    suffix = data_config["no_think_suffix"] if data_config.get("append_no_think_suffix", False) else ""
    messages = [
        {"role": "system", "content": data_config["system_prompt"]},
        {"role": "user", "content": f"{question}{suffix}"},
        {"role": "assistant", "content": answer},
    ]
    return {"messages": messages}


def tokenize_chat_example(example: dict[str, Any], tokenizer, max_seq_length: int) -> dict[str, Any]:
    messages = example["messages"]
    prompt_messages = messages[:-1]
    assistant_text = messages[-1]["content"].strip()

    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    full_text = prompt_text + assistant_text

    prompt_ids = tokenizer(
        prompt_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_seq_length,
    )["input_ids"]
    full = tokenizer(
        full_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_seq_length,
    )

    input_ids = full["input_ids"]
    attention_mask = full["attention_mask"]
    prompt_len = min(len(prompt_ids), len(input_ids))
    labels = [-100] * prompt_len + input_ids[prompt_len:]
    labels = labels[: len(input_ids)]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "seq_len": len(input_ids),
    }


@dataclass
class CausalLMCollator:
    tokenizer: Any
    pad_to_multiple_of: int | None = 8

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        max_len = max(len(item["input_ids"]) for item in features)
        if self.pad_to_multiple_of is not None:
            max_len = math.ceil(max_len / self.pad_to_multiple_of) * self.pad_to_multiple_of

        pad_id = self.tokenizer.pad_token_id
        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []

        for item in features:
            pad_len = max_len - len(item["input_ids"])
            batch_input_ids.append(item["input_ids"] + [pad_id] * pad_len)
            batch_attention_mask.append(item["attention_mask"] + [0] * pad_len)
            batch_labels.append(item["labels"] + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }


def parse_torch_dtype(value: str) -> torch.dtype:
    import torch

    normalized = value.lower()
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported torch dtype: {value}")
    return mapping[normalized]


def save_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def rotate_checkpoints(checkpoint_root: Path, keep_last: int) -> None:
    if keep_last <= 0 or not checkpoint_root.exists():
        return

    checkpoints = sorted(
        (path for path in checkpoint_root.iterdir() if path.is_dir() and path.name.startswith("step-")),
        key=lambda path: int(path.name.split("-")[-1]),
    )
    while len(checkpoints) > keep_last:
        old_checkpoint = checkpoints.pop(0)
        shutil.rmtree(old_checkpoint, ignore_errors=True)


def trainable_parameter_summary(model) -> dict[str, int]:
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    return {"trainable": trainable, "total": total}
