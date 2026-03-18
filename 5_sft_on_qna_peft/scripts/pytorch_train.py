import math
import os
from dataclasses import dataclass
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

SCRIPT_DIR = Path(__file__).resolve().parent
RUN_DIR = SCRIPT_DIR.parent
PROJECT_DIR = RUN_DIR.parent

DATA_DIR = RUN_DIR / "data"
MODEL_PATH = PROJECT_DIR / "model" / "Qwen_Qwen3-0.6B"
OUTPUT_DIR = RUN_DIR / "adapters" / "domain_adapter_full_15000_hf"

SEED = 42

PER_DEVICE_TRAIN_BATCH_SIZE = 8
PER_DEVICE_EVAL_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 8

MAX_STEPS = 500
LEARNING_RATE = 5e-5
MAX_SEQ_LENGTH = 512

LOGGING_STEPS = 10
EVAL_STEPS = 100
SAVE_STEPS = 100
SAVE_TOTAL_LIMIT = 4

COMPILE_MODEL = True
GRADIENT_CHECKPOINTING = False
DATALOADER_NUM_WORKERS = 4

LIMIT_EVAL_SAMPLES = 50 * PER_DEVICE_EVAL_BATCH_SIZE


@dataclass
class CausalLMCollator:
    tokenizer: AutoTokenizer
    pad_to_multiple_of: int | None = 8

    def __call__(self, features):
        max_len = max(len(f["input_ids"]) for f in features)

        if self.pad_to_multiple_of is not None:
            max_len = ((max_len + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of) * self.pad_to_multiple_of

        pad_id = self.tokenizer.pad_token_id

        input_ids = []
        attention_mask = []
        labels = []

        for f in features:
            seq_len = len(f["input_ids"])
            pad_len = max_len - seq_len

            input_ids.append(f["input_ids"] + [pad_id] * pad_len)
            attention_mask.append(f["attention_mask"] + [0] * pad_len)
            labels.append(f["labels"] + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def get_latest_checkpoint(output_dir: str) -> str | None:
    output_dir_path = Path(output_dir)
    if not output_dir_path.exists():
        return None

    checkpoints = sorted(
        [p for p in output_dir_path.glob("checkpoint-*") if p.is_dir()],
        key=lambda p: int(p.name.split("-")[-1]),
    )
    return str(checkpoints[-1]) if checkpoints else None


def format_and_tokenize(example, tokenizer):
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
        max_length=MAX_SEQ_LENGTH,
    )["input_ids"]

    full = tokenizer(
        full_text,
        add_special_tokens=False,
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
    )

    input_ids = full["input_ids"]
    attention_mask = full["attention_mask"]

    prompt_len = min(len(prompt_ids), len(input_ids))
    labels = [-100] * prompt_len + input_ids[prompt_len:]
    labels = labels[:len(input_ids)]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "seq_len": len(input_ids),
    }


def print_example(dataset, tokenizer) -> None:
    sample = dataset[0]
    prompt_messages = sample["messages"][:-1]
    assistant_text = sample["messages"][-1]["content"].strip()

    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    print("=" * 80)
    print("Sample formatted example")
    print("=" * 80)
    print("PROMPT:")
    print(prompt_text[:800])
    print("-" * 80)
    print("ANSWER:")
    print(assistant_text[:400])
    print("=" * 80)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required, but no CUDA device was found.")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model path does not exist: {MODEL_PATH}")

    if not (MODEL_PATH / "config.json").exists():
        raise FileNotFoundError(f"config.json not found in model path: {MODEL_PATH}")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    set_seed(SEED)

    print(f"Loading tokenizer from: {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_PATH),
        use_fast=True,
        local_files_only=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading datasets...")
    raw_datasets = load_dataset(
        "json",
        data_files={
            "train": str(DATA_DIR / "train.jsonl"),
            "validation": str(DATA_DIR / "valid.jsonl"),
            "test": str(DATA_DIR / "test.jsonl"),
        },
    )

    if LIMIT_EVAL_SAMPLES is not None:
        eval_n = min(LIMIT_EVAL_SAMPLES, len(raw_datasets["validation"]))
        raw_datasets["validation"] = raw_datasets["validation"].select(range(eval_n))

    print_example(raw_datasets["train"], tokenizer)

    print("Tokenizing datasets...")
    tokenized = raw_datasets.map(
        lambda ex: format_and_tokenize(ex, tokenizer),
        remove_columns=["messages"],
        desc="Formatting chat + masking prompt",
    )

    tokenized["train"] = tokenized["train"].sort("seq_len")
    tokenized["validation"] = tokenized["validation"].sort("seq_len")
    tokenized["test"] = tokenized["test"].sort("seq_len")

    print(f"Loading model from: {MODEL_PATH}")
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_PATH),
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        local_files_only=True,
    )

    if COMPILE_MODEL:
        model = torch.compile(model)

    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.tie_word_embeddings = False

    if GRADIENT_CHECKPOINTING:
        model.gradient_checkpointing_enable()

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        max_steps=MAX_STEPS,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        gradient_checkpointing=GRADIENT_CHECKPOINTING,
        bf16=True,
        fp16=False,
        tf32=True,
        optim="adamw_torch_fused",
        lr_scheduler_type="constant",
        warmup_steps=0,
        logging_strategy="steps",
        logging_steps=LOGGING_STEPS,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        report_to="none",
        dataloader_num_workers=DATALOADER_NUM_WORKERS,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        label_names=["labels"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        processing_class=tokenizer,
        data_collator=CausalLMCollator(tokenizer),
    )

    resume_checkpoint = get_latest_checkpoint(str(OUTPUT_DIR))
    if resume_checkpoint:
        print(f"Resuming from checkpoint: {resume_checkpoint}")

    print("Starting training...")
    train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)

    print("Saving final model and tokenizer...")
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    train_metrics = train_result.metrics
    train_metrics["train_samples"] = len(tokenized["train"])
    trainer.log_metrics("train", train_metrics)
    trainer.save_metrics("train", train_metrics)
    trainer.save_state()

    print("Running validation...")
    eval_metrics = trainer.evaluate(eval_dataset=tokenized["validation"])
    eval_metrics["eval_samples"] = len(tokenized["validation"])
    try:
        eval_metrics["perplexity"] = math.exp(eval_metrics["eval_loss"])
    except Exception:
        pass
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    print("Running test evaluation...")
    test_metrics = trainer.evaluate(eval_dataset=tokenized["test"], metric_key_prefix="test")
    test_metrics["test_samples"] = len(tokenized["test"])
    try:
        test_metrics["test_perplexity"] = math.exp(test_metrics["test_loss"])
    except Exception:
        pass
    trainer.log_metrics("test", test_metrics)
    trainer.save_metrics("test", test_metrics)

    print("=" * 80)
    print("Training complete.")
    print(f"Final model saved to: {OUTPUT_DIR}")
    print("=" * 80)