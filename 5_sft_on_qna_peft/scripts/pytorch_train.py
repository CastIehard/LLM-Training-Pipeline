import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightning as L
import torch
from datasets import load_dataset
from lightning.pytorch.loggers import TensorBoardLogger
from peft import TaskType, get_peft_model, AdaLoraConfig, LoraConfig
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
RUN_DIR = SCRIPT_DIR.parent
PROJECT_DIR = RUN_DIR.parent

DATA_DIR = RUN_DIR / "data"
MODEL_PATH = PROJECT_DIR / "model" / "Qwen_Qwen3-0.6B"
ADAPTER_OUTPUT_DIR = RUN_DIR / "adapters" / "Qwen_Qwen3-0.6B_adalora_lightning"
TENSORBOARD_ROOT_DIR = RUN_DIR / "tb_logs"

MAX_EPOCHS: int = 15


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 42

    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 16

    learning_rate: float = 5e-5
    max_seq_length: int = 512

    val_check_interval: float = 0.5
    logging_steps: int = 10
    dataloader_num_workers: int = 15

    compile_model: bool = False
    run_name: str = "qwen3_adalora"


LORA_CONFIG = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=32,
    lora_alpha=64,
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)

ADALORA_CONFIG = AdaLoraConfig(
    task_type=TaskType.CAUSAL_LM,
    lora_alpha=64,
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    total_step=2116 * MAX_EPOCHS
)


@dataclass
class CausalLMCollator:
    tokenizer: AutoTokenizer
    pad_to_multiple_of: int | None = 8

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)

        if self.pad_to_multiple_of is not None:
            m = self.pad_to_multiple_of
            max_len = ((max_len + m - 1) // m) * m

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


def format_and_tokenize(example: dict[str, Any], tokenizer: AutoTokenizer, max_seq_length: int, ) -> dict[str, Any]:
    messages = example["messages"]

    prompt_messages = messages[:-1]
    assistant_text = messages[-1]["content"].strip()

    prompt_text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True, )
    full_text = prompt_text + assistant_text

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False, truncation=True, max_length=max_seq_length, )["input_ids"]

    full = tokenizer(full_text, add_special_tokens=False, truncation=True, max_length=max_seq_length, )

    input_ids = full["input_ids"]
    attention_mask = full["attention_mask"]

    prompt_len = min(len(prompt_ids), len(input_ids))
    labels = [-100] * prompt_len + input_ids[prompt_len:]
    labels = labels[: len(input_ids)]

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels, "seq_len": len(input_ids)}


def print_example(dataset, tokenizer: AutoTokenizer) -> None:
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


class ChatSFTDataModule(L.LightningDataModule):
    def __init__(self, cfg: TrainConfig, tokenizer: AutoTokenizer) -> None:
        super().__init__()
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.collator = CausalLMCollator(tokenizer)
        self._is_setup = False
        self._printed_example = False

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: str | None = None) -> None:
        if self._is_setup:
            return

        print("Loading datasets...")
        raw_datasets = load_dataset(
            "json",
            data_files={
                "train": str(DATA_DIR / "train.jsonl"),
                "validation": str(DATA_DIR / "valid.jsonl"),
                "test": str(DATA_DIR / "test.jsonl"),
            },
        )

        if not self._printed_example:
            print_example(raw_datasets["train"], self.tokenizer)
            self._printed_example = True

        remove_columns = raw_datasets["train"].column_names

        print("Tokenizing datasets...")
        tokenized = raw_datasets.map(
            lambda ex: format_and_tokenize(ex, self.tokenizer, self.cfg.max_seq_length),
            remove_columns=remove_columns,
            desc="Formatting chat + masking prompt",
        )

        tokenized["train"] = tokenized["train"].sort("seq_len")
        tokenized["validation"] = tokenized["validation"].sort("seq_len")
        tokenized["test"] = tokenized["test"].sort("seq_len")

        self.train_dataset = tokenized["train"]
        self.val_dataset = tokenized["validation"]
        self.test_dataset = tokenized["test"]

        self._is_setup = True

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.per_device_train_batch_size,
            shuffle=True,
            collate_fn=self.collator,
            num_workers=self.cfg.dataloader_num_workers,
            pin_memory=True,
            persistent_workers=self.cfg.dataloader_num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.per_device_eval_batch_size,
            shuffle=False,
            collate_fn=self.collator,
            num_workers=self.cfg.dataloader_num_workers,
            pin_memory=True,
            persistent_workers=self.cfg.dataloader_num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.cfg.per_device_eval_batch_size,
            shuffle=False,
            collate_fn=self.collator,
            num_workers=self.cfg.dataloader_num_workers,
            pin_memory=True,
            persistent_workers=self.cfg.dataloader_num_workers > 0,
        )


class LitQwenSFT(L.LightningModule):
    def __init__(self, cfg: TrainConfig, tokenizer: AutoTokenizer) -> None:
        super().__init__()
        self.cfg = cfg
        self.tokenizer = tokenizer

        print(f"Loading model from: {MODEL_PATH}")
        base_model = AutoModelForCausalLM.from_pretrained(str(MODEL_PATH), torch_dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True, )

        base_model.config.use_cache = False
        base_model.config.pad_token_id = tokenizer.pad_token_id

        self.model = get_peft_model(base_model, ADALORA_CONFIG)
        self.model.print_trainable_parameters()

    def forward(self, batch: dict[str, torch.Tensor]):
        return self.model(**batch)

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        outputs = self.model(**batch)
        loss = outputs.loss
        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True, logger=True, batch_size=batch["input_ids"].size(0), )
        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        outputs = self.model(**batch)
        loss = outputs.loss
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=batch["input_ids"].size(0), )

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        outputs = self.model(**batch)
        loss = outputs.loss
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=batch["input_ids"].size(0), )

    def configure_optimizers(self):
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        return AdamW(trainable_params, lr=self.cfg.learning_rate, fused=True)


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required, but no CUDA device was found.")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model path does not exist: {MODEL_PATH}")

    if not (MODEL_PATH / "config.json").exists():
        raise FileNotFoundError(f"config.json not found in model path: {MODEL_PATH}")

    ADAPTER_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TENSORBOARD_ROOT_DIR.mkdir(parents=True, exist_ok=True)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    cfg = TrainConfig()
    L.seed_everything(cfg.seed, workers=True)

    print(f"Loading tokenizer from: {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), use_fast=True, local_files_only=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    datamodule = ChatSFTDataModule(cfg, tokenizer)
    model = LitQwenSFT(cfg, tokenizer)
    if cfg.compile_model:
        model = torch.compile(model)

    logger = TensorBoardLogger(save_dir=str(TENSORBOARD_ROOT_DIR), name=cfg.run_name, default_hp_metric=False)

    trainer = L.Trainer(
        precision="bf16-mixed",
        logger=logger,
        max_epochs=MAX_EPOCHS,
        accumulate_grad_batches=cfg.gradient_accumulation_steps,
        log_every_n_steps=cfg.logging_steps,
        val_check_interval=cfg.val_check_interval,
        check_val_every_n_epoch=1,
        num_sanity_val_steps=0,
        enable_checkpointing=False,
    )

    print("Starting training...")
    trainer.fit(model, datamodule=datamodule)

    print("Running validation...")
    trainer.validate(model, datamodule=datamodule)

    print("Running test evaluation...")
    trainer.test(model, datamodule=datamodule)

    print("Saving final adapter and tokenizer...")
    model.model.save_pretrained(str(ADAPTER_OUTPUT_DIR))
    tokenizer.save_pretrained(str(ADAPTER_OUTPUT_DIR))

    print("=" * 80)
    print("Training complete.")
    print(f"Adapter saved to: {ADAPTER_OUTPUT_DIR}")
    print(f"TensorBoard logs saved to: {logger.log_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
