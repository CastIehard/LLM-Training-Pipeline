from __future__ import annotations

import argparse
import math

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from datasets import load_dataset
from peft import PrefixTuningConfig, TaskType, get_peft_model
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler

from common import (
    CausalLMCollator,
    get_split_paths,
    load_config,
    parse_torch_dtype,
    resolve_model_name_or_path,
    resolve_path,
    rotate_checkpoints,
    save_yaml,
    tokenize_chat_example,
    trainable_parameter_summary,
    write_json,
)


def evaluate(model, dataloader, accelerator: Accelerator, max_batches: int | None = None) -> dict[str, float]:
    model.eval()
    losses = []

    for step, batch in enumerate(dataloader):
        if max_batches is not None and step >= max_batches:
            break
        with torch.no_grad():
            outputs = model(**batch)
        loss = outputs.loss.detach()
        losses.append(accelerator.gather_for_metrics(loss.unsqueeze(0)))

    if not losses:
        return {"loss": float("nan"), "perplexity": float("nan")}

    mean_loss = torch.cat(losses).mean().item()
    perplexity = math.exp(mean_loss) if mean_loss < 20 else float("inf")
    model.train()
    return {"loss": mean_loss, "perplexity": perplexity}


def save_adapter(accelerator: Accelerator, model, tokenizer, output_dir: Path, metadata: dict) -> None:
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        unwrapped_model.save_pretrained(output_dir, safe_serialization=True)
        tokenizer.save_pretrained(output_dir)
        write_json(output_dir / "metadata.json", metadata)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prefix-tune Qwen3 with PEFT and Accelerate.")
    parser.add_argument("--config", required=True, help="Path to the YAML config file.")
    args = parser.parse_args()

    config = load_config(args.config)
    training_config = config["training"]
    model_config = config["model"]
    peft_config = config["peft"]
    evaluation_config = config["evaluation"]

    model_dtype = parse_torch_dtype(model_config["torch_dtype"])
    if model_dtype == torch.bfloat16:
        mixed_precision = "bf16"
    elif model_dtype == torch.float16:
        mixed_precision = "fp16"
    else:
        mixed_precision = "no"

    accelerator = Accelerator(
        gradient_accumulation_steps=training_config["gradient_accumulation_steps"],
        mixed_precision=mixed_precision,
    )

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = bool(training_config.get("tf32", True))
        torch.backends.cudnn.allow_tf32 = bool(training_config.get("tf32", True))

    set_seed(config["seed"])

    model_name_or_path = resolve_model_name_or_path(model_config["name_or_path"])
    split_paths = get_split_paths(config)
    output_dir = resolve_path(training_config["output_dir"])
    checkpoint_root = output_dir / "checkpoints"
    final_adapter_dir = output_dir / "final_adapter"

    accelerator.print(f"Loading tokenizer from {model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast=True,
        local_files_only=model_config.get("local_files_only", False),
        trust_remote_code=model_config.get("trust_remote_code", False),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    accelerator.print("Loading dataset splits")
    raw_datasets = load_dataset(
        "json",
        data_files={
            "train": str(split_paths["train"]),
            "validation": str(split_paths["validation"]),
            "test": str(split_paths["test"]),
        },
    )

    max_seq_length = config["data"]["max_seq_length"]
    tokenized = raw_datasets.map(
        lambda example: tokenize_chat_example(example, tokenizer, max_seq_length),
        remove_columns=["messages"],
        desc="Tokenizing chat dataset",
    )
    tokenized["train"] = tokenized["train"].sort("seq_len")
    tokenized["validation"] = tokenized["validation"].sort("seq_len")

    train_loader = DataLoader(
        tokenized["train"],
        shuffle=True,
        batch_size=training_config["per_device_train_batch_size"],
        num_workers=training_config["dataloader_num_workers"],
        pin_memory=True,
        collate_fn=CausalLMCollator(tokenizer),
    )
    eval_loader = DataLoader(
        tokenized["validation"],
        shuffle=False,
        batch_size=training_config["per_device_eval_batch_size"],
        num_workers=training_config["dataloader_num_workers"],
        pin_memory=True,
        collate_fn=CausalLMCollator(tokenizer),
    )

    accelerator.print(f"Loading base model from {model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=model_dtype,
        attn_implementation=model_config.get("attn_implementation"),
        local_files_only=model_config.get("local_files_only", False),
        trust_remote_code=model_config.get("trust_remote_code", False),
    )
    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.pad_token_id

    if training_config.get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable()

    prefix_tuning_config = PrefixTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        num_virtual_tokens=peft_config["num_virtual_tokens"],
        token_dim=model.config.hidden_size,
        num_transformer_submodules=1,
        num_attention_heads=model.config.num_attention_heads,
        num_layers=model.config.num_hidden_layers,
        encoder_hidden_size=peft_config.get("encoder_hidden_size") or model.config.hidden_size,
        prefix_projection=peft_config.get("prefix_projection", True),
    )
    model = get_peft_model(model, prefix_tuning_config)

    if training_config.get("compile_model", False) and hasattr(torch, "compile"):
        model = torch.compile(model)

    parameter_counts = trainable_parameter_summary(model)
    accelerator.print(
        f"Trainable parameters: {parameter_counts['trainable']:,} / {parameter_counts['total']:,}"
    )

    optimizer = AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=training_config["learning_rate"],
        weight_decay=training_config["weight_decay"],
    )

    steps_per_epoch = math.ceil(len(train_loader) / training_config["gradient_accumulation_steps"])
    if training_config.get("max_steps"):
        max_train_steps = int(training_config["max_steps"])
        num_train_epochs = math.ceil(max_train_steps / max(steps_per_epoch, 1))
    else:
        num_train_epochs = int(training_config["num_train_epochs"])
        max_train_steps = max(steps_per_epoch, 1) * num_train_epochs

    lr_scheduler = get_scheduler(
        name=training_config["lr_scheduler_type"],
        optimizer=optimizer,
        num_warmup_steps=training_config["num_warmup_steps"],
        num_training_steps=max_train_steps,
    )

    model, optimizer, train_loader, eval_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, eval_loader, lr_scheduler
    )

    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        save_yaml(output_dir / "resolved_config.yaml", config)

    global_step = 0
    best_eval_loss = float("inf")
    train_loss_sum = 0.0
    train_loss_steps = 0

    for epoch in range(num_train_epochs):
        model.train()
        for batch in train_loader:
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                train_loss_sum += loss.detach().float().item()
                train_loss_steps += 1

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), training_config["max_grad_norm"])

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if not accelerator.sync_gradients:
                continue

            global_step += 1

            if global_step % training_config["logging_steps"] == 0:
                mean_train_loss = train_loss_sum / max(train_loss_steps, 1)
                accelerator.print(
                    f"step={global_step} epoch={epoch + 1} train_loss={mean_train_loss:.4f} lr={lr_scheduler.get_last_lr()[0]:.6f}"
                )
                train_loss_sum = 0.0
                train_loss_steps = 0

            if global_step % training_config["eval_steps"] == 0:
                metrics = evaluate(
                    model,
                    eval_loader,
                    accelerator,
                    max_batches=evaluation_config.get("max_eval_batches"),
                )
                accelerator.print(
                    f"step={global_step} eval_loss={metrics['loss']:.4f} eval_ppl={metrics['perplexity']:.2f}"
                )
                if metrics["loss"] < best_eval_loss:
                    best_eval_loss = metrics["loss"]
                    save_adapter(
                        accelerator,
                        model,
                        tokenizer,
                        output_dir / "best_adapter",
                        {
                            "global_step": global_step,
                            "eval_loss": metrics["loss"],
                            "eval_perplexity": metrics["perplexity"],
                        },
                    )

            if global_step % training_config["save_steps"] == 0:
                checkpoint_dir = checkpoint_root / f"step-{global_step}"
                save_adapter(
                    accelerator,
                    model,
                    tokenizer,
                    checkpoint_dir,
                    {
                        "global_step": global_step,
                        "epoch": epoch + 1,
                    },
                )
                if accelerator.is_main_process:
                    rotate_checkpoints(checkpoint_root, int(training_config["save_total_limit"]))

            if global_step >= max_train_steps:
                break

        if global_step >= max_train_steps:
            break

    final_metrics = evaluate(
        model,
        eval_loader,
        accelerator,
        max_batches=evaluation_config.get("max_eval_batches"),
    )
    accelerator.print(
        f"final_eval_loss={final_metrics['loss']:.4f} final_eval_ppl={final_metrics['perplexity']:.2f}"
    )
    save_adapter(
        accelerator,
        model,
        tokenizer,
        final_adapter_dir,
        {
            "global_step": global_step,
            "eval_loss": final_metrics["loss"],
            "eval_perplexity": final_metrics["perplexity"],
            "trainable_parameters": parameter_counts["trainable"],
            "total_parameters": parameter_counts["total"],
        },
    )


if __name__ == "__main__":
    main()
