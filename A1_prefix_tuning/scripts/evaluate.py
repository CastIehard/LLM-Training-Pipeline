from __future__ import annotations

import argparse
import math

import torch
from datasets import load_dataset
from peft import PeftModel
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import (
    CausalLMCollator,
    get_split_paths,
    load_config,
    parse_torch_dtype,
    resolve_model_name_or_path,
    resolve_path,
    tokenize_chat_example,
    write_json,
)


def decode_prompt(messages, tokenizer) -> str:
    return tokenizer.apply_chat_template(
        messages[:-1],
        tokenize=False,
        add_generation_prompt=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained prefix adapter.")
    parser.add_argument("--config", required=True, help="Path to the YAML config file.")
    parser.add_argument(
        "--adapter-dir",
        default=None,
        help="Optional adapter directory. Defaults to training.output_dir/final_adapter.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    model_config = config["model"]
    training_config = config["training"]
    eval_config = config["evaluation"]

    split_paths = get_split_paths(config)
    adapter_dir = resolve_path(args.adapter_dir) if args.adapter_dir else resolve_path(training_config["output_dir"]) / "final_adapter"
    model_name_or_path = resolve_model_name_or_path(model_config["name_or_path"])

    tokenizer = AutoTokenizer.from_pretrained(
        adapter_dir if adapter_dir.exists() else model_name_or_path,
        use_fast=True,
        local_files_only=model_config.get("local_files_only", False),
        trust_remote_code=model_config.get("trust_remote_code", False),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    raw_datasets = load_dataset(
        "json",
        data_files={
            "train": str(split_paths["train"]),
            "validation": str(split_paths["validation"]),
            "test": str(split_paths["test"]),
        },
    )

    split_name = eval_config["split"]
    max_seq_length = config["data"]["max_seq_length"]
    tokenized = raw_datasets[split_name].map(
        lambda example: tokenize_chat_example(example, tokenizer, max_seq_length),
        remove_columns=["messages"],
        desc=f"Tokenizing {split_name} split",
    )
    tokenized = tokenized.sort("seq_len")

    dataloader = DataLoader(
        tokenized,
        shuffle=False,
        batch_size=training_config["per_device_eval_batch_size"],
        num_workers=training_config["dataloader_num_workers"],
        pin_memory=True,
        collate_fn=CausalLMCollator(tokenizer),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=parse_torch_dtype(model_config["torch_dtype"]),
        attn_implementation=model_config.get("attn_implementation"),
        local_files_only=model_config.get("local_files_only", False),
        trust_remote_code=model_config.get("trust_remote_code", False),
    )
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model.to(device)
    model.eval()

    losses = []
    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            if eval_config.get("max_eval_batches") is not None and step >= eval_config["max_eval_batches"]:
                break
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            losses.append(outputs.loss.item())

    mean_loss = sum(losses) / max(len(losses), 1)
    perplexity = math.exp(mean_loss) if mean_loss < 20 else float("inf")

    generation_samples = []
    generation_temperature = float(eval_config.get("generation_temperature", 0.0))
    do_sample = generation_temperature > 0
    for index, example in enumerate(raw_datasets[split_name]):
        if index >= int(eval_config["num_generation_samples"]):
            break

        prompt_text = decode_prompt(example["messages"], tokenizer)
        inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
        generation_kwargs = {
            "max_new_tokens": int(eval_config["generation_max_new_tokens"]),
            "do_sample": do_sample,
        }
        if do_sample:
            generation_kwargs["temperature"] = generation_temperature
            generation_kwargs["top_p"] = float(eval_config.get("generation_top_p", 1.0))

        generated = model.generate(**inputs, **generation_kwargs)
        prompt_length = inputs["input_ids"].shape[1]
        prediction = tokenizer.decode(generated[0][prompt_length:], skip_special_tokens=True)
        generation_samples.append(
            {
                "prompt": prompt_text,
                "reference": example["messages"][-1]["content"],
                "prediction": prediction.strip(),
            }
        )

    payload = {
        "split": split_name,
        "adapter_dir": str(adapter_dir),
        "loss": mean_loss,
        "perplexity": perplexity,
        "num_batches": len(losses),
        "samples": generation_samples,
    }
    write_json(resolve_path(training_config["output_dir"]) / "evaluation.json", payload)
    print(payload)


if __name__ == "__main__":
    main()
