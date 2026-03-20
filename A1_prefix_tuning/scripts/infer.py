from __future__ import annotations

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import (
    apply_runtime_environment,
    get_model_cache_dir,
    load_config,
    parse_torch_dtype,
    resolve_model_name_or_path,
    resolve_path,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference with a trained prefix adapter.")
    parser.add_argument("--config", required=True, help="Path to the YAML config file.")
    parser.add_argument("--prompt", required=True, help="User prompt to send to the model.")
    parser.add_argument(
        "--adapter-dir",
        default=None,
        help="Optional adapter directory. Defaults to training.output_dir/final_adapter.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()

    config = load_config(args.config)
    apply_runtime_environment(config)
    model_config = config["model"]
    adapter_dir = resolve_path(args.adapter_dir) if args.adapter_dir else resolve_path(config["training"]["output_dir"]) / "final_adapter"
    model_name_or_path = resolve_model_name_or_path(model_config["name_or_path"])
    model_cache_dir = get_model_cache_dir(config)

    tokenizer = AutoTokenizer.from_pretrained(
        adapter_dir if adapter_dir.exists() else model_name_or_path,
        use_fast=True,
        cache_dir=str(model_cache_dir),
        local_files_only=model_config.get("local_files_only", False),
        trust_remote_code=model_config.get("trust_remote_code", False),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=parse_torch_dtype(model_config["torch_dtype"]),
        attn_implementation=model_config.get("attn_implementation"),
        cache_dir=str(model_cache_dir),
        local_files_only=model_config.get("local_files_only", False),
        trust_remote_code=model_config.get("trust_remote_code", False),
    )
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model.to(device)
    model.eval()

    messages = [
        {"role": "system", "content": config["data"]["system_prompt"]},
        {"role": "user", "content": f"{args.prompt}{config['data']['no_think_suffix']}" if config["data"].get("append_no_think_suffix", False) else args.prompt},
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    output = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
    )

    prompt_length = inputs["input_ids"].shape[1]
    response = tokenizer.decode(output[0][prompt_length:], skip_special_tokens=True).strip()
    print(response)


if __name__ == "__main__":
    main()
