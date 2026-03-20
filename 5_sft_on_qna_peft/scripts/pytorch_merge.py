import os
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
RUN_DIR = SCRIPT_DIR.parent
PROJECT_DIR = RUN_DIR.parent

BASE_MODEL_PATH = PROJECT_DIR / "model" / "Qwen_Qwen3-0.6B"
ADAPTER_PATH = RUN_DIR / "adapters" / "Qwen_Qwen3-0.6B_lora_lightning"
MERGED_MODEL_PATH = RUN_DIR / "merged" / "Qwen_Qwen3-0.6B_lora_merged"


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required, but no CUDA device was found.")

    if not BASE_MODEL_PATH.exists():
        raise FileNotFoundError(f"Base model path does not exist: {BASE_MODEL_PATH}")

    if not ADAPTER_PATH.exists():
        raise FileNotFoundError(f"Adapter path does not exist: {ADAPTER_PATH}")

    if not (BASE_MODEL_PATH / "config.json").exists():
        raise FileNotFoundError(f"config.json not found in base model path: {BASE_MODEL_PATH}")

    if not (ADAPTER_PATH / "adapter_config.json").exists():
        raise FileNotFoundError(f"adapter_config.json not found in adapter path: {ADAPTER_PATH}")

    MERGED_MODEL_PATH.mkdir(parents=True, exist_ok=True)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print(f"Loading tokenizer from: {BASE_MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(
        str(BASE_MODEL_PATH),
        use_fast=True,
        local_files_only=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading base model from: {BASE_MODEL_PATH}")
    base_model = AutoModelForCausalLM.from_pretrained(
        str(BASE_MODEL_PATH),
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        local_files_only=True,
    )

    base_model.config.pad_token_id = tokenizer.pad_token_id

    print(f"Loading adapter from: {ADAPTER_PATH}")
    peft_model = PeftModel.from_pretrained(
        base_model,
        str(ADAPTER_PATH),
        is_trainable=False,
    )

    print("Merging adapter into base model...")
    merged_model = peft_model.merge_and_unload()

    print(f"Saving merged model to: {MERGED_MODEL_PATH}")
    merged_model.save_pretrained(
        str(MERGED_MODEL_PATH),
        safe_serialization=True,
    )
    tokenizer.save_pretrained(str(MERGED_MODEL_PATH))

    print("=" * 80)
    print("Merge complete.")
    print(f"Merged model saved to: {MERGED_MODEL_PATH}")
    print("=" * 80)


if __name__ == "__main__":
    main()
