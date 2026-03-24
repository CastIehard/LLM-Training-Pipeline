#!/usr/bin/env python3
from __future__ import annotations

import itertools
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

os.environ["MLX_CUDA_GRAPH_CACHE_SIZE"] = "160000"

# =========================
# Edit only this section
# =========================

CPT_LEARNING_RATES = ["1e-6"]  # , "2e-6", "3e-6", "5e-6", "1e-5", "5e-5"
CPT_EPOCHS = [1, 3, 5]

SFT_LEARNING_RATES = ["5e-5"]
SFT_EPOCHS = [2]  # 4

# For your setup you noted: one SFT epoch = 2115 iterations.
SFT_ITERS_PER_EPOCH = 2115

# Behavior flags
SKIP_EXISTING_CPT = True
SKIP_EXISTING_MODEL_COPY = True
SKIP_EXISTING_SFT = True
STOP_ON_ERROR = False
DRY_RUN = False
CREATE_LORA_CONFIG_BACKUP = True

# =========================
# Paths and naming
# =========================

REPO_ROOT = Path(__file__).resolve().parent

CPT_SCRIPT = REPO_ROOT / "6_continuous_pretraining" / "cpt_single_triple_file.py"
CPT_INPUT_FILE = REPO_ROOT / "data" / "cpt_training_set" / "train_cpt_triples_pipe_text.md"
CPT_RUNS_DIR = REPO_ROOT / "6_continuous_pretraining" / "runs"

MODEL_STORE_DIR = REPO_ROOT / "model"

SFT_CONFIG_PATH = REPO_ROOT / "5_sft_on_qna_peft" / "lora_config.yaml"
SFT_TRAIN_SCRIPT = REPO_ROOT / "5_sft_on_qna_peft" / "scripts" / "train.py"
SFT_ADAPTERS_DIR = REPO_ROOT / "5_sft_on_qna_peft" / "adapters"

BASE_MODEL_NAME_OR_PATH = (REPO_ROOT / "model" / "Qwen_Qwen3-0.6B-cpt1e-6")
BASE_MODEL_TAG = Path(BASE_MODEL_NAME_OR_PATH).name

COPIED_MODEL_PREFIX = f"{str(BASE_MODEL_TAG)}-tuplecpt"

MANIFEST_PATH = REPO_ROOT / "manual_hpo_manifest.jsonl"


# =========================
# Helpers
# =========================

def log(msg: str) -> None:
    print(msg, flush=True)


def run_cmd(cmd: list[str], cwd: Path | None = None) -> None:
    rendered = " ".join(cmd)
    log(f"\n$ {rendered}")
    if DRY_RUN:
        return
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def ensure_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")


def dir_has_adapters_safetensors(path: Path) -> bool:
    return (path / "adapters.safetensors").is_file()


def dir_has_model_safetensors(path: Path) -> bool:
    return (path / "model.safetensors").is_file()


def copied_model_name(cpt_lr: str, cpt_epochs: int) -> str:
    return f"{str(COPIED_MODEL_PREFIX)}_{cpt_lr}_{cpt_epochs}-epoch"


def build_adapter_name(cpt_lr: str, cpt_epochs: int, sft_lr: str, sft_epochs: int, multiple_sft_variants: bool) -> str:
    base = copied_model_name(cpt_lr, cpt_epochs)
    return f"{base}-lr_{sft_lr}-ep_{sft_epochs}-SFT"


def sft_iters_from_epochs(sft_epochs: int) -> int:
    return sft_epochs * SFT_ITERS_PER_EPOCH


def replace_yaml_scalar(text: str, key: str, value: str) -> str:
    pattern = rf"(?m)^({re.escape(key)}:\s*).*$"
    if not re.search(pattern, text):
        raise KeyError(f"Could not find YAML key '{key}' in {SFT_CONFIG_PATH}")
    return re.sub(pattern, lambda m: f"{m.group(1)}{value}", text, count=1)


def backup_lora_config() -> Path | None:
    if not CREATE_LORA_CONFIG_BACKUP:
        return None
    backup_path = SFT_CONFIG_PATH.with_suffix(SFT_CONFIG_PATH.suffix + ".bak")
    if not backup_path.exists():
        shutil.copy2(SFT_CONFIG_PATH, backup_path)
        log(f"Created backup: {backup_path}")
    return backup_path


def patch_rope_theta_in_final_model_config(final_model_config_path: Path) -> int | float:
    ensure_exists(final_model_config_path, "final_model config.json")

    with final_model_config_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    rope_parameters = data.get("rope_parameters")
    if not isinstance(rope_parameters, dict):
        raise KeyError(f"'rope_parameters' missing or not a dict in {final_model_config_path}")

    if "rope_theta" not in rope_parameters:
        raise KeyError(f"'rope_parameters.rope_theta' missing in {final_model_config_path}")

    rope_theta = rope_parameters["rope_theta"]

    if data.get("rope_theta") == rope_theta:
        log(f"rope_theta already present at top level in {final_model_config_path}")
        return rope_theta

    new_data = {}
    inserted = False
    for key, value in data.items():
        if key == "rope_parameters" and not inserted:
            new_data["rope_theta"] = rope_theta
            inserted = True
        new_data[key] = value

    if not inserted:
        new_data["rope_theta"] = rope_theta

    with final_model_config_path.open("w", encoding="utf-8") as f:
        json.dump(new_data, f, indent=2, ensure_ascii=False)
        f.write("\n")

    log(f"Patched rope_theta={rope_theta} into top level of {final_model_config_path}")
    return rope_theta


def copy_final_model_to_model_store(src_final_model_dir: Path, dst_model_dir: Path) -> None:
    ensure_exists(src_final_model_dir, "source final_model directory")
    MODEL_STORE_DIR.mkdir(parents=True, exist_ok=True)

    if dst_model_dir.exists():
        if SKIP_EXISTING_MODEL_COPY and dir_has_model_safetensors(dst_model_dir):
            log(f"Model copy already exists and contains .safetensors, skipping: {dst_model_dir}")
            return
        log(f"Model copy directory exists but is incomplete, replacing: {dst_model_dir}")
        if not DRY_RUN:
            shutil.rmtree(dst_model_dir)

    if DRY_RUN:
        log(f"[DRY RUN] Would copy {src_final_model_dir} -> {dst_model_dir}")
        return

    shutil.copytree(src_final_model_dir, dst_model_dir)
    log(f"Copied model: {dst_model_dir}")


def update_lora_config_for_run(model_name: str, adapter_name: str, sft_lr: str, sft_epochs: int) -> int:
    ensure_exists(SFT_CONFIG_PATH, "lora_config.yaml")

    model_rel = f"../model/{model_name}"
    adapter_rel = f"./adapters/{adapter_name}"
    iters = sft_iters_from_epochs(sft_epochs)

    text = SFT_CONFIG_PATH.read_text(encoding="utf-8")
    text = replace_yaml_scalar(text, "model", f"\"{model_rel}\"")
    text = replace_yaml_scalar(text, "adapter_path", f"\"{adapter_rel}\"")
    text = replace_yaml_scalar(text, "learning_rate", sft_lr)
    text = replace_yaml_scalar(text, "iters", str(iters))

    if DRY_RUN:
        log(f"[DRY RUN] Would update {SFT_CONFIG_PATH} with model={model_rel}, adapter_path={adapter_rel}, learning_rate={sft_lr}, iters={iters}")
        return iters

    SFT_CONFIG_PATH.write_text(text, encoding="utf-8")
    log(f"Updated {SFT_CONFIG_PATH} for adapter {adapter_name}")
    return iters


def append_manifest(record: dict) -> None:
    line = json.dumps(record, ensure_ascii=False)
    if DRY_RUN:
        log(f"[DRY RUN] Would append manifest record: {line}")
        return
    with MANIFEST_PATH.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def run_cpt(cpt_lr: str, cpt_epochs: int) -> tuple[Path, Path, int | float]:
    run_name = copied_model_name(cpt_lr, cpt_epochs)
    output_dir = CPT_RUNS_DIR / run_name
    final_model_dir = output_dir / "final_model"
    final_model_config = final_model_dir / "config.json"

    cpt_already_done = final_model_dir.exists() and final_model_config.exists() and dir_has_model_safetensors(final_model_dir)

    if cpt_already_done and SKIP_EXISTING_CPT:
        log(f"\nCPT already exists and contains .safetensors, skipping training: {output_dir}")
    else:
        if final_model_dir.exists() and not cpt_already_done:
            log(f"\nCPT output exists but looks incomplete, re-running: {output_dir}")
        cmd = [
            sys.executable,
            str(CPT_SCRIPT),
            "--input_file", str(CPT_INPUT_FILE),
            "--output_dir", str(output_dir),
            f"--num_train_epochs={cpt_epochs}",
            f"--learning_rate={cpt_lr}",
            "--model_name_or_path", str(BASE_MODEL_NAME_OR_PATH),
        ]
        run_cmd(cmd, cwd=REPO_ROOT)

    ensure_exists(final_model_dir, "final_model directory after CPT")
    ensure_exists(final_model_config, "final_model config.json after CPT")
    if not dir_has_model_safetensors(final_model_dir):
        raise FileNotFoundError(f"No .safetensors file found in final_model directory after CPT: {final_model_dir}")

    rope_theta = patch_rope_theta_in_final_model_config(final_model_config)
    return output_dir, final_model_dir, rope_theta


def run_sft_for_model(cpt_lr: str, cpt_epochs: int, sft_lr: str, sft_epochs: int, multiple_sft_variants: bool) -> dict:
    model_name = copied_model_name(cpt_lr, cpt_epochs)
    model_dir = MODEL_STORE_DIR / model_name
    adapter_name = build_adapter_name(cpt_lr, cpt_epochs, sft_lr, sft_epochs, multiple_sft_variants)
    adapter_dir = SFT_ADAPTERS_DIR / adapter_name

    adapter_already_done = dir_has_adapters_safetensors(adapter_dir)

    if adapter_already_done and SKIP_EXISTING_SFT:
        log(f"SFT adapter already exists and contains .safetensors, skipping training: {adapter_dir}")
        return {
            "model_name": model_name,
            "model_dir": str(model_dir),
            "adapter_name": adapter_name,
            "adapter_dir": str(adapter_dir),
            "sft_learning_rate": sft_lr,
            "sft_epochs": sft_epochs,
            "sft_iters": sft_iters_from_epochs(sft_epochs),
            "skipped_sft": True,
        }

    if adapter_dir.exists() and not adapter_already_done:
        log(f"SFT adapter directory exists but has no .safetensors, removing incomplete run: {adapter_dir}")
        if not DRY_RUN:
            shutil.rmtree(adapter_dir)

    sft_iters = update_lora_config_for_run(
        model_name=model_name,
        adapter_name=adapter_name,
        sft_lr=sft_lr,
        sft_epochs=sft_epochs,
    )

    cmd = [sys.executable, str(SFT_TRAIN_SCRIPT)]
    run_cmd(cmd, cwd=REPO_ROOT)

    if not dir_has_adapters_safetensors(adapter_dir):
        raise FileNotFoundError(f"SFT finished but no .safetensors file was found in adapter directory: {adapter_dir}")

    return {
        "model_name": model_name,
        "model_dir": str(model_dir),
        "adapter_name": adapter_name,
        "adapter_dir": str(adapter_dir),
        "sft_learning_rate": sft_lr,
        "sft_epochs": sft_epochs,
        "sft_iters": sft_iters,
        "skipped_sft": False,
    }


def main() -> None:
    ensure_exists(CPT_SCRIPT, "CPT script")
    ensure_exists(CPT_INPUT_FILE, "CPT input file")
    ensure_exists(SFT_CONFIG_PATH, "SFT config file")
    ensure_exists(SFT_TRAIN_SCRIPT, "SFT train script")

    backup_lora_config()

    sft_grid = list(itertools.product(SFT_LEARNING_RATES, SFT_EPOCHS))
    multiple_sft_variants = len(sft_grid) > 1

    all_results = []
    os.environ["MLX_CUDA_GRAPH_CACHE_SIZE"] = "160000"

    for cpt_lr, cpt_epochs in itertools.product(CPT_LEARNING_RATES, CPT_EPOCHS):
        log(f"\n==============================")
        log(f"Base model sweep: lr={cpt_lr}, epochs={cpt_epochs}")
        log(f"==============================")

        output_dir, final_model_dir, rope_theta = run_cpt(cpt_lr, cpt_epochs)

        model_name = copied_model_name(cpt_lr, cpt_epochs)
        dst_model_dir = MODEL_STORE_DIR / model_name
        copy_final_model_to_model_store(final_model_dir, dst_model_dir)

        for sft_lr, sft_epochs in sft_grid:
            log(f"\n  -> SFT sweep: lr={sft_lr}, epochs={sft_epochs}")

            sft_result = run_sft_for_model(
                cpt_lr=cpt_lr,
                cpt_epochs=cpt_epochs,
                sft_lr=sft_lr,
                sft_epochs=sft_epochs,
                multiple_sft_variants=multiple_sft_variants,
            )

            record = {
                "cpt_learning_rate": cpt_lr,
                "cpt_epochs": cpt_epochs,
                "cpt_output_dir": str(output_dir),
                "copied_model_name": model_name,
                "copied_model_dir": str(dst_model_dir),
                "rope_theta": rope_theta,
                **sft_result,
            }
            append_manifest(record)
            all_results.append(record)

    log("\nDone.")
    log(f"Manifest: {MANIFEST_PATH}")
    log("\nCompleted runs:")
    for record in all_results:
        log(
            f"- CPT lr={record['cpt_learning_rate']}, CPT ep={record['cpt_epochs']}, "
            f"SFT lr={record['sft_learning_rate']}, SFT ep={record['sft_epochs']}, "
            f"model={record['copied_model_name']}, adapter={record['adapter_name']}"
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"\nERROR: {e}")
        if STOP_ON_ERROR:
            raise
