#!/usr/bin/env python3
from __future__ import annotations

import copy
import itertools
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    import yaml
except ImportError as e:
    raise SystemExit(
        "PyYAML is required for this script. Install it with:\n"
        "  pip install pyyaml"
    ) from e


# =========================
# Edit only this section
# =========================

DRY_RUN = False
STOP_ON_ERROR = False
CREATE_CONFIG_BACKUP = True
RESTORE_ORIGINAL_CONFIG_AT_END = True

INCLUDE_CPT_MODELS = True
INCLUDE_SFT_ADAPTERS = True

# Naming scheme used by your HPO scripts
COPIED_MODEL_PREFIX = "Qwen_Qwen3-0.6B-tuplecpt"

# =========================
# Paths
# =========================

REPO_ROOT = Path(__file__).resolve().parent

BENCHMARK_MAIN = REPO_ROOT / "4_benchmark" / "main.py"
BENCHMARK_CONFIG = REPO_ROOT / "4_benchmark" / "config.yaml"

MODEL_STORE_DIR = REPO_ROOT / "model"
SFT_ADAPTERS_DIR = REPO_ROOT / "5_sft_on_qna_peft" / "adapters"

RUN_LOG_PATH = REPO_ROOT / "4_benchmark" / "batch_benchmark_runs.jsonl"


# =========================
# Helpers
# =========================

@dataclass(frozen=True)
class BenchmarkTarget:
    kind: str
    name: str
    model_dir_value: str
    path: Path


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


def model_dir_has_safetensors(path: Path) -> bool:
    return path.exists() and path.is_dir() and any(p.is_file() for p in path.glob("*.safetensors"))


def adapter_dir_has_safetensors(path: Path) -> bool:
    return (path / "adapters.safetensors").is_file()


def is_cpt_model_dir(path: Path) -> bool:
    return (
        path.is_dir()
        and path.name.startswith(f"{COPIED_MODEL_PREFIX}_")
        and model_dir_has_safetensors(path)
    )


def is_sft_adapter_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    if not adapter_dir_has_safetensors(path):
        return False

    name = path.name

    return (
        name.startswith(f"{COPIED_MODEL_PREFIX}_")
        and "-epoch-SFT" in name
    )


def discover_targets() -> list[BenchmarkTarget]:
    targets: list[BenchmarkTarget] = []

    if INCLUDE_CPT_MODELS:
        ensure_exists(MODEL_STORE_DIR, "model store directory")
        for path in sorted(MODEL_STORE_DIR.iterdir(), key=lambda p: p.name):
            if is_cpt_model_dir(path):
                targets.append(
                    BenchmarkTarget(
                        kind="cpt_model",
                        name=path.name,
                        model_dir_value="model",
                        path=path,
                    )
                )

    if INCLUDE_SFT_ADAPTERS:
        ensure_exists(SFT_ADAPTERS_DIR, "SFT adapters directory")
        for path in sorted(SFT_ADAPTERS_DIR.iterdir(), key=lambda p: p.name):
            if is_sft_adapter_dir(path):
                targets.append(
                    BenchmarkTarget(
                        kind="sft_adapter",
                        name=path.name,
                        model_dir_value="5_sft_on_qna_peft/adapters",
                        path=path,
                    )
                )

    return targets


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise TypeError(f"YAML root must be a mapping in {path}")
    return data


def write_yaml(path: Path, data: dict) -> None:
    if DRY_RUN:
        log(f"[DRY RUN] Would write benchmark config: {path}")
        return
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def create_backup_if_needed(path: Path) -> Path | None:
    if not CREATE_CONFIG_BACKUP:
        return None
    backup_path = path.with_suffix(path.suffix + ".bak")
    if not backup_path.exists():
        backup_path.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
        log(f"Created backup: {backup_path}")
    return backup_path


def build_config_for_target(base_config: dict, target: BenchmarkTarget) -> dict:
    cfg = copy.deepcopy(base_config)

    cfg["answer_llm"]["provider"] = "huggingface"
    cfg["answer_llm"]["huggingface"]["model"] = target.name
    cfg["answer_llm"]["huggingface"]["model_dir"] = target.model_dir_value

    return cfg


def restore_original_config(path: Path, original_text: str) -> None:
    if not RESTORE_ORIGINAL_CONFIG_AT_END:
        return
    if DRY_RUN:
        log(f"[DRY RUN] Would restore original config: {path}")
        return
    path.write_text(original_text, encoding="utf-8")
    log(f"Restored original config: {path}")


def append_run_log(record: dict) -> None:
    line = json.dumps(record, ensure_ascii=False)
    if DRY_RUN:
        log(f"[DRY RUN] Would append run log: {line}")
        return
    with RUN_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def benchmark_target(base_config: dict, target: BenchmarkTarget) -> None:
    cfg = build_config_for_target(base_config, target)

    log("\n========================================")
    log(f"Benchmarking {target.kind}: {target.name}")
    log(f"model_dir={target.model_dir_value}")
    log("========================================")

    write_yaml(BENCHMARK_CONFIG, cfg)

    cmd = [sys.executable, str(BENCHMARK_MAIN)]
    run_cmd(cmd, cwd=REPO_ROOT)

    append_run_log(
        {
            "kind": target.kind,
            "name": target.name,
            "model_dir": target.model_dir_value,
            "path": str(target.path),
        }
    )


def main() -> None:
    ensure_exists(BENCHMARK_MAIN, "benchmark main.py")
    ensure_exists(BENCHMARK_CONFIG, "benchmark config.yaml")

    create_backup_if_needed(BENCHMARK_CONFIG)

    original_config_text = BENCHMARK_CONFIG.read_text(encoding="utf-8")
    base_config = load_yaml(BENCHMARK_CONFIG)

    targets = discover_targets()

    if not targets:
        log("No benchmark targets found.")
        return

    log(f"Discovered {len(targets)} benchmark target(s):")
    for target in targets:
        log(f"- [{target.kind}] {target.name}")

    try:
        for target in targets:
            try:
                benchmark_target(base_config, target)
            except Exception as e:
                log(f"\nERROR while benchmarking {target.name}: {e}")
                if STOP_ON_ERROR:
                    raise
    finally:
        restore_original_config(BENCHMARK_CONFIG, original_config_text)

    log("\nDone.")
    log(f"Run log: {RUN_LOG_PATH}")


if __name__ == "__main__":
    main()