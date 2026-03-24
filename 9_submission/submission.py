"""
Submission helpers for saving, validating, and smoke-testing Hugging Face model submissions.

This script implements the model submission workflow from the provided submission guide:
- save the model with safe serialization
- save the tokenizer alongside it
- validate that the expected submission files exist
- optionally smoke-test loading the saved folder

Typical usage inside training code:

    from submission import save_model, validate_submission_folder, test_submission_load

    save_dir = save_model(model, tokenizer)
    validate_submission_folder(save_dir, verbose=True)
    test_submission_load(save_dir)

CLI usage:

    python submission.py --validate ./final_submission
    python submission.py --test ./final_submission
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

REQUIRED_FILES = (
    "model.safetensors",
    "config.json",
    "tokenizer.json",
)

RECOMMENDED_FILES = (
    "tokenizer_config.json",
    "special_tokens_map.json",
    "generation_config.json",
)

LEGACY_OPTIONAL_FILES = (
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
)


def _timestamped_submission_dir(prefix: str = "submission") -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"./{prefix}_{timestamp}"


def _existing_files(folder: Path) -> set[str]:
    if not folder.exists() or not folder.is_dir():
        return set()
    return {p.name for p in folder.iterdir() if p.is_file()}


def save_model(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    local_model_path: Optional[str] = None,
) -> str:
    """
    Save a trained model and tokenizer in a submission-ready Hugging Face folder.

    Args:
        model: The trained model to save.
        tokenizer: The corresponding tokenizer.
        local_model_path: Optional output directory. If omitted, a timestamped
            './submission_<timestamp>' folder is created.

    Returns:
        The absolute path to the saved submission folder.
    """
    try:
        if local_model_path is None:
            local_model_path = _timestamped_submission_dir()
            print(f"Saving to default location: {local_model_path}")

        output_dir = Path(local_model_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        model.save_pretrained(output_dir, safe_serialization=True)
        tokenizer.save_pretrained(output_dir)

        abs_path = str(output_dir.resolve())
        print(f"Model successfully saved to: {abs_path}")
        return abs_path

    except Exception as exc:
        print(f"Error saving model: {exc}")
        raise


def validate_submission_folder(folder_path: str | os.PathLike[str], verbose: bool = True) -> Dict[str, Any]:
    """
    Validate that a saved submission folder contains the expected files.

    Returns a dictionary with presence information and convenience flags.
    """
    folder = Path(folder_path)
    files = _existing_files(folder)

    required_present = [name for name in REQUIRED_FILES if name in files]
    required_missing = [name for name in REQUIRED_FILES if name not in files]
    recommended_present = [name for name in RECOMMENDED_FILES if name in files]
    recommended_missing = [name for name in RECOMMENDED_FILES if name not in files]
    legacy_present = [name for name in LEGACY_OPTIONAL_FILES if name in files]

    result: Dict[str, Any] = {
        "folder": str(folder.resolve()) if folder.exists() else str(folder),
        "exists": folder.exists(),
        "is_directory": folder.is_dir(),
        "required_present": required_present,
        "required_missing": required_missing,
        "recommended_present": recommended_present,
        "recommended_missing": recommended_missing,
        "legacy_optional_present": legacy_present,
        "all_files": sorted(files),
        "is_minimal_submission_ready": len(required_missing) == 0,
        "is_full_compatibility_ready": len(required_missing) == 0 and len(recommended_missing) == 0,
    }

    if verbose:
        print(f"Submission folder: {result['folder']}")
        print(f"Exists: {result['exists']}")
        print(f"Is directory: {result['is_directory']}")
        print()
        print("Required files:")
        for name in REQUIRED_FILES:
            status = "OK" if name in files else "MISSING"
            print(f"  - {name}: {status}")
        print()
        print("Recommended files:")
        for name in RECOMMENDED_FILES:
            status = "OK" if name in files else "MISSING"
            print(f"  - {name}: {status}")
        if legacy_present:
            print()
            print("Legacy / optional files found:")
            for name in legacy_present:
                print(f"  - {name}")
        print()
        print(f"Minimal submission ready: {result['is_minimal_submission_ready']}")
        print(f"Full compatibility ready: {result['is_full_compatibility_ready']}")

    return result


def test_submission_load(
    folder_path: str | os.PathLike[str],
    use_causal_lm: bool = False,
    trust_remote_code: bool = False,
) -> Dict[str, str]:
    """
    Smoke-test loading the saved submission.

    By default, this follows the guide's generic load check using AutoModel and
    AutoTokenizer. Set use_causal_lm=True if you specifically want to verify
    AutoModelForCausalLM loading.
    """
    folder = str(Path(folder_path).resolve())

    try:
        if use_causal_lm:
            model = AutoModelForCausalLM.from_pretrained(folder, trust_remote_code=trust_remote_code)
            model_loader = "AutoModelForCausalLM"
        else:
            model = AutoModel.from_pretrained(folder, trust_remote_code=trust_remote_code)
            model_loader = "AutoModel"

        tokenizer = AutoTokenizer.from_pretrained(folder, trust_remote_code=trust_remote_code)

        result = {
            "folder": folder,
            "model_loader": model_loader,
            "model_class": model.__class__.__name__,
            "tokenizer_class": tokenizer.__class__.__name__,
            "status": "ok",
        }
        print("Submission load test successful.")
        print(json.dumps(result, indent=2))
        return result

    except Exception as exc:
        print(f"Submission load test failed: {exc}")
        raise


def print_submission_checklist() -> None:
    """Print a concise checklist derived from the guide."""
    print("Submission checklist")
    print("====================")
    print("Essential files:")
    for name in REQUIRED_FILES:
        print(f"  - {name}")
    print()
    print("Recommended files:")
    for name in RECOMMENDED_FILES:
        print(f"  - {name}")
    print()
    print("Suggested verification:")
    print("  1. Save the model with save_model(model, tokenizer)")
    print("  2. Run validate_submission_folder(path)")
    print("  3. Run test_submission_load(path)")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Save, validate, and test a model submission folder.")
    parser.add_argument("--validate", type=str, help="Validate an existing submission folder.")
    parser.add_argument("--test", type=str, help="Smoke-test loading an existing submission folder.")
    parser.add_argument(
        "--test-causal-lm",
        type=str,
        help="Smoke-test loading an existing submission folder with AutoModelForCausalLM.",
    )
    parser.add_argument("--checklist", action="store_true", help="Print the submission checklist.")
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True during load tests.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    did_something = False

    if args.checklist:
        print_submission_checklist()
        did_something = True

    if args.validate:
        validate_submission_folder(args.validate, verbose=True)
        did_something = True

    if args.test:
        test_submission_load(args.test, use_causal_lm=False, trust_remote_code=args.trust_remote_code)
        did_something = True

    if args.test_causal_lm:
        test_submission_load(args.test_causal_lm, use_causal_lm=True, trust_remote_code=args.trust_remote_code)
        did_something = True

    if not did_something:
        print_submission_checklist()
        print()
        print("Example usage inside training code:")
        print("  from submission import save_model")
        print("  save_dir = save_model(model, tokenizer)")
        print()
        print("Example CLI usage:")
        print("  python submission.py --validate ./submission_folder")
        print("  python submission.py --test ./submission_folder")


if __name__ == "__main__":
    #model_path = "/home/martin/PycharmProjects/UTN-3-LLM-Final-Project/model/Qwen_Qwen3-0.6B-cpt1e-6-SFT/"
#
    #model = AutoModelForCausalLM.from_pretrained(model_path)
    #tokenizer = AutoTokenizer.from_pretrained(model_path)
#
    #save_model(model, tokenizer, "final_submission")

    main()
