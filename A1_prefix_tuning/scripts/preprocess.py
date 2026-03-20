from __future__ import annotations

import argparse

from common import (
    apply_runtime_environment,
    format_chat_example,
    get_split_paths,
    load_config,
    read_jsonl,
    resolve_path,
    save_yaml,
    write_json,
    write_jsonl,
    grouped_split,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create train/validation/test JSONL files for prefix tuning.")
    parser.add_argument("--config", required=True, help="Path to the YAML config file.")
    args = parser.parse_args()

    config = load_config(args.config)
    storage = apply_runtime_environment(config)
    source_path = resolve_path(config["data"]["source_jsonl"])
    split_paths = get_split_paths(config)

    rows = read_jsonl(source_path)
    train_rows, valid_rows, test_rows = grouped_split(
        rows=rows,
        group_key=config["data"]["group_by_key"],
        train_ratio=config["data"]["train_ratio"],
        valid_ratio=config["data"]["valid_ratio"],
        seed=config["seed"],
    )

    formatted_train = [format_chat_example(row, config) for row in train_rows]
    formatted_valid = [format_chat_example(row, config) for row in valid_rows]
    formatted_test = [format_chat_example(row, config) for row in test_rows]

    write_jsonl(split_paths["train"], formatted_train)
    write_jsonl(split_paths["validation"], formatted_valid)
    write_jsonl(split_paths["test"], formatted_test)

    summary = {
        "source_path": str(source_path),
        "persistent_root": str(storage["persistent_root"]),
        "scratch_root": str(storage["scratch_root"]),
        "train_samples": len(formatted_train),
        "validation_samples": len(formatted_valid),
        "test_samples": len(formatted_test),
    }
    write_json(resolve_path(config["data"]["output_dir"]) / "split_summary.json", summary)
    save_yaml(resolve_path(config["data"]["output_dir"]) / "resolved_config.yaml", config)

    print(summary)


if __name__ == "__main__":
    main()
