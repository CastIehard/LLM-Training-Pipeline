import argparse
import subprocess
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent


def run_step(name: str, cmd: list[str]) -> None:
    print(f"\n{'=' * 80}\n{name}\n{'=' * 80}")
    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, cwd=BASE_DIR)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the prefix-tuning pipeline locally.")
    parser.add_argument(
        "--config",
        default=str(BASE_DIR / "config.yaml"),
        help="Path to the YAML config file.",
    )
    args = parser.parse_args()

    run_step(
        "Preprocess dataset",
        [sys.executable, str(BASE_DIR / "scripts" / "preprocess.py"), "--config", args.config],
    )
    run_step(
        "Train prefix adapter",
        [
            sys.executable,
            "-m",
            "accelerate.commands.launch",
            "--num_processes",
            "1",
            str(BASE_DIR / "scripts" / "train.py"),
            "--config",
            args.config,
        ],
    )
    run_step(
        "Evaluate final adapter",
        [sys.executable, str(BASE_DIR / "scripts" / "evaluate.py"), "--config", args.config],
    )


if __name__ == "__main__":
    main()

