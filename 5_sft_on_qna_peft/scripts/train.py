import subprocess
import sys
from pathlib import Path
import yaml

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "lora_config.yaml"


def run_training():
    cmd = [
        sys.executable,
        "-m",
        "mlx_lm.lora",
        "--config",
        str(CONFIG_PATH),
        "--train",
        "--mask-prompt",
    ]

    print("Running training command:")
    print(" ".join(cmd))
    subprocess.run(cmd, cwd=BASE_DIR, check=True)


def run_merge():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    model_value = config["model"]
    adapter_value = config["adapter_path"]

    MODEL_PATH = (CONFIG_PATH.parent / model_value).resolve()
    ADAPTER_PATH = (CONFIG_PATH.parent / adapter_value).resolve()
    SAVE_PATH = MODEL_PATH.parent / f"{ADAPTER_PATH.name}"

    cmd = [
        "mlx_lm.fuse",
        "--model", str(MODEL_PATH),
        "--adapter-path", str(ADAPTER_PATH),
        "--save-path", str(SAVE_PATH),
    ]

    print("\nRunning merge command:")
    print(" ".join(cmd))
    subprocess.run(cmd, cwd=BASE_DIR, check=True)
    print(f"\nModel successfully merged and saved to {SAVE_PATH}!")


def main():
    try:
        run_training()
        print("\nTraining completed successfully. Starting merge...")
        run_merge()
    except subprocess.CalledProcessError as e:
        print(f"\nA subprocess failed with exit code {e.returncode}.")
        sys.exit(e.returncode)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()