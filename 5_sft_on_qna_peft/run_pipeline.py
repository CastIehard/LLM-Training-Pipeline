import subprocess
import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

def run_step(name, script):
    print(f"\n{'='*40}\nStarting: {name}\n{'='*40}")
    result = subprocess.run([sys.executable, str(script)], cwd=BASE_DIR)
    if result.returncode != 0:
        print(f"Error executing {name}. Exiting.")
        sys.exit(result.returncode)

def main():
    preprocess_script = BASE_DIR / "scripts" / "preprocess.py"
    train_script = BASE_DIR / "scripts" / "train.py"
    evaluate_script = BASE_DIR / "scripts" / "evaluate.py"

    run_step("Data Preprocessing", preprocess_script)
    run_step("Model Training", train_script)
    run_step("Model Evaluation", evaluate_script)

    print("\nTraining Pipeline Complete!")
    print("Run inference with:")
    print("mlx_lm.generate --model ../model/Qwen_Qwen3-0.6B --adapter-path ./adapters/domain_adapter_v2 --prompt 'your question'")

if __name__ == "__main__":
    main()
