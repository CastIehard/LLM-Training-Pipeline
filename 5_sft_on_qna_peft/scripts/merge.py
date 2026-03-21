import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BASE_DIR.parent

MODEL_PATH = PROJECT_ROOT / "model" / "Qwen_Qwen3-0.6B-cpt5e-3"
ADAPTER_PATH = BASE_DIR / "adapters" / "domain_adapter_cpt5e-3-SFT"
SAVE_PATH = PROJECT_ROOT / "model" / "Qwen_Qwen3-0.6B-cpt5e-3-SFT"

cmd = [
    "mlx_lm.fuse",
    "--model", str(MODEL_PATH),
    "--adapter-path", str(ADAPTER_PATH),
    "--save-path", str(SAVE_PATH)
]

print("Running command:", " ".join(cmd))
subprocess.run(cmd, cwd=BASE_DIR)
print(f"\nModel successfully merged and saved to {SAVE_PATH}!")
