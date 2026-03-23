import subprocess
from pathlib import Path
import yaml

BASE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BASE_DIR.parent
CONFIG_PATH = BASE_DIR / "lora_config.yaml"

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

model_value = config["model"]
adapter_value = config["adapter_path"]

MODEL_PATH = (CONFIG_PATH.parent / model_value).resolve()
ADAPTER_PATH = (CONFIG_PATH.parent / adapter_value).resolve()
SAVE_PATH = MODEL_PATH.parent / f"{MODEL_PATH.name}-SFT"

cmd = [
    "mlx_lm.fuse",
    "--model", str(MODEL_PATH),
    "--adapter-path", str(ADAPTER_PATH),
    "--save-path", str(SAVE_PATH),
]

print("Running command:", " ".join(cmd))
subprocess.run(cmd, cwd=BASE_DIR, check=True)
print(f"\nModel successfully merged and saved to {SAVE_PATH}!")