import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

cmd = [
    sys.executable,
    "-m",
    "mlx_lm.lora",
    "--model",
    str(BASE_DIR.parent / "model/Qwen_Qwen3-0.6B"),
    "--adapter-path",
    str(BASE_DIR / "adapters/domain_adapter"),
    "--data",
    str(BASE_DIR / "data"),
    "--test"
]

print("Running command:", " ".join(cmd))
subprocess.run(cmd, cwd=BASE_DIR)
