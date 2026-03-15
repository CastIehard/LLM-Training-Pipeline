import subprocess
import sys
from pathlib import Path

# Adjust paths relative to script
BASE_DIR = Path(__file__).resolve().parent.parent

cmd = [
    sys.executable,
    "-m",
    "mlx_lm.lora",
    "--config",
    str(BASE_DIR / "lora_config.yaml"),
    "--train",
    "--mask-prompt"  # Enable masking prompt
]

print("Running command:", " ".join(cmd))
subprocess.run(cmd, cwd=BASE_DIR)
