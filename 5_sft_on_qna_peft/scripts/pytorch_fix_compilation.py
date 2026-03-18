import torch
from pathlib import Path
from safetensors.torch import load_file, save_file

model_dir = Path("model/Qwen_Qwen3-0.6B-SFT_full_15000")
src = model_dir / "model.safetensors"
dst = model_dir / "model_fixed.safetensors"

state = load_file(str(src))
fixed_state = {}

for k, v in state.items():
    new_k = k.removeprefix("_orig_mod.")
    fixed_state[new_k] = v

save_file(fixed_state, str(dst))
print("Saved fixed checkpoint to", dst)