from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = Path("./final_submission").expanduser().resolve()

SYSTEM_PROMPT = " "
PROMPT = "Give me a short introduction to large language models."
MAX_NEW_TOKENS = 1024

model = AutoModelForCausalLM.from_pretrained(
    str(MODEL_PATH),
    torch_dtype="auto",
    device_map="auto",
    local_files_only=True,
)
tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), local_files_only=True)

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": PROMPT},
]

if hasattr(tokenizer, "apply_chat_template"):
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
else:
    text = f"{SYSTEM_PROMPT}\n\nUser: {PROMPT}\nAssistant:"

model_inputs = tokenizer([text], return_tensors="pt")
model_inputs = {k: v.to(model.device) for k, v in model_inputs.items()}

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=MAX_NEW_TOKENS,
)

generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs["input_ids"], generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)
