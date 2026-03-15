import json
import random
from collections import defaultdict
from pathlib import Path
import os

# Adapt paths to the actual project structure
BASE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BASE_DIR.parent

INPUT = PROJECT_ROOT / "data" / "llm_qna.jsonl"
OUT_DIR = BASE_DIR / "data"
os.makedirs(OUT_DIR, exist_ok=True)

TRAIN_OUT = OUT_DIR / "train.jsonl"
VALID_OUT = OUT_DIR / "valid.jsonl"
TEST_OUT  = OUT_DIR / "test.jsonl"

SYSTEM_PROMPT = (
    "You answer questions factually and concisely. "
    "Do not invent information. "
    "If the answer is unknown, say so clearly."
)

rows = []
with open(INPUT, "r") as f:
    for line in f:
        rows.append(json.loads(line))

groups = defaultdict(list)

# Use 'hash' if available, otherwise just use a generated sequence or an available ID
for i, r in enumerate(rows):
    # Depending on format, hash could be missing
    h = r.get("hash", str(i))
    groups[h].append(r)

hashes = list(groups.keys())
random.seed(42)
random.shuffle(hashes)

n = len(hashes)

train_hash = set(hashes[:int(0.8*n)])
valid_hash = set(hashes[int(0.8*n):int(0.9*n)])
test_hash  = set(hashes[int(0.9*n):])

train = []
valid = []
test = []

def format_example(r):
    # Handle the structure if it doesn't match exactly
    question = r.get("question", r.get("user", ""))
    answer = r.get("answer", r.get("assistant", ""))
    return {
        "messages":[
            {"role":"system","content":SYSTEM_PROMPT},
            {"role":"user","content":question},
            {"role":"assistant","content":answer}
        ]
    }

for h,items in groups.items():
    for r in items:
        example = format_example(r)
        
        # Validation Check 3
        assert example["messages"][0]["role"] == "system"
        assert example["messages"][1]["role"] == "user"
        assert example["messages"][2]["role"] == "assistant"

        if h in train_hash:
            train.append(example)
        elif h in valid_hash:
            valid.append(example)
        else:
            test.append(example)

def write(path,data):
    with open(path,"w") as f:
        for row in data:
            f.write(json.dumps(row,ensure_ascii=False)+"\n")

write(TRAIN_OUT,train)
write(VALID_OUT,valid)
write(TEST_OUT,test)

print("train samples:",len(train))
print("valid samples:",len(valid))
print("test samples:",len(test))

# Validation Check 1
total = len(train) + len(valid) + len(test)
print(f"train %: {len(train)/total:.2f}")
print(f"valid %: {len(valid)/total:.2f}")
print(f"test %: {len(test)/total:.2f}")

# Validation Check 2
assert len(train_hash.intersection(valid_hash)) == 0
assert len(train_hash.intersection(test_hash)) == 0
assert len(valid_hash.intersection(test_hash)) == 0
print("No hashes intersect between splits.")