import json
import random
from collections import defaultdict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BASE_DIR.parent

INPUT_PATH = PROJECT_ROOT / "data" / "llm_qna.jsonl"
OUTPUT_DIR = BASE_DIR / "data"

TRAIN_PATH = OUTPUT_DIR / "train.jsonl"
VALID_PATH = OUTPUT_DIR / "valid.jsonl"
TEST_PATH = OUTPUT_DIR / "test.jsonl"

SYSTEM_PROMPT = (
    "You answer questions factually and concisely. "
    "Do not invent information. "
    "If the answer is unknown, say so clearly."
)

# Qwen3 can disable thinking by adding /no_think,
NO_THINK_SUFFIX = " /no_think"
RANDOM_SEED = 42
TRAIN_RATIO = 0.8
VALID_RATIO = 0.1


def load_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file]


def group_rows(rows: list[dict]) -> dict[str, list[dict]]:
    grouped = defaultdict(list)
    for index, row in enumerate(rows):
        key = row.get("hash", str(index))
        grouped[key].append(row)
    return grouped


def split_keys(keys, seed: int = RANDOM_SEED,) -> tuple[set[str], set[str], set[str]]:
    shuffled_keys = list(keys)
    rng = random.Random(seed)
    rng.shuffle(shuffled_keys)

    total = len(shuffled_keys)
    train_end = int(total * TRAIN_RATIO)
    valid_end = int(total * (TRAIN_RATIO + VALID_RATIO))

    return (
        set(shuffled_keys[:train_end]),
        set(shuffled_keys[train_end:valid_end]),
        set(shuffled_keys[valid_end:]),
    )


def format_example(row: dict) -> dict:
    question = row.get("question", row.get("user", ""))
    answer = row.get("answer", row.get("assistant", ""))

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{question}{NO_THINK_SUFFIX}"},
        {"role": "assistant", "content": answer},
    ]

    assert [message["role"] for message in messages] == ["system", "user", "assistant"]

    return {"messages": messages}


def assign_examples(groups: dict[str, list[dict]], train_keys: set[str], valid_keys: set[str], ) -> tuple[list[dict], list[dict], list[dict]]:
    train, valid, test = [], [], []

    for group_key, rows in groups.items():
        target = (train if group_key in train_keys else valid if group_key in valid_keys else test)
        target.extend(format_example(row) for row in rows)

    return train, valid, test


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def print_summary(train: list[dict], valid: list[dict], test: list[dict]) -> None:
    total = len(train) + len(valid) + len(test)

    print("train samples:", len(train))
    print("valid samples:", len(valid))
    print("test samples:", len(test))

    print(f"train %: {len(train) / total:.2f}")
    print(f"valid %: {len(valid) / total:.2f}")
    print(f"test %: {len(test) / total:.2f}")


def validate_splits(train_keys: set[str], valid_keys: set[str], test_keys: set[str], ) -> None:
    assert train_keys.isdisjoint(valid_keys)
    assert train_keys.isdisjoint(test_keys)
    assert valid_keys.isdisjoint(test_keys)
    print("No hashes intersect between splits.")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = load_rows(INPUT_PATH)
    groups = group_rows(rows)

    train_keys, valid_keys, test_keys = split_keys(groups.keys())
    train, valid, test = assign_examples(groups, train_keys, valid_keys)

    write_jsonl(TRAIN_PATH, train)
    write_jsonl(VALID_PATH, valid)
    write_jsonl(TEST_PATH, test)

    print_summary(train, valid, test)
    validate_splits(train_keys, valid_keys, test_keys)


if __name__ == "__main__":
    main()
