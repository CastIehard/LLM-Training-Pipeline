"""LLM Classification and Q&A Generation Module for UTN Project.

This script combines document classification and Q&A generation into a single LLM
call. For each cleaned document it asks the LLM to:

  1. Decide whether the content is relevant (UTN, Germany, Nuremberg, or studying).
     Content about other universities (e.g. TU Munich, FAU) is treated as irrelevant.
  2. If relevant: classify the document into a category and generate Q&A pairs.
  3. If irrelevant: mark the document and skip Q&A generation.

Results are stored in:
  - data/index.json   → updated with `category` and `llm_processed` flags
  - data/llm_qna.jsonl → Q&A pairs (one JSON object per line)
"""

import json
import os
import time
from pathlib import Path

import yaml
from dotenv import load_dotenv
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm.auto import tqdm

load_dotenv()

VALID_CATEGORIES = {"UTN", "Germany", "Nuremberg", "Studies"}
IRRELEVANT_CATEGORY = "Other"


def load_config(
    config_path: str = "3_llm_classification_and_questions/config.yaml",
) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    api_key_value = config["llm"]["openai"]["api_key"]
    if api_key_value.startswith("${"):
        env_var = api_key_value[2:-1]
        config["llm"]["openai"]["api_key"] = os.environ.get(env_var, "")

    return config


def create_llm_client(config: dict) -> OpenAI:
    """Create an OpenAI-compatible client based on the configured provider."""
    provider = config["llm"]["provider"]
    timeout = float(os.getenv("OPENAI_TIMEOUT", "120"))
    max_retries = int(os.getenv("OPENAI_MAX_RETRIES", "2"))

    if provider == "openai":
        api_key = config["llm"]["openai"]["api_key"]
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        return OpenAI(api_key=api_key, timeout=timeout, max_retries=max_retries)
    elif provider == "local":
        return OpenAI(
            base_url=config["llm"]["local"]["base_url"],
            api_key="lm-studio",  # LM Studio does not require a real API key
            timeout=timeout,
        )
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")


def get_llm_settings(config: dict) -> dict:
    """Return the model settings for the active provider."""
    provider = config["llm"]["provider"]
    return config["llm"]["openai"] if provider == "openai" else config["llm"]["local"]


def load_md_content(raw_md_dir: str, filename: str) -> str:
    """Load full content from a markdown file. Returns empty string on failure."""
    filepath = Path(raw_md_dir) / filename
    if not filepath.exists():
        return ""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""


def calculate_question_count(content_length: int, config: dict) -> tuple[int, int]:
    """Calculate the min/max number of questions based on document length."""
    gen = config["generation"]
    base = max(1, content_length // gen["content_length_per_question"])
    q_min = max(gen["min_questions"], base - gen["question_range"])
    q_max = min(gen["max_questions"], base + gen["question_range"])
    if q_min > q_max:
        q_min = q_max
    return q_min, q_max


def build_prompt(config: dict, content: str, num_min: int, num_max: int) -> str:
    """Build the combined classification and Q&A generation prompt."""
    return config["prompt"].format(
        num_questions_min=num_min,
        num_questions_max=num_max,
        content=content,
    )


@retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(5))
def call_llm_with_retry(client: OpenAI, settings: dict, prompt: str) -> dict | str:
    """Send a document to the LLM and parse the response.

    Returns:
        "irrelevant" if the LLM marks the document as irrelevant.
        dict with keys "category" (str) and "qa_pairs" (list) if relevant.

    Raises:
        ValueError  if the response cannot be parsed or fails validation.
        json.JSONDecodeError  if the JSON payload is malformed (triggers retry).
    """
    response = client.chat.completions.create(
        model=settings["model"],
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a document analyzer. "
                    "Follow the instructions exactly and respond only as specified."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=settings["temperature"],
        max_tokens=settings["max_tokens"],
    )

    raw = response.choices[0].message.content.strip()

    # Irrelevant response is a single word
    if raw.lower() == "irrelevant":
        return "irrelevant"

    # Strip markdown code fences if the model added them
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    result = json.loads(raw)

    if "category" not in result or "qa_pairs" not in result:
        raise ValueError(f"Response is missing required keys: {result}")

    if result["category"] not in VALID_CATEGORIES:
        raise ValueError(
            f"Invalid category '{result['category']}', "
            f"expected one of {sorted(VALID_CATEGORIES)}"
        )

    if not isinstance(result["qa_pairs"], list):
        raise ValueError("'qa_pairs' must be a list")

    for pair in result["qa_pairs"]:
        if "question" not in pair or "answer" not in pair:
            raise ValueError(f"Invalid Q&A pair format: {pair}")

    return result


def process_document(
    client: OpenAI, config: dict, content: str, num_min: int, num_max: int
) -> dict | str | None:
    """Classify a document and generate Q&A pairs in one LLM call.

    Returns:
        "irrelevant" — document is not relevant to UTN/Germany/studying.
        dict          — {"category": ..., "qa_pairs": [...]} for relevant documents.
        None          — all retry attempts failed.
    """
    prompt = build_prompt(config, content, num_min, num_max)
    settings = get_llm_settings(config)

    try:
        return call_llm_with_retry(client, settings, prompt)
    except Exception as e:
        actual_error = e.__cause__ if e.__cause__ else e
        print(f"\n  API Error: {type(actual_error).__name__}: {actual_error}")
        return None


def load_index(index_path: str) -> list[dict]:
    """Load the index.json file."""
    with open(index_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_index(index_path: str, data: list[dict]) -> None:
    """Save the index.json file."""
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_processed_hashes(output_file: str) -> set[str]:
    """Return the set of document hashes already written to the JSONL output file."""
    processed: set[str] = set()
    if not Path(output_file).exists():
        return processed

    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    processed.add(json.loads(line).get("hash", ""))
                except json.JSONDecodeError:
                    continue

    return processed


def append_qa_to_file(
    output_file: str,
    hash_id: str,
    qa_pairs: list[dict],
    model_name: str,
    category: str,
) -> int:
    """Append Q&A pairs to the JSONL output file. Returns the number of pairs written."""
    with open(output_file, "a", encoding="utf-8") as f:
        for pair in qa_pairs:
            entry = {
                "hash": hash_id,
                "question": pair["question"],
                "answer": pair["answer"],
                "model": model_name,
                "category": category,
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return len(qa_pairs)


def needs_processing(entry: dict, processed_hashes: set[str]) -> bool:
    """Return True if the entry still needs LLM processing."""
    if entry.get("llm_processed") is True:
        return False
    if entry.get("hash", "") in processed_hashes:
        return False
    return True


def main():
    """Run the combined classification and Q&A generation pipeline."""
    print("=" * 60)
    print("LLM Classification and Q&A Generation")
    print("=" * 60)

    config = load_config()
    provider = config["llm"]["provider"]
    settings = get_llm_settings(config)
    model_name = settings["model"]

    print(f"\nProvider: {provider}")
    print(f"Model: {model_name}")

    index_path = config["data"]["index_file"]
    raw_md_dir = config["data"]["raw_md_dir"]
    output_file = config["data"]["output_file"]

    print(f"\nLoading index from: {index_path}")
    index_data = load_index(index_path)
    print(f"Total entries: {len(index_data)}")

    processed_hashes = load_processed_hashes(output_file)
    print(f"Already in output file: {len(processed_hashes)}")

    to_process = [
        entry
        for entry in index_data
        if entry.get("cleaning_status") == "kept"
        and needs_processing(entry, processed_hashes)
    ]
    print(f"Pending: {len(to_process)}")

    if not to_process:
        print("\nAll entries are already processed. Nothing to do.")
        return

    print(f"\nInitializing {provider} client...")
    client = create_llm_client(config)

    delay = config["processing"]["delay"]
    total_questions = 0
    irrelevant_count = 0
    failed_count = 0
    entries_processed = 0
    start_time = time.time()

    print("\nStarting classification and Q&A generation...")

    for entry in tqdm(to_process, desc="Processing Documents"):
        hash_id = entry["hash"]
        filename = entry.get("filename", "")
        content_length = entry.get("content_length", 0)

        content = load_md_content(raw_md_dir, filename)
        if not content:
            tqdm.write(f"  Skipping {hash_id[:8]}: empty or missing file")
            failed_count += 1
            entries_processed += 1
            continue

        num_min, num_max = calculate_question_count(content_length, config)
        result = process_document(client, config, content, num_min, num_max)

        if result is None:
            failed_count += 1
        elif result == "irrelevant":
            entry["category"] = IRRELEVANT_CATEGORY
            entry["llm_processed"] = True
            irrelevant_count += 1
        else:
            entry["category"] = result["category"]
            written = append_qa_to_file(
                output_file, hash_id, result["qa_pairs"], model_name, result["category"]
            )
            entry["llm_processed"] = True
            entry["questions_generated"] = True
            total_questions += written

        entries_processed += 1
        time.sleep(delay)

        if entries_processed % 10 == 0:
            save_index(index_path, index_data)
            tqdm.write(f"  [Progress saved: {entries_processed} entries processed]")

    save_index(index_path, index_data)

    duration = time.time() - start_time
    print(f"\nCompleted in {duration:.2f} seconds!")
    print(f"Total questions generated: {total_questions}")
    print(f"Irrelevant documents: {irrelevant_count}")
    print(f"Failed entries: {failed_count}")
    print(f"Output saved to: {output_file}")


if __name__ == "__main__":
    main()
