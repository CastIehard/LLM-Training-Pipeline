"""
LLM Question Generation Module for UTN Project.

This script generates Q&A pairs from scraped content using either
OpenAI API or a local LLM (LM Studio). The Q&A pairs are stored
in a JSONL file (one JSON object per line).
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

# Load environment variables from .env file
load_dotenv()


def load_config(config_path: str = "4_llm_questions/config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Replace environment variables in API key
    if config["llm"]["openai"]["api_key"].startswith("${"):
        env_var = config["llm"]["openai"]["api_key"][2:-1]
        config["llm"]["openai"]["api_key"] = os.environ.get(env_var, "")

    return config


def create_llm_client(config: dict) -> OpenAI:
    """Create OpenAI client based on provider setting."""
    provider = config["llm"]["provider"]
    timeout = float(os.getenv("OPENAI_TIMEOUT", "60"))
    max_retries = int(os.getenv("OPENAI_MAX_RETRIES", "2"))

    if provider == "openai":
        api_key = config["llm"]["openai"]["api_key"]
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        return OpenAI(api_key=api_key, timeout=timeout, max_retries=max_retries)
    elif provider == "local":
        return OpenAI(
            base_url=config["llm"]["local"]["base_url"],
            api_key="lm-studio",
            timeout=timeout,
        )
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")


def get_llm_settings(config: dict) -> dict:
    """Get model settings based on provider."""
    provider = config["llm"]["provider"]
    if provider == "openai":
        return config["llm"]["openai"]
    return config["llm"]["local"]


def load_md_content(raw_md_dir: str, filename: str) -> str:
    """Load full content from a markdown file."""
    filepath = Path(raw_md_dir) / filename
    if not filepath.exists():
        return ""

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""


def calculate_question_count(content_length: int, config: dict) -> tuple[int, int]:
    """
    Calculate min and max number of questions based on content length.

    Returns (min_questions, max_questions) tuple.
    """
    gen_config = config["generation"]
    length_per_q = gen_config["content_length_per_question"]
    q_range = gen_config["question_range"]
    min_q = gen_config["min_questions"]
    max_q = gen_config["max_questions"]

    # Calculate base number of questions
    base_questions = max(1, content_length // length_per_q)

    # Apply range
    calc_min = max(min_q, base_questions - q_range)
    calc_max = min(max_q, base_questions + q_range)

    # Ensure min <= max
    if calc_min > calc_max:
        calc_min = calc_max

    return calc_min, calc_max


def build_prompt(config: dict, content: str, num_min: int, num_max: int) -> str:
    """Build the Q&A generation prompt."""
    return config["prompt"].format(
        num_questions_min=num_min,
        num_questions_max=num_max,
        content=content,
    )


@retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(5))
def generate_qa_with_retry(client: OpenAI, settings: dict, prompt: str) -> list[dict]:
    """
    Send content to LLM and get Q&A pairs with automatic retry.

    Returns list of {"question": ..., "answer": ...} dicts.
    """
    response = client.chat.completions.create(
        model=settings["model"],
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that generates question-answer pairs. Always respond with valid JSON.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=settings["temperature"],
        max_tokens=settings["max_tokens"],
    )

    content = response.choices[0].message.content.strip()

    # Handle markdown code blocks
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
        content = content.strip()

    # Parse JSON
    qa_pairs = json.loads(content)

    # Validate format
    if not isinstance(qa_pairs, list):
        raise ValueError(f"Expected list, got {type(qa_pairs).__name__}")

    for pair in qa_pairs:
        if "question" not in pair or "answer" not in pair:
            raise ValueError(f"Invalid Q&A pair format: {pair}")

    return qa_pairs


def generate_qa(
    client: OpenAI, config: dict, content: str, num_min: int, num_max: int
) -> list[dict] | None:
    """
    Generate Q&A pairs from content.

    Returns list of Q&A pairs or None if generation fails.
    """
    prompt = build_prompt(config, content, num_min, num_max)
    settings = get_llm_settings(config)

    try:
        return generate_qa_with_retry(client, settings, prompt)
    except Exception as e:
        actual_error = e.__cause__ if e.__cause__ else e
        print(f"\n  API Error: {type(actual_error).__name__}")
        print(f"  Full response: {actual_error}")
        if hasattr(actual_error, "response"):
            print(f"  Response body: {actual_error.response}")
        if hasattr(actual_error, "body"):
            print(f"  Error body: {actual_error.body}")
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
    """Load hashes that have already been processed from JSONL file."""
    processed = set()
    if not Path(output_file).exists():
        return processed

    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entry = json.loads(line)
                    processed.add(entry.get("hash", ""))
                except json.JSONDecodeError:
                    continue

    return processed


def append_qa_to_file(
    output_file: str, hash_id: str, qa_pairs: list[dict], model_name: str, category: str
) -> int:
    """
    Append Q&A pairs to JSONL file.

    Each Q&A pair becomes one line with: hash, question, answer, model, category
    Returns number of lines written.
    """
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


def main():
    """Main function to run the Q&A generation pipeline."""
    print("=" * 60)
    print("LLM Q&A Generation")
    print("=" * 60)

    # Load configuration
    config = load_config()
    provider = config["llm"]["provider"]
    settings = get_llm_settings(config)
    model_name = settings["model"]

    print(f"\nProvider: {provider}")
    print(f"Model: {model_name}")
    print(
        f"Content length per question: {config['generation']['content_length_per_question']}"
    )

    # Load index data
    index_path = config["data"]["index_file"]
    raw_md_dir = config["data"]["raw_md_dir"]
    output_file = config["data"]["output_file"]

    print(f"\nLoading index from: {index_path}")
    index_data = load_index(index_path)
    print(f"Total entries: {len(index_data)}")

    # Load already processed hashes
    processed_hashes = load_processed_hashes(output_file)
    print(f"Already processed: {len(processed_hashes)}")

    # Filter entries to process
    skip_trash = config["processing"]["skip_trash"]
    to_process = []
    for entry in index_data:
        hash_id = entry.get("hash", "")
        category = entry.get("category", "")

        # Skip entries already marked as questions generated
        if entry.get("questions_generated") is True:
            continue

        # Skip already processed (fallback check via JSONL file)
        if hash_id in processed_hashes:
            continue

        # Only process documents that passed data cleaning
        if entry.get("cleaning_status") != "kept":
            continue

        # Skip trash if configured
        if skip_trash and category == "Trash":
            continue

        to_process.append(entry)

    print(f"Pending processing: {len(to_process)}")

    if not to_process:
        print("\nAll entries are already processed. Nothing to do.")
        return

    # Create LLM client
    print(f"\nInitializing {provider} client...")
    client = create_llm_client(config)

    # Process entries
    delay = config["processing"]["delay"]
    total_questions = 0
    failed_count = 0
    start_time = time.time()

    print("\nStarting Q&A generation...")

    for entry in tqdm(to_process, desc="Generating Q&A"):
        hash_id = entry["hash"]
        filename = entry.get("filename", "")
        content_length = entry.get("content_length", 0)
        category = entry.get("category", "")

        # Load full content
        content = load_md_content(raw_md_dir, filename)
        if not content:
            tqdm.write(f"  Skipping {hash_id[:8]}: empty or missing file")
            failed_count += 1
            continue

        # Calculate question count
        num_min, num_max = calculate_question_count(content_length, config)

        # Generate Q&A
        qa_pairs = generate_qa(client, config, content, num_min, num_max)

        if qa_pairs:
            written = append_qa_to_file(
                output_file, hash_id, qa_pairs, model_name, category
            )
            total_questions += written
            entry["questions_generated"] = True
        else:
            failed_count += 1

        # Delay between requests
        time.sleep(delay)

        # Save progress periodically (every 10 entries)
        if (total_questions + failed_count) % 10 == 0:
            save_index(index_path, index_data)

    # Final save
    save_index(index_path, index_data)

    # Summary
    duration = time.time() - start_time
    print(f"\nQ&A generation complete in {duration:.2f} seconds!")
    print(f"Total questions generated: {total_questions}")
    print(f"Failed entries: {failed_count}")
    print(f"Output saved to: {output_file}")


if __name__ == "__main__":
    main()
