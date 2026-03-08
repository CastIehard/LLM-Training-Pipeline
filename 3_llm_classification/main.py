"""
LLM Classification Module for UTN Project.

This script classifies scraped URLs into predefined categories using either
OpenAI API or a local LLM (LM Studio). The classification results are stored
in the index.json file.
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


def load_config(config_path: str = "3_llm_classification/config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Replace environment variables in API key
    if config["llm"]["openai"]["api_key"].startswith("${"):
        env_var = config["llm"]["openai"]["api_key"][2:-1]
        config["llm"]["openai"]["api_key"] = os.environ.get(env_var, "")

    return config


def get_valid_categories(config: dict) -> list[str]:
    """Extract valid category names from config."""
    return [cat["name"] for cat in config["categories"]]


def load_md_content(raw_md_dir: str, filename: str, max_lines: int = 50) -> str:
    """Load first N lines from a markdown file."""
    filepath = Path(raw_md_dir) / filename
    if not filepath.exists():
        return "[File not found]"

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            lines = []
            for i, line in enumerate(f):
                if i >= max_lines:
                    break
                lines.append(line.rstrip())
            return "\n".join(lines) if lines else "[Empty file]"
    except Exception as e:
        return f"[Error reading file: {e}]"


def build_prompt(config: dict, url: str, content: str) -> str:
    """Build the classification prompt with categories, URL, and content."""
    categories_text = "\n".join(
        f"- {cat['name']}: {cat['description']}" for cat in config["categories"]
    )
    prompt = config["prompt"].format(
        categories=categories_text, url=url, content=content
    )
    return prompt


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
        # LM Studio compatible client
        return OpenAI(
            base_url=config["llm"]["local"]["base_url"],
            api_key="lm-studio",  # LM Studio doesn't require real API key
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


@retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(5))
def classify_url_with_retry(
    client: OpenAI, settings: dict, prompt: str, valid_categories: list[str]
) -> str:
    """
    Send URL to LLM for classification with automatic retry.

    Retries are handled by the @retry decorator (tenacity).
    """
    response = client.chat.completions.create(
        model=settings["model"],
        messages=[
            {
                "role": "system",
                "content": "You are a URL classifier. Respond with only the category name.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=settings["temperature"],
        max_tokens=settings["max_tokens"],
    )

    # Extract and validate response
    category = response.choices[0].message.content.strip()

    # Check if the response is a valid category
    if category in valid_categories:
        return category

    # Try to match partial or case-insensitive
    for valid_cat in valid_categories:
        if valid_cat.lower() in category.lower():
            return valid_cat

    raise ValueError(f"Invalid category response: '{category}'")


def classify_url(client: OpenAI, config: dict, url: str, content: str) -> str | None:
    """
    Send URL and content to LLM for classification.

    Returns the category name or None if classification fails.
    """
    prompt = build_prompt(config, url, content)
    settings = get_llm_settings(config)
    valid_categories = get_valid_categories(config)

    try:
        return classify_url_with_retry(client, settings, prompt, valid_categories)
    except Exception as e:
        actual_error = e.__cause__ if e.__cause__ else e
        print(f"\n  API Error: {type(actual_error).__name__}")
        print(f"  Full response: {actual_error}")
        # If it's an OpenAI API error, try to print more details
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


def needs_classification(entry: dict, valid_categories: list[str]) -> bool:
    """
    Check if an entry needs classification.

    Returns True if:
    - No 'category' field exists
    - 'category' is None or empty
    - 'category' is not in the list of valid categories
    """
    if "category" not in entry:
        return True
    if not entry["category"]:
        return True
    if entry["category"] not in valid_categories:
        return True
    return False


def main():
    """Main function to run the classification pipeline."""
    print("=" * 60)
    print("LLM URL Classification")
    print("=" * 60)

    # Load configuration
    config = load_config()
    provider = config["llm"]["provider"]
    valid_categories = get_valid_categories(config)

    print(f"\nProvider: {provider}")
    print(f"Categories: {', '.join(valid_categories)}")

    # Load index data
    index_path = config["data"]["index_file"]
    raw_md_dir = config["data"]["raw_md_dir"]
    print(f"\nLoading index from: {index_path}")
    print(f"Reading content from: {raw_md_dir}")
    index_data = load_index(index_path)
    print(f"Total entries: {len(index_data)}")

    # Count entries that need classification (only cleaned/kept documents)
    to_classify_indices = [
        i
        for i, e in enumerate(index_data)
        if e.get("cleaning_status") == "kept"
        and needs_classification(e, valid_categories)
    ]
    cleaned_count = sum(1 for e in index_data if e.get("cleaning_status") == "kept")
    already_classified = cleaned_count - len(to_classify_indices)

    print(f"Already classified: {already_classified}")
    print(f"Pending classification: {len(to_classify_indices)}")

    if not to_classify_indices:
        print("\nAll entries are already classified. Nothing to do.")
        return

    # Create LLM client
    print(f"\nInitializing {provider} client...")
    client = create_llm_client(config)

    # Process entries
    delay = config["processing"]["delay"]
    classified_count = 0
    failed_count = 0
    start_time = time.time()

    print("\nStarting classification...")

    for idx in tqdm(to_classify_indices, desc="Classifying Contents"):
        entry = index_data[idx]
        url = entry["source_url"]
        filename = entry.get("filename", "")

        # Load content from MD file
        content = load_md_content(raw_md_dir, filename, max_lines=50)

        category = classify_url(client, config, url, content)

        if category:
            entry["category"] = category
            classified_count += 1
        else:
            entry["category"] = "Sonstiges"  # Default fallback
            failed_count += 1

        # Delay between requests
        time.sleep(delay)

        # Save progress periodically (every 10 entries)
        if (classified_count + failed_count) % 10 == 0:
            save_index(index_path, index_data)
            tqdm.write(
                f"  [Progress saved: {classified_count + failed_count} entries processed]"
            )

    # Final save
    save_index(index_path, index_data)

    # Summary
    duration = time.time() - start_time
    print(f"\nClassification complete in {duration:.2f} seconds!")
    print(f"Successfully classified: {classified_count}")
    print(f"Failed (defaulted to 'Sonstiges'): {failed_count}")
    print(f"Index saved to: {index_path}")


if __name__ == "__main__":
    main()
