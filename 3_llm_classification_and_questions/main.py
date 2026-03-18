"""LLM Classification and Q&A Generation Module for UTN Project.

This script combines document classification and Q&A generation into a single LLM
call. For each cleaned document it asks the LLM to:

  1. Decide whether the content is relevant.
  2. If relevant: classify the document into a category and generate Q&A pairs.
  3. Simple Post-Processing (Blacklist/Whitelist).
  4. Second LLM Pass to verify if the question is standalone and high-quality.

Results are stored in:
  - data/index.json   → updated with `category` and `llm_processed` flags
  - data/llm_qna.jsonl → Q&A pairs (one JSON object per line)
"""

import json
import os
import re
import time
from pathlib import Path

import yaml
from dotenv import load_dotenv
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm.auto import tqdm

load_dotenv()

VALID_CATEGORIES = {"UTN", "Germany", "Nuremberg", "Studies"}
IRRELEVANT_CATEGORY = "Irrelevant"


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
    """Send a document to the LLM and parse the response."""
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

    if raw.lower() == "irrelevant":
        return "irrelevant"

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
    """Classify a document and generate Q&A pairs in one LLM call."""
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
    """Append Q&A pairs to the JSONL output file."""
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


def chunk_content(content: str, max_chunk_size: int = 10000) -> list[str]:
    """Split content into chunks of max_chunk_size characters."""
    if len(content) <= max_chunk_size:
        return [content]
    
    chunks = []
    lines = content.split('\n')
    current_chunk = []
    current_length = 0
    
    for line in lines:
        line_len = len(line) + 1 # +1 for newline
        if current_length + line_len > max_chunk_size and current_chunk:
            chunks.append('\n'.join(current_chunk))
            current_chunk = [line]
            current_length = line_len
        else:
            current_chunk.append(line)
            current_length += line_len
            
    if current_chunk:
        chunks.append('\n'.join(current_chunk))
        
    return chunks


def needs_processing(entry: dict, processed_hashes: set[str]) -> bool:
    """Return True if the entry still needs LLM processing."""
    if entry.get("llm_processed") is True:
        return False
    if entry.get("hash", "") in processed_hashes:
        return False
    return True


def filter_generated_qa(output_file: str, config: dict) -> None:
    """Filter out generated Q&A containing excluded keywords, unless they contain whitelisted terms."""
    out_path = Path(output_file)
    removed_path = Path(str(out_path).replace(".jsonl", "_removed.json"))
    
    filtering_cfg = config.get("filtering", {})
    blacklist_terms = filtering_cfg.get("blacklist", [])
    whitelist_terms = filtering_cfg.get("whitelist", [])

    if not out_path.exists():
        return

    all_questions = []
    
    with open(out_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                all_questions.append(line)
                
    if removed_path.exists():
        print(f"Retrieving previously removed questions from {removed_path} for re-evaluation...")
        with open(removed_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    all_questions.append(line)
        open(removed_path, "w").close()

    if not blacklist_terms:
        with open(out_path, "w", encoding="utf-8") as f:
            f.writelines(all_questions)
        return

    kept_lines = []
    removed_lines = []

    for line in all_questions:
        try:
            qna_obj = json.loads(line)
        except Exception:
            qna_obj = {"raw": line}

        # IMPORTANT: Ignore items removed by Second LLM Pass
        if qna_obj.get("llm_rejected"):
            removed_lines.append(line)
            continue

        blacklist_hit = None
        for term in blacklist_terms:
            if term in line:
                blacklist_hit = term
                break

        whitelist_hit = None
        if blacklist_hit:
            for term in whitelist_terms:
                if term in line:
                    whitelist_hit = term
                    break

        if blacklist_hit and not whitelist_hit:
            qna_obj["reason"] = f"Removed due to blacklist term '{blacklist_hit}'"
            removed_lines.append(json.dumps(qna_obj, ensure_ascii=False) + "\n")
        else:
            kept_lines.append(line)

    with open(out_path, "w", encoding="utf-8") as f:
        f.writelines(kept_lines)
            
    if removed_lines:
        print(f"\nSimple Filter output: {len(kept_lines)} kept, {len(removed_lines)} removed.")
        with open(removed_path, "w", encoding="utf-8") as f:
            f.writelines(removed_lines)
        print(f"Removed items are in: {removed_path}")
    else:
        print("\nAll questions passed the simple filtering rules.")


def save_qnas_safely(filepath: str, qnas: list[dict]) -> None:
    """Safely rewrite the JSONL file to ensure we don't lose data if it crashes mid-write."""
    tmp_path = str(filepath) + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        for qna in qnas:
            f.write(json.dumps(qna, ensure_ascii=False) + "\n")
    os.replace(tmp_path, filepath)


def append_to_removed(filepath: str, qna: dict) -> None:
    """Append a single rejected Q&A to the removed file."""
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(qna, ensure_ascii=False) + "\n")


def second_pass_llm_filter(output_file: str, config: dict, client: OpenAI) -> None:
    """Second LLM pass to evaluate Q&A pairs one by one for high quality."""
    prompt_template = config.get("second_pass", {}).get("prompt", "")
    if not prompt_template or not config.get("second_pass", {}).get("enabled", False):
        return

    out_path = Path(output_file)
    removed_path = Path(str(out_path).replace(".jsonl", "_removed.json"))

    if not out_path.exists():
        return

    settings = get_llm_settings(config)
    delay = config.get("processing", {}).get("delay", 0)

    # Load all current good questions
    all_qnas = []
    with open(out_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    all_qnas.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    # Count how many still need to be processed
    to_process_count = sum(1 for q in all_qnas if "second_pass_approved" not in q)

    if to_process_count == 0:
        print("\nAll questions have already passed the Second Pass LLM filter.")
        return

    print(f"\nStarting Second Pass LLM Filtering on {to_process_count} unchecked Q&A pairs...")

    bar = tqdm(total=to_process_count, desc="LLM 2nd Pass Filter", unit="qna", dynamic_ncols=True)

    kept_count = 0
    removed_count = 0

    i = 0
    while i < len(all_qnas):
        qna = all_qnas[i]

        # Skip if already processed previously
        if "second_pass_approved" in qna:
            i += 1
            continue

        question = qna.get("question", "")
        answer = qna.get("answer", "")
        prompt = prompt_template.format(question=question, answer=answer)

        try:
            response = client.chat.completions.create(
                model=settings["model"],
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=500,
            )
            raw_decision = response.choices[0].message.content.strip().lower()
            clean_decision = re.sub(r'[^a-z]', '', raw_decision)
            
            # Use simple exact matching since prompt asks exactly for the word
            if clean_decision == "relevant":
                is_good = True
            elif clean_decision == "irrelevant":
                is_good = False
            else:
                raise ValueError(f"Second pass LLM response is not 'relevant' or 'irrelevant': {raw_decision}")
                
        except Exception as e:
            tqdm.write(f"  Error evaluating Q&A via LLM: {e}")
            time.sleep(3)
            continue
            
        tqdm.write(f"Question: {question[:80]}... | Decision: {raw_decision}")                         
        
        if is_good:
            # Tag as approved, KEEP in all_qnas, overwrite file
            qna["second_pass_approved"] = True
            save_qnas_safely(out_path, all_qnas)
            kept_count += 1
            i += 1
        else:
            # Tag as rejected, remove from all_qnas, append to removed list, overwrite main file
            qna["second_pass_approved"] = False
            qna["llm_rejected"] = True
            qna["reason"] = "Removed by LLM in second pass (non-standalone, irrelevant, or bad format)"
            
            append_to_removed(str(removed_path), qna)
            
            all_qnas.pop(i)
            save_qnas_safely(out_path, all_qnas)
            removed_count += 1
            # i stays the same because the list shifted left

        bar.update(1)
        bar.set_postfix(kept=kept_count, removed=removed_count)
        time.sleep(delay)

    bar.close()
    print(f"\nSecond Pass complete: {kept_count} newly kept, {removed_count} removed.")
    print(f"Removed items were appended to: {removed_path}")


def rephrase_qna_pairs(output_file: str, config: dict, client: OpenAI) -> None:
    """Rephrase each approved Q&A pair into 10 variations using the LLM."""
    rephrasing_cfg = config.get("rephrasing", {})
    if not rephrasing_cfg.get("enabled", False):
        print("\nRephrasing is disabled by config.")
        return

    prompt_template = rephrasing_cfg.get("prompt", "")
    if not prompt_template:
        print("\nNo rephrasing prompt found in config.")
        return

    out_path = Path(output_file)
    if not out_path.exists():
        print("\nNo Q&A output file found for rephrasing.")
        return

    # Load all QnAs that passed second pass
    all_qnas = []
    with open(out_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    qna = json.loads(line)
                    if qna.get("second_pass_approved") is True and not qna.get("rephrased"):
                        all_qnas.append(qna)
                except Exception:
                    continue

    if not all_qnas:
        print("\nNo Q&A pairs found for rephrasing.")
        return

    print(f"\nStarting rephrasing for {len(all_qnas)} Q&A pairs...")
    bar = tqdm(all_qnas, desc="Rephrasing QnA", unit="qna", dynamic_ncols=True)
    for qna in bar:
        question = qna.get("question", "")
        answer = qna.get("answer", "")
        prompt = prompt_template.format(question=question, answer=answer)
        try:
            response = client.chat.completions.create(
                model=get_llm_settings(config)["model"],
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=5000,
            )
            raw = response.choices[0].message.content.strip()
            # Remove code block if present
            if raw.startswith("```"):
                parts = raw.split("```")
                raw = parts[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()
            variations = json.loads(raw)
            if not isinstance(variations, list) or len(variations) != 10:
                raise ValueError("Rephrasing LLM did not return a list of 10 Q&A pairs.")
        except Exception as e:
            tqdm.write(f"  Error rephrasing Q&A: {e}")
            continue

        # Save each variation immediately
        for idx, var in enumerate(variations, 1):
            rephrased_qna = dict(qna)  # Copy original fields
            rephrased_qna["question"] = f"rephrased: original {idx}: {var.get('question', '')}"
            rephrased_qna["answer"] = var.get("answer", "")
            rephrased_qna["rephrased"] = True
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rephrased_qna, ensure_ascii=False) + "\n")
        # Optionally, mark the original as rephrased to avoid duplicate rephrasing
        qna["rephrased"] = True
        save_qnas_safely(out_path, all_qnas)
        time.sleep(config.get("processing", {}).get("delay", 0))
    bar.close()
    print("\nRephrasing complete.")


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

    client = create_llm_client(config)

    # -------------------------------------------------------------
    # 0. First Pass: Q&A Generation
    # -------------------------------------------------------------
    if to_process:
        print(f"\nInitializing {provider} client for Pass 1...")
        delay = config["processing"]["delay"]
        total_questions = 0
        irrelevant_count = 0
        failed_count = 0
        entries_processed = 0
        start_time = time.time()

        print("\nStarting classification and Q&A generation...")

        bar = tqdm(to_process, desc="Processing Documents", unit="doc", dynamic_ncols=True)
        for entry in bar:
            hash_id = entry["hash"]
            filename = entry.get("filename", "")

            bar.set_postfix(
                hash=hash_id[:8],
                questions=total_questions,
                irrelevant=irrelevant_count,
                failed=failed_count,
            )

            content = load_md_content(raw_md_dir, filename)
            if not content:
                tqdm.write(f"  Skipping {hash_id[:8]}: empty or missing file")
                failed_count += 1
                entries_processed += 1
                continue

            chunks = chunk_content(content, max_chunk_size=10000)
            
            all_qa_pairs = []
            final_category = None
            is_failed = False
            is_irrelevant = True

            for chunk_idx, chunk in enumerate(chunks):
                chunk_length = len(chunk)
                num_min, num_max = calculate_question_count(chunk_length, config)
                result = process_document(client, config, chunk, num_min, num_max)

                if result is None:
                    tqdm.write(f"  FAILED {hash_id[:8]} (chunk {chunk_idx+1}/{len(chunks)}): LLM call unsuccessful")
                    is_failed = True
                    break
                elif result != "irrelevant":
                    is_irrelevant = False
                    all_qa_pairs.extend(result["qa_pairs"])
                    final_category = result["category"]

            if is_failed:
                failed_count += 1
            elif is_irrelevant:
                entry["category"] = IRRELEVANT_CATEGORY
                entry["llm_processed"] = True
                irrelevant_count += 1
                tqdm.write(f"  [{hash_id[:8]}] irrelevant")
            else:
                entry["category"] = final_category
                written = append_qa_to_file(
                    output_file, hash_id, all_qa_pairs, model_name, final_category
                )
                entry["llm_processed"] = True
                entry["questions_generated"] = True
                total_questions += written
                tqdm.write(
                    f"  [{hash_id[:8]}] {final_category} — {written} questions (from {len(chunks)} chunks)"
                )

            entries_processed += 1
            save_index(index_path, index_data)
            time.sleep(delay)

        duration = time.time() - start_time
        print(f"\nCompleted in {duration:.2f} seconds!")
        print(f"Total questions generated: {total_questions}")
        print(f"Irrelevant documents: {irrelevant_count}")
        print(f"Failed entries: {failed_count}")
        print(f"Output saved to: {output_file}")
    else:
        print("\nAll entries are already processed in Pass 1.")
    
    # -------------------------------------------------------------
    # 1. Post-processing simple filter (Blacklist / Whitelist)
    # -------------------------------------------------------------
    filter_generated_qa(output_file, config)

    # -------------------------------------------------------------
    # 2. Post-processing Second Pass LLM Filter
    # -------------------------------------------------------------
    if config.get("second_pass", {}).get("enabled", False):
        second_pass_llm_filter(output_file, config, client)

    # -------------------------------------------------------------
    # 3. Rephrasing Q&A pairs (optional)
    # -------------------------------------------------------------
    rephrasing_cfg = config.get("rephrasing", {})
    if rephrasing_cfg.get("enabled", False):
        rephrase_qna_pairs(output_file, config, client)


if __name__ == "__main__":
    main()