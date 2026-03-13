"""
Benchmark Module for UTN Project.

This script benchmarks LLM performance on Q&A pairs:
1. Samples questions from each category
2. Gets answers from a test LLM
3. Uses a judge LLM to score the answers
4. Saves detailed results to timestamped folders

Supports three providers:
- openai: OpenAI API
- local: LM Studio (OpenAI-compatible local server)
- huggingface: Local HuggingFace models (transformers)
"""

import json
import os
import random
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import yaml
from dotenv import load_dotenv
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm.auto import tqdm

# HuggingFace model cache (loaded once, reused)
_hf_model = None
_hf_tokenizer = None

# Load environment variables from .env file
load_dotenv()


def load_config(config_path: str = "4_benchmark/config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Replace environment variables in API keys
    for llm_key in ["answer_llm", "judge_llm"]:
        if config[llm_key]["openai"]["api_key"].startswith("${"):
            env_var = config[llm_key]["openai"]["api_key"][2:-1]
            config[llm_key]["openai"]["api_key"] = os.environ.get(env_var, "")

    return config


def load_huggingface_model(llm_config: dict):
    """Load HuggingFace model and tokenizer."""
    global _hf_model, _hf_tokenizer

    if _hf_model is not None:
        return _hf_model, _hf_tokenizer

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        raise ImportError(
            "transformers and torch are required for huggingface provider. "
            "Install with: pip install transformers torch"
        )

    hf_config = llm_config["huggingface"]
    model_name = hf_config["model"]
    model_dir = Path(hf_config.get("model_dir", "model"))
    local_path = model_dir / model_name.replace("/", "_")

    # Check if model exists locally, otherwise download
    if local_path.exists():
        print(f"  Loading model from local cache: {local_path}")
        load_path = str(local_path)
    else:
        print(f"  Downloading model from HuggingFace: {model_name}")
        print(f"  Will be cached to: {local_path}")
        load_path = model_name

    # Load tokenizer
    print("  Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        load_path,
        trust_remote_code=True,
    )

    # Load model
    print("  Loading model (this may take a while)...")
    device = hf_config.get("device", "auto")
    dtype_str = hf_config.get("dtype", "auto")

    # Map dtype string to torch dtype
    dtype_map = {
        "auto": "auto",
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map.get(dtype_str, "auto")

    model = AutoModelForCausalLM.from_pretrained(
        load_path,
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=True,
    )

    # Save model locally if downloaded from HuggingFace
    if not local_path.exists() and load_path == model_name:
        print(f"  Saving model to: {local_path}")
        local_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(local_path)
        tokenizer.save_pretrained(local_path)

    _hf_model = model
    _hf_tokenizer = tokenizer
    print("  Model loaded successfully!")
    return model, tokenizer


def generate_huggingface_response(llm_config: dict, prompt: str) -> str:
    """Generate response using HuggingFace model."""
    import torch

    model, tokenizer = load_huggingface_model(llm_config)
    hf_config = llm_config["huggingface"]

    # Build chat messages
    messages = [{"role": "user", "content": prompt}]

    # Apply chat template if available
    if hasattr(tokenizer, "apply_chat_template"):
        input_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        input_text = prompt

    # Tokenize
    inputs = tokenizer(input_text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=hf_config.get("max_tokens", 500),
            temperature=hf_config.get("temperature", 0.3),
            do_sample=hf_config.get("temperature", 0.3) > 0,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the generated part
    input_length = inputs["input_ids"].shape[1]
    generated_tokens = outputs[0][input_length:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    return response.strip()


def create_llm_client(llm_config: dict) -> OpenAI | None:
    """Create OpenAI client based on provider setting.

    Returns None for huggingface provider (uses different inference path).
    """
    provider = llm_config["provider"]
    timeout = float(os.getenv("OPENAI_TIMEOUT", "60"))
    max_retries = int(os.getenv("OPENAI_MAX_RETRIES", "2"))

    if provider == "openai":
        api_key = llm_config["openai"]["api_key"]
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        return OpenAI(api_key=api_key, timeout=timeout, max_retries=max_retries)
    elif provider == "local":
        return OpenAI(
            base_url=llm_config["local"]["base_url"],
            api_key="lm-studio",
            timeout=timeout,
        )
    elif provider == "huggingface":
        # Pre-load the model
        load_huggingface_model(llm_config)
        return None  # HuggingFace uses direct inference, not OpenAI client
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")


def get_llm_settings(llm_config: dict) -> dict:
    """Get model settings based on provider."""
    provider = llm_config["provider"]
    if provider == "openai":
        return llm_config["openai"]
    elif provider == "huggingface":
        return llm_config["huggingface"]
    return llm_config["local"]


def load_qna_data(qna_file: str) -> list[dict]:
    """Load Q&A pairs from JSONL file."""
    entries = []
    with open(qna_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def sample_questions(
    qna_data: list[dict], questions_per_category: int, categories: list[str]
) -> list[dict]:
    """Sample questions randomly from each category with unique document constraint.
    
    Ensures that no two questions in the entire benchmark come from the same
    document (same hash).
    """
    # Group by category
    by_category = defaultdict(list)
    for entry in qna_data:
        cat = entry.get("category", "Unknown")
        by_category[cat].append(entry)

    # Filter categories if specified
    if categories:
        by_category = {k: v for k, v in by_category.items() if k in categories}

    sampled = []
    used_hashes = set()

    # Shuffle categories to ensure fairness if multiple categories share hashes
    category_list = list(by_category.keys())
    random.shuffle(category_list)

    for cat in category_list:
        entries = by_category[cat]
        # Shuffle entries within category for random sampling
        random.shuffle(entries)
        
        cat_sampled_count = 0
        for entry in entries:
            if cat_sampled_count >= questions_per_category:
                break
                
            h = entry.get("hash")
            if h and h not in used_hashes:
                sampled.append(entry)
                used_hashes.add(h)
                cat_sampled_count += 1
            elif not h:
                # If no hash (shouldn't happen with our data), still sample it
                sampled.append(entry)
                cat_sampled_count += 1

        print(f"  {cat}: {cat_sampled_count} questions sampled (requested {questions_per_category})")

    random.shuffle(sampled)
    return sampled


def save_benchmark_file(benchmark_file: str, questions: list[dict]) -> None:
    """Save sampled questions to benchmark JSONL file."""
    Path(benchmark_file).parent.mkdir(parents=True, exist_ok=True)
    with open(benchmark_file, "w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")


def remove_questions_from_source(qna_file: str, questions_to_remove: list[dict]) -> None:
    """Remove benchmarked questions from the source JSONL file.
    
    Uses both hash and question text to ensure we only remove the specific
    sampled questions, not all questions from the same document.
    """
    # Load all existing data
    all_data = load_qna_data(qna_file)
    
    # Create a set of unique identifiers (hash + question text) for removal
    to_remove = {
        (q.get("hash", ""), q.get("question", "")) 
        for q in questions_to_remove
    }
    
    # Filter out questions that match the identifiers
    remaining_data = [
        q for q in all_data 
        if (q.get("hash", ""), q.get("question", "")) not in to_remove
    ]
    
    removed_count = len(all_data) - len(remaining_data)
    
    # Save the remaining data back to the same file
    if removed_count > 0:
        with open(qna_file, "w", encoding="utf-8") as f:
            for q in remaining_data:
                f.write(json.dumps(q, ensure_ascii=False) + "\n")
        print(f"  Removed {removed_count} specific questions from source data: {qna_file}")
    else:
        print(f"  No questions were removed (already removed or identifiers missing).")


@retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(5))
def get_answer_with_retry(client: OpenAI, settings: dict, prompt: str) -> str:
    """Get answer from LLM with automatic retry."""
    response = client.chat.completions.create(
        model=settings["model"],
        messages=[
            {"role": "user", "content": prompt},
        ],
        temperature=settings["temperature"],
        max_tokens=settings["max_tokens"],
    )
    return response.choices[0].message.content.strip()


def get_answer(
    client: OpenAI | None, settings: dict, config: dict, question: str, llm_config: dict
) -> str | None:
    """Get answer from the answer LLM."""
    prompt = config["answer_prompt"].format(question=question)

    try:
        # Use HuggingFace inference if provider is huggingface
        if llm_config["provider"] == "huggingface":
            return generate_huggingface_response(llm_config, prompt)
        return get_answer_with_retry(client, settings, prompt)
    except Exception as e:
        actual_error = e.__cause__ if e.__cause__ else e
        print(f"\n  Answer Error: {type(actual_error).__name__}: {actual_error}")
        return None


@retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(5))
def judge_answer_with_retry(client: OpenAI, settings: dict, prompt: str) -> dict:
    """Judge answer with automatic retry. Includes logic to fix truncated JSON."""
    response = client.chat.completions.create(
        model=settings["model"],
        messages=[
            {
                "role": "system",
                "content": "You are a fair evaluator. Always respond with a single, complete, valid JSON object. Do not include any text before or after the JSON.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=settings.get("temperature", 0.0),
        max_tokens=512, # Increased max_tokens for reasoning
    )

    content = response.choices[0].message.content.strip()

    # Improved JSON extraction
    try:
        # 1. Try simple json.loads first
        result = json.loads(content)
    except json.JSONDecodeError as base_e:
        # 2. Try to find any curly braces
        import re
        json_match = re.search(r"(\{.*\})", content, re.DOTALL)
        if json_match:
            try:
                result = json.loads(json_match.group(1))
            except json.JSONDecodeError as inner_e:
                # 3. Handle truncation
                if "Unterminated string" in str(inner_e) or "Expecting value" in str(inner_e):
                    fixed_content = json_match.group(1).strip()
                    if fixed_content.count('"') % 2 != 0:
                        fixed_content += '"'
                    if not fixed_content.endswith("}"):
                        fixed_content += "}"
                    try:
                        result = json.loads(fixed_content)
                    except:
                        raise inner_e
                else:
                    raise inner_e
        else:
            raise base_e

    # Validate result is a dictionary
    if not isinstance(result, dict):
        raise ValueError(f"Expected JSON object, got {type(result)}")
    
    # Extract and validate score
    score = result.get("score", 0)
    if not isinstance(score, (int, float)):
        try:
            score = float(score)
        except (ValueError, TypeError):
            score = 0.0

    # Map to valid [0, 0.5, 1] scale
    if score not in [0, 0.5, 1, 0.0, 1.0]:
        if score < 0.25:
            result["score"] = 0.0
        elif score < 0.75:
            result["score"] = 0.5
        else:
            result["score"] = 1.0
    else:
        result["score"] = float(score)

    return result


def judge_answer(
    client: OpenAI,
    settings: dict,
    config: dict,
    question: str,
    expected: str,
    ai_answer: str,
) -> dict | None:
    """Get judgment from the judge LLM."""
    prompt = config["judge_prompt"].format(
        question=question,
        expected_answer=expected,
        ai_answer=ai_answer,
    )

    try:
        return judge_answer_with_retry(client, settings, prompt)
    except Exception as e:
        actual_error = e.__cause__ if e.__cause__ else e
        print(f"\n  Judge Error: {type(actual_error).__name__}: {actual_error}")
        return None


def create_results_dir(base_dir: str) -> Path:
    """Create timestamped results directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = Path(base_dir) / timestamp
    results_path.mkdir(parents=True, exist_ok=True)
    return results_path


def save_summary(
    results_dir: Path,
    answer_config: dict,
    judge_config: dict,
    results: list[dict],
    duration: float,
) -> None:
    """Save summary info to file."""
    answer_settings = get_llm_settings(answer_config)
    judge_settings = get_llm_settings(judge_config)

    # Calculate stats
    scores = [r["score"] for r in results if r["score"] is not None]
    total_questions = len(results)
    answered = len(scores)
    failed = total_questions - answered

    avg_score = sum(scores) / len(scores) if scores else 0
    full_correct = sum(1 for s in scores if s == 1.0)
    partial = sum(1 for s in scores if s == 0.5)
    wrong = sum(1 for s in scores if s == 0.0)

    # Stats by category
    by_category = defaultdict(list)
    for r in results:
        if r["score"] is not None:
            by_category[r["category"]].append(r["score"])

    summary = {
        "run_timestamp": datetime.now().isoformat(),
        "duration_seconds": round(duration, 2),
        "answer_model": {
            "provider": answer_config["provider"],
            "model": answer_settings["model"],
            "temperature": answer_settings["temperature"],
        },
        "judge_model": {
            "provider": judge_config["provider"],
            "model": judge_settings["model"],
            "temperature": judge_settings["temperature"],
        },
        "stats": {
            "total_questions": total_questions,
            "answered": answered,
            "failed": failed,
            "average_score": round(avg_score, 4),
            "full_correct": full_correct,
            "partial_correct": partial,
            "wrong": wrong,
        },
        "stats_by_category": {
            cat: {
                "count": len(scores_list),
                "average": (
                    round(sum(scores_list) / len(scores_list), 4) if scores_list else 0
                ),
                "full_correct": sum(1 for s in scores_list if s == 1.0),
                "partial": sum(1 for s in scores_list if s == 0.5),
                "wrong": sum(1 for s in scores_list if s == 0.0),
            }
            for cat, scores_list in by_category.items()
        },
    }

    summary_file = results_dir / "summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Also print summary
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(
        f"Answer Model: {answer_settings['model']} (temp={answer_settings['temperature']})"
    )
    print(f"Judge Model: {judge_settings['model']}")
    print(f"Duration: {duration:.2f}s")
    print("-" * 60)
    print(f"Total Questions: {total_questions}")
    print(f"Average Score: {avg_score:.2%}")
    print(f"Full Correct (1.0): {full_correct}")
    print(f"Partial (0.5): {partial}")
    print(f"Wrong (0.0): {wrong}")
    print(f"Failed: {failed}")
    print("-" * 60)
    print("By Category:")
    print(
        f"  {'Category':<15} {'Avg':>8} {'Full':>6} {'Part':>6} {'Wrong':>6} {'Count':>6}"
    )
    print(f"  {'-'*15} {'-'*8} {'-'*6} {'-'*6} {'-'*6} {'-'*6}")
    for cat, data in summary["stats_by_category"].items():
        print(
            f"  {cat:<15} {data['average']:>7.1%} {data['full_correct']:>6} {data['partial']:>6} {data['wrong']:>6} {data['count']:>6}"
        )


def save_detailed_results(results_dir: Path, results: list[dict]) -> None:
    """Save detailed results to JSONL file."""
    detailed_file = results_dir / "detailed_results.jsonl"
    with open(detailed_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    """Main function to run the benchmark."""
    print("=" * 60)
    print("LLM Benchmark")
    print("=" * 60)

    # Load configuration
    config = load_config()
    answer_config = config["answer_llm"]
    judge_config = config["judge_llm"]

    answer_settings = get_llm_settings(answer_config)
    judge_settings = get_llm_settings(judge_config)

    print(f"\nAnswer Model: {answer_config['provider']} / {answer_settings['model']}")
    print(f"Judge Model: {judge_config['provider']} / {judge_settings['model']}")

    # Load Q&A data
    qna_file = config["data"]["qna_file"]
    questions_per_cat = config["benchmark"]["questions_per_category"]

    # Construct benchmark filename based on questions_per_category
    benchmark_dir = Path(config["data"]["benchmark_file"]).parent
    benchmark_file = benchmark_dir / f"benchmark_{questions_per_cat}.jsonl"

    # Check if benchmark file already exists
    if benchmark_file.exists():
        print(f"\nReusing existing benchmark file: {benchmark_file}")
        print(f"(To regenerate, delete {benchmark_file})")
        benchmark_questions = load_qna_data(str(benchmark_file))
        print(f"Loaded {len(benchmark_questions)} questions from existing benchmark")
        
        # Ensure benchmark questions are removed from source (in case the run was interrupted)
        remove_questions_from_source(qna_file, benchmark_questions)
    else:
        print(f"\nLoading Q&A from: {qna_file}")
        qna_data = load_qna_data(qna_file)
        print(f"Total Q&A pairs: {len(qna_data)}")

        # Sample questions
        categories = config["benchmark"]["categories"]
        print(f"\nSampling {questions_per_cat} questions per category...")
        benchmark_questions = sample_questions(qna_data, questions_per_cat, categories)
        print(f"Total benchmark questions: {len(benchmark_questions)}")

        # Save benchmark file
        save_benchmark_file(str(benchmark_file), benchmark_questions)
        print(f"Saved to: {benchmark_file}")
        
        # Remove sampled benchmark questions from the source JSONL data file
        remove_questions_from_source(qna_file, benchmark_questions)

    # Create results directory
    results_dir = create_results_dir(config["data"]["results_dir"])
    print(f"\nResults will be saved to: {results_dir}")

    # Create LLM clients
    print("\nInitializing LLM clients...")
    answer_client = create_llm_client(answer_config)
    judge_client = create_llm_client(judge_config)

    # Run benchmark
    delay = config["processing"]["delay"]
    results = []
    start_time = time.time()

    print("\nRunning benchmark...")

    for entry in tqdm(benchmark_questions, desc="Benchmarking"):
        question = entry["question"]
        expected_answer = entry["answer"]
        category = entry.get("category", "Unknown")
        hash_id = entry.get("hash", "")

        # Get answer from test LLM
        ai_answer = get_answer(
            answer_client, answer_settings, config, question, answer_config
        )

        if ai_answer is None:
            results.append(
                {
                    "hash": hash_id,
                    "category": category,
                    "question": question,
                    "expected_answer": expected_answer,
                    "ai_answer": None,
                    "score": None,
                    "reason": "Failed to get answer",
                }
            )
            continue

        # Judge the answer
        judgment = judge_answer(
            judge_client, judge_settings, config, question, expected_answer, ai_answer
        )

        if judgment is None:
            results.append(
                {
                    "hash": hash_id,
                    "category": category,
                    "question": question,
                    "expected_answer": expected_answer,
                    "ai_answer": ai_answer,
                    "score": None,
                    "reason": "Failed to judge",
                }
            )
            continue

        results.append(
            {
                "hash": hash_id,
                "category": category,
                "question": question,
                "expected_answer": expected_answer,
                "ai_answer": ai_answer,
                "score": judgment["score"],
                "reason": judgment.get("reason", ""),
            }
        )

        if delay > 0:
            time.sleep(delay)

    duration = time.time() - start_time

    # Save results
    save_detailed_results(results_dir, results)
    save_summary(results_dir, answer_config, judge_config, results, duration)

    print(f"\nResults saved to: {results_dir}")


if __name__ == "__main__":
    main()
