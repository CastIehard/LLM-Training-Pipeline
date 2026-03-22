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
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import torch
import yaml
from dotenv import load_dotenv
from openai import OpenAI
from peft import PeftModel
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# HuggingFace model cache keyed by model-loading settings
_hf_cache = {}

# Load environment variables from .env file
load_dotenv()


def load_config(config_path: str = "4_benchmark/config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Replace environment variables in API keys
    for llm_key in ["answer_llm", "judge_llm"]:
        if config[llm_key]["provider"] == "openai":
            api_key = config[llm_key]["openai"]["api_key"]
            if api_key.startswith("${"):
                env_var = api_key[2:-1]
                config[llm_key]["openai"]["api_key"] = os.environ.get(env_var, "")

    return config


def _chunked(items: list, chunk_size: int):
    """Yield fixed-size chunks from a list."""
    chunk_size = max(1, int(chunk_size))
    for i in range(0, len(items), chunk_size):
        yield items[i:i + chunk_size]


def _strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks if present."""
    if not text:
        return ""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _normalize_judge_result(result: dict) -> dict:
    """Validate and normalize judge JSON."""
    if not isinstance(result, dict):
        raise ValueError(f"Expected JSON object, got {type(result)}")

    score = result.get("score", 0)
    if not isinstance(score, (int, float)):
        try:
            score = float(score)
        except (ValueError, TypeError):
            score = 0.0

    if score not in [0, 0.5, 1, 0.0, 1.0]:
        if score < 0.25:
            normalized_score = 0.0
        elif score < 0.75:
            normalized_score = 0.5
        else:
            normalized_score = 1.0
    else:
        normalized_score = float(score)

    reason = str(result.get("reason", "")).strip()

    return {
        "score": normalized_score,
        "reason": reason,
        "reasoning": str(result.get("reasoning", "")).strip(),
    }


def _parse_judge_content(content: str) -> dict:
    """Extract and parse a JSON object from model output."""
    cleaned = _strip_think_tags(content).strip()

    try:
        parsed = json.loads(cleaned)
        return _normalize_judge_result(parsed)
    except json.JSONDecodeError as base_e:
        match = re.search(r"(\{.*\})", cleaned, re.DOTALL)
        if not match:
            raise ValueError(f"No JSON object found in judge response: {cleaned[:300]}") from base_e

        candidate = match.group(1)

        try:
            parsed = json.loads(candidate)
            return _normalize_judge_result(parsed)
        except json.JSONDecodeError as inner_e:
            if "Unterminated string" in str(inner_e) or "Expecting value" in str(inner_e):
                fixed = candidate.strip()
                if fixed.count('"') % 2 != 0:
                    fixed += '"'
                if not fixed.endswith("}"):
                    fixed += "}"
                parsed = json.loads(fixed)
                return _normalize_judge_result(parsed)
            raise


def _get_hf_cache_key(llm_config: dict) -> str:
    """Create a stable cache key for HuggingFace model loading."""
    hf_config = llm_config["huggingface"]
    model_name = hf_config["model"]
    model_dir = str(Path(hf_config.get("model_dir", "model")))
    adapter_dir = hf_config.get("adapter_dir")
    device = hf_config.get("device", "auto")
    dtype = hf_config.get("dtype", "auto")
    trust_remote_code = hf_config.get("trust_remote_code", True)

    return json.dumps(
        {
            "model": model_name,
            "model_dir": model_dir,
            "adapter_dir": adapter_dir,
            "device": device,
            "dtype": dtype,
            "trust_remote_code": trust_remote_code,
        },
        sort_keys=True,
    )


def load_huggingface_model(llm_config: dict):
    """Load HuggingFace model and tokenizer, cached by model config."""
    global _hf_cache

    cache_key = _get_hf_cache_key(llm_config)
    if cache_key in _hf_cache:
        return _hf_cache[cache_key]["model"], _hf_cache[cache_key]["tokenizer"]

    hf_config = llm_config["huggingface"]
    model_name = hf_config["model"]
    model_dir = Path(hf_config.get("model_dir", "model"))
    adapter_dir_value = hf_config.get("adapter_dir")
    adapter_path = Path(adapter_dir_value).expanduser().resolve() if adapter_dir_value else None
    local_path = model_dir / model_name.replace("/", "_")
    trust_remote_code = hf_config.get("trust_remote_code", True)
    is_adapter_run = adapter_path is not None

    if local_path.exists():
        print(f"  Loading model from local cache: {local_path}")
        load_path = str(local_path)
        should_save_local = False
    else:
        print(f"  Downloading model from HuggingFace: {model_name}")
        print(f"  Will be cached to: {local_path}")
        load_path = model_name
        should_save_local = not is_adapter_run

    print("  Loading tokenizer...")
    tokenizer_load_path = str(adapter_path) if is_adapter_run and adapter_path.exists() else load_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_load_path,
        padding_side='left',
        trust_remote_code=trust_remote_code,
    )

    print("  Loading model (this may take a while)...")
    device = hf_config.get("device", "auto")
    dtype_str = hf_config.get("dtype", "auto")

    dtype_map = {
        "auto": "auto",
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    torch_dtype = dtype_map.get(dtype_str, "auto")

    model = AutoModelForCausalLM.from_pretrained(
        load_path,
        torch_dtype=torch_dtype,
        device_map=device,
        trust_remote_code=trust_remote_code,
    )

    if is_adapter_run:
        if not adapter_path.exists():
            raise FileNotFoundError(f"Adapter directory does not exist: {adapter_path}")
        print(f"  Loading PEFT adapter from: {adapter_path}")
        model = PeftModel.from_pretrained(model, str(adapter_path))

    model.eval()

    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if should_save_local:
        print(f"  Saving model to: {local_path}")
        local_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(local_path)
        tokenizer.save_pretrained(local_path)

    if hf_config.get("compile", False):
        try:
            model = torch.compile(model)
        except Exception as e:
            print(f"  Warning: torch.compile failed, continuing without compile: {e}")

    _hf_cache[cache_key] = {
        "model": model,
        "tokenizer": tokenizer,
    }

    print("  Model loaded successfully!")
    return model, tokenizer


def _prepare_hf_inputs(tokenizer, prompts: list[str]) -> dict:
    """Tokenize a batch of prompts, using chat template when available."""
    if hasattr(tokenizer, "apply_chat_template"):
        rendered_prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for prompt in prompts
        ]
        return tokenizer(rendered_prompts, return_tensors="pt", padding=True)

    return tokenizer(prompts, return_tensors="pt", padding=True)


def _generate_huggingface_batch(
        llm_config: dict,
        prompts: list[str],
        *,
        max_new_tokens: int,
        temperature: float,
) -> list[str]:
    """Generate a batch of responses using a HuggingFace model."""
    model, tokenizer = load_huggingface_model(llm_config)

    inputs = _prepare_hf_inputs(tokenizer, prompts)

    model_device = getattr(model, "device", None)
    if model_device is None:
        model_device = next(model.parameters()).device

    inputs = {k: v.to(model_device) for k, v in inputs.items()}
    input_lengths = inputs["attention_mask"].sum(dim=1).tolist()

    generate_kwargs = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "do_sample": temperature > 0,
        "pad_token_id": tokenizer.pad_token_id,
        "use_cache": True,
    }

    with torch.inference_mode():
        outputs = model.generate(**inputs, **generate_kwargs)

    responses = []
    for i, input_length in enumerate(input_lengths):
        generated_tokens = outputs[i, int(input_length):]
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        responses.append(response)

    return responses


def generate_huggingface_responses(
        llm_config: dict,
        prompts: list[str],
        *,
        max_new_tokens: int,
        temperature: float,
        batch_size: int,
        desc: str,
        delay: float = 0.0,
) -> list[str | None]:
    """Generate responses in HuggingFace batches with per-item failure isolation."""
    if not prompts:
        return []

    results = [None] * len(prompts)
    indexed_prompts = list(enumerate(prompts))

    with tqdm(total=len(prompts), desc=desc) as pbar:
        for batch in _chunked(indexed_prompts, batch_size):
            batch_indices = [idx for idx, _ in batch]
            batch_prompts = [prompt for _, prompt in batch]

            try:
                batch_responses = _generate_huggingface_batch(
                    llm_config,
                    batch_prompts,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                )

                for idx, response in zip(batch_indices, batch_responses):
                    results[idx] = response

            except Exception as batch_error:
                print(f"\n  {desc} batch error: {type(batch_error).__name__}: {batch_error}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # Fallback to single-item generation so one bad item does not lose the whole batch
                for idx, prompt in batch:
                    try:
                        single_response = _generate_huggingface_batch(
                            llm_config,
                            [prompt],
                            max_new_tokens=max_new_tokens,
                            temperature=temperature,
                        )[0]
                        results[idx] = single_response
                    except Exception as item_error:
                        actual_error = item_error.__cause__ if item_error.__cause__ else item_error
                        print(f"\n  {desc} item error: {type(actual_error).__name__}: {actual_error}")
                        results[idx] = None

            pbar.update(len(batch))

            if delay > 0:
                time.sleep(delay)

    return results


def create_llm_client(llm_config: dict) -> OpenAI | None:
    """Create OpenAI client based on provider setting.

    Returns None for huggingface provider (uses direct inference path).
    """
    provider = llm_config["provider"]
    timeout = float(os.getenv("OPENAI_TIMEOUT", "60"))
    max_retries = int(os.getenv("OPENAI_MAX_RETRIES", "2"))

    if provider == "openai":
        api_key = llm_config["openai"]["api_key"]
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        return OpenAI(api_key=api_key, timeout=timeout, max_retries=max_retries)

    if provider == "local":
        return OpenAI(
            base_url=llm_config["local"]["base_url"],
            api_key="lm-studio",
            timeout=timeout,
        )

    if provider == "huggingface":
        # Pre-load the model
        load_huggingface_model(llm_config)
        return None

    raise ValueError(f"Unknown LLM provider: {provider}")


def get_llm_settings(llm_config: dict) -> dict:
    """Get model settings based on provider."""
    provider = llm_config["provider"]
    if provider == "openai":
        return llm_config["openai"]
    if provider == "huggingface":
        return llm_config["huggingface"]
    return llm_config["local"]


def load_qna_data(qna_file: str) -> list[dict]:
    """Load Q&A pairs from JSONL file."""
    with open(qna_file, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def sample_questions(qna_data: list[dict], questions_per_category: int, categories: list[str]) -> list[dict]:
    """Sample questions randomly from each category with unique document constraint.

    Ensures that no two questions in the entire benchmark come from the same
    document (same hash).
    """
    by_category = defaultdict(list)
    for entry in qna_data:
        cat = entry.get("category", "Unknown")
        by_category[cat].append(entry)

    if categories:
        by_category = {k: v for k, v in by_category.items() if k in categories}

    sampled = []
    used_hashes = set()

    category_list = list(by_category.keys())
    random.shuffle(category_list)

    for cat in category_list:
        entries = by_category[cat]
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
    all_data = load_qna_data(qna_file)

    to_remove = {
        (q.get("hash", ""), q.get("question", ""))
        for q in questions_to_remove
    }

    remaining_data = [
        q for q in all_data
        if (q.get("hash", ""), q.get("question", "")) not in to_remove
    ]

    removed_count = len(all_data) - len(remaining_data)

    if removed_count > 0:
        with open(qna_file, "w", encoding="utf-8") as f:
            for q in remaining_data:
                f.write(json.dumps(q, ensure_ascii=False) + "\n")
        print(f"  Removed {removed_count} specific questions from source data: {qna_file}")
    else:
        print(f"  No questions were removed (already removed or identifiers missing).")


@retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(5))
def get_answer_with_retry(client: OpenAI, settings: dict, prompt: str) -> str:
    """Get answer from OpenAI-compatible LLM with automatic retry."""
    response = client.chat.completions.create(
        model=settings["model"],
        messages=[{"role": "user", "content": prompt}],
        temperature=settings["temperature"],
        max_tokens=settings["max_tokens"],
    )
    return response.choices[0].message.content.strip()


@retry(wait=wait_exponential(multiplier=0.25, min=0.25, max=2), stop=stop_after_attempt(2))
def judge_answer_with_retry(client: OpenAI, settings: dict, prompt: str, provider: str) -> dict:
    """Judge answer with automatic retry using prompt-enforced JSON output."""
    request_kwargs = {
        "model": settings["model"],
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a fair evaluator. "
                    "Always respond with a single, complete, valid JSON object. "
                    "Do not include any text before or after the JSON."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": settings.get("temperature", 0.0),
        "max_tokens": settings.get("max_tokens", 512),
    }

    if provider == "local":
        request_kwargs["extra_body"] = {
            "chat_template_kwargs": {"enable_thinking": False},
        }

    response = client.chat.completions.create(**request_kwargs)
    content = response.choices[0].message.content.strip()
    return _parse_judge_content(content)


def _build_answer_prompt(config: dict, question: str) -> str:
    """Build prompt for answer generation."""
    return config["answer_prompt"].format(question=question)


def _build_judge_prompt(config: dict, question: str, expected_answer: str, ai_answer: str, ) -> str:
    """Build prompt for judging."""
    eval_ai_answer = _strip_think_tags(ai_answer)

    prompt = config["judge_prompt"].format(
        question=question,
        expected_answer=expected_answer,
        ai_answer=eval_ai_answer,
    )

    return prompt


def _run_single_answer(
        client: OpenAI | None,
        settings: dict,
        llm_config: dict,
        prompt: str,
) -> str | None:
    """Run one answer request for any provider."""
    provider = llm_config["provider"]

    try:
        if provider == "huggingface":
            return _generate_huggingface_batch(
                llm_config,
                [prompt],
                max_new_tokens=settings["max_tokens"],
                temperature=settings["temperature"],
            )[0]

        return get_answer_with_retry(client, settings, prompt)

    except Exception as e:
        actual_error = e.__cause__ if e.__cause__ else e
        print(f"\n  Answer Error: {type(actual_error).__name__}: {actual_error}")
        return None


def _run_single_judgment(
        client: OpenAI | None,
        settings: dict,
        llm_config: dict,
        prompt: str,
) -> dict | None:
    """Run one judge request for any provider."""
    provider = llm_config["provider"]

    try:
        if provider == "huggingface":
            raw = _generate_huggingface_batch(
                llm_config,
                [prompt],
                max_new_tokens=settings.get("max_tokens", 512),
                temperature=0.0,
            )[0]
            return _parse_judge_content(raw)

        return judge_answer_with_retry(client, settings, prompt, provider)

    except Exception as e:
        actual_error = e.__cause__ if e.__cause__ else e
        print(f"\n  Judge Error: {type(actual_error).__name__}: {actual_error}")
        return None


def _get_phase_batch_size(config: dict, llm_config: dict, phase: str) -> int:
    """Resolve batch size for answers or judgments."""
    processing = config.get("processing", {})

    explicit = processing.get(f"{phase}_batch_size")
    if explicit is not None:
        return max(1, int(explicit))

    if llm_config["provider"] == "huggingface":
        return max(1, int(llm_config["huggingface"].get("batch_size", 8 if phase == "answer" else 4)))

    return 1


def _get_phase_max_workers(config: dict, phase: str) -> int:
    """Resolve max workers for OpenAI-compatible providers."""
    processing = config.get("processing", {})
    explicit = processing.get(f"{phase}_max_workers")
    if explicit is not None:
        return max(1, int(explicit))
    return 1


def generate_answers_batch(
        client: OpenAI | None,
        settings: dict,
        config: dict,
        benchmark_questions: list[dict],
        llm_config: dict,
        delay: float = 0.0,
) -> list[str | None]:
    """Generate all answers first, preserving original order."""
    prompts = [_build_answer_prompt(config, entry["question"]) for entry in benchmark_questions]
    provider = llm_config["provider"]
    batch_size = _get_phase_batch_size(config, llm_config, "answer")
    max_workers = _get_phase_max_workers(config, "answer")

    if provider == "huggingface":
        return generate_huggingface_responses(
            llm_config,
            prompts,
            max_new_tokens=settings["max_tokens"],
            temperature=settings["temperature"],
            batch_size=batch_size,
            desc="Generating answers",
            delay=delay,
        )

    results = [None] * len(prompts)
    indexed_prompts = list(enumerate(prompts))

    with tqdm(total=len(prompts), desc="Generating answers") as pbar:
        for batch in _chunked(indexed_prompts, batch_size):
            if max_workers > 1:
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {
                        executor.submit(_run_single_answer, client, settings, llm_config, prompt): idx
                        for idx, prompt in batch
                    }
                    for future in as_completed(futures):
                        idx = futures[future]
                        results[idx] = future.result()
                        pbar.update(1)
            else:
                for idx, prompt in batch:
                    results[idx] = _run_single_answer(client, settings, llm_config, prompt)
                    pbar.update(1)

            if delay > 0:
                time.sleep(delay)

    return results


def generate_judgments_batch(
        client: OpenAI | None,
        settings: dict,
        config: dict,
        benchmark_questions: list[dict],
        ai_answers: list[str | None],
        llm_config: dict,
        delay: float = 0.0,
) -> list[dict | None]:
    """Judge all successful answers after answer generation is complete."""
    judge_inputs = []
    for idx, (entry, ai_answer) in enumerate(zip(benchmark_questions, ai_answers)):
        if ai_answer is None:
            continue

        judge_inputs.append(
            (
                idx,
                _build_judge_prompt(
                    config,
                    question=entry["question"],
                    expected_answer=entry["answer"],
                    ai_answer=ai_answer,
                ),
            )
        )

    if not judge_inputs:
        return [None] * len(ai_answers)

    provider = llm_config["provider"]
    batch_size = _get_phase_batch_size(config, llm_config, "judge")
    max_workers = _get_phase_max_workers(config, "judge")

    results = [None] * len(ai_answers)

    if provider == "huggingface":
        prompts = [prompt for _, prompt in judge_inputs]
        hf_results = generate_huggingface_responses(
            llm_config,
            prompts,
            max_new_tokens=settings.get("max_tokens", 512),
            temperature=0.0,
            batch_size=batch_size,
            desc="Judging answers",
            delay=delay,
        )

        for (original_idx, _), raw_response in zip(judge_inputs, hf_results):
            if raw_response is None:
                results[original_idx] = None
                continue

            try:
                results[original_idx] = _parse_judge_content(raw_response)
            except Exception as e:
                actual_error = e.__cause__ if e.__cause__ else e
                print(f"\n  Judge Error: {type(actual_error).__name__}: {actual_error}")
                results[original_idx] = None

        return results

    with tqdm(total=len(judge_inputs), desc="Judging answers") as pbar:
        for batch in _chunked(judge_inputs, batch_size):
            if max_workers > 1:
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {
                        executor.submit(_run_single_judgment, client, settings, llm_config, prompt): idx
                        for idx, prompt in batch
                    }
                    for future in as_completed(futures):
                        idx = futures[future]
                        results[idx] = future.result()
                        pbar.update(1)
            else:
                for idx, prompt in batch:
                    results[idx] = _run_single_judgment(client, settings, llm_config, prompt)
                    pbar.update(1)

            if delay > 0:
                time.sleep(delay)

    return results


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

    scores = [r["score"] for r in results if r["score"] is not None]
    total_questions = len(results)
    answered = len(scores)
    failed = total_questions - answered

    avg_score = sum(scores) / len(scores) if scores else 0
    full_correct = sum(1 for s in scores if s == 1.0)
    partial = sum(1 for s in scores if s == 0.5)
    wrong = sum(1 for s in scores if s == 0.0)

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
            "adapter_dir": answer_settings.get("adapter_dir"),
            "temperature": answer_settings["temperature"],
        },
        "judge_model": {
            "provider": judge_config["provider"],
            "model": judge_settings["model"],
            "adapter_dir": judge_settings.get("adapter_dir"),
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

    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Answer Model: {answer_settings['model']} (temp={answer_settings['temperature']})")
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
    print(f"  {'Category':<15} {'Avg':>8} {'Full':>6} {'Part':>6} {'Wrong':>6} {'Count':>6}")
    print(f"  {'-' * 15} {'-' * 8} {'-' * 6} {'-' * 6} {'-' * 6} {'-' * 6}")
    for cat, data in summary["stats_by_category"].items():
        print(
            f"  {cat:<15} {data['average']:>7.1%} "
            f"{data['full_correct']:>6} {data['partial']:>6} {data['wrong']:>6} {data['count']:>6}"
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

    config = load_config()
    answer_config = config["answer_llm"]
    judge_config = config["judge_llm"]

    answer_settings = get_llm_settings(answer_config)
    judge_settings = get_llm_settings(judge_config)

    print(f"\nAnswer Model: {answer_config['provider']} / {answer_settings['model']}")
    print(f"Judge Model: {judge_config['provider']} / {judge_settings['model']}")

    qna_file = config["data"]["qna_file"]
    benchmark_file = Path(config["data"]["benchmark_file"])

    if benchmark_file.exists():
        print(f"\nReusing existing benchmark file: {benchmark_file}")
        print(f"(To regenerate, delete {benchmark_file})")
        benchmark_questions = load_qna_data(str(benchmark_file))
        print(f"Loaded {len(benchmark_questions)} questions from existing benchmark")

        remove_questions_from_source(qna_file, benchmark_questions)
    else:
        print(f"\nLoading Q&A from: {qna_file}")
        qna_data = load_qna_data(qna_file)
        print(f"Total Q&A pairs: {len(qna_data)}")

        categories = config["benchmark"]["categories"]
        questions_per_cat = config["benchmark"]["questions_per_category"]
        print(f"\nSampling {questions_per_cat} questions per category...")
        benchmark_questions = sample_questions(qna_data, questions_per_cat, categories)
        print(f"Total benchmark questions: {len(benchmark_questions)}")

        save_benchmark_file(str(benchmark_file), benchmark_questions)
        print(f"Saved to: {benchmark_file}")

        remove_questions_from_source(qna_file, benchmark_questions)

    results_dir = create_results_dir(config["data"]["results_dir"])
    print(f"\nResults will be saved to: {results_dir}")

    print("\nInitializing LLM clients...")
    answer_client = create_llm_client(answer_config)
    judge_client = create_llm_client(judge_config)

    delay = float(config.get("processing", {}).get("delay", 0.0))
    answer_phase_delay = delay / 2 if delay > 0 else 0.0
    judge_phase_delay = delay / 2 if delay > 0 else 0.0

    start_time = time.time()

    print("\nGenerating all answers first...")
    ai_answers = generate_answers_batch(
        answer_client,
        answer_settings,
        config,
        benchmark_questions,
        answer_config,
        delay=answer_phase_delay,
    )

    print("\nJudging all successful answers...")
    judgments = generate_judgments_batch(
        judge_client,
        judge_settings,
        config,
        benchmark_questions,
        ai_answers,
        judge_config,
        delay=judge_phase_delay,
    )

    results = []
    for entry, ai_answer, judgment in zip(benchmark_questions, ai_answers, judgments):
        question = entry["question"]
        expected_answer = entry["answer"]
        category = entry.get("category", "Unknown")
        hash_id = entry.get("hash", "")

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

    duration = time.time() - start_time

    save_detailed_results(results_dir, results)
    save_summary(results_dir, answer_config, judge_config, results, duration)

    print(f"\nResults saved to: {results_dir}")


if __name__ == "__main__":
    main()
