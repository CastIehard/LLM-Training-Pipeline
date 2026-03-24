#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from gepa import TimeoutStopCondition, SignalStopper
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from gepa.optimize_anything import GEPAConfig, EngineConfig, ReflectionConfig, optimize_anything
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "Could not import GEPA. Install it with `pip install -U gepa`."
    ) from exc


UNKNOWN_ANSWER_PATTERNS = (
    "unknown",
    "not known",
    "not specified",
    "not enough information",
    "unclear",
    "cannot determine",
    "can't determine",
    "do not know",
    "don't know",
    "no information",
)

FINAL_ANSWER_PATTERNS = [
    re.compile(r"final\s+answer\s*[:\-]\s*(.+)$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"answer\s*[:\-]\s*(.+)$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"therefore[, ]+the answer is\s*(.+)$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"so[, ]+the answer is\s*(.+)$", re.IGNORECASE | re.MULTILINE),
]


@dataclass
class QAExample:
    ex_id: str
    question: str
    answer: str
    source_system_prompt: str


@dataclass
class EvalResult:
    prompt: str
    score: float
    exact_match: float
    count: int


class LocalQwenChat:
    def __init__(
        self,
        model_path: str,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        trust_remote_code: bool = True,
    ):
        self.model_path = model_path
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda":
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            dtype = torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=dtype,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=trust_remote_code,
        )
        if self.device != "cuda":
            self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        messages = []
        if system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt.strip()})
        messages.append({"role": "user", "content": user_prompt.strip()})

        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(prompt_text, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        do_sample = self.temperature > 1e-8
        gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = self.temperature
            gen_kwargs["top_p"] = 0.95

        output_ids = self.model.generate(**inputs, **gen_kwargs)
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def normalize_openai_compatible_api_base(api_base: str) -> str:
    base = str(api_base).strip().rstrip("/")
    if not base:
        raise ValueError("OpenAI-compatible API base must not be empty.")
    return base


def root_api_base(api_base: str) -> str:
    base = normalize_openai_compatible_api_base(api_base)
    if base.endswith("/v1"):
        return base[:-3].rstrip("/")
    return base


def v1_api_base(api_base: str) -> str:
    base = normalize_openai_compatible_api_base(api_base)
    if base.endswith("/v1"):
        return base
    return base + "/v1"


def openai_compatible_models_url(api_base: str) -> str:
    return v1_api_base(api_base) + "/models"


def probe_openai_compatible_models(api_base: str, api_key: Optional[str] = None, timeout: float = 5.0) -> List[str]:
    url = openai_compatible_models_url(api_base)
    headers: Dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI-compatible server returned HTTP {exc.code} for {url}: {detail}") from exc
    except Exception as exc:
        raise RuntimeError(f"Could not reach the OpenAI-compatible server at {url}: {exc}") from exc

    out: List[str] = []
    for item in payload.get("data", []):
        if isinstance(item, dict) and item.get("id"):
            out.append(str(item["id"]))
    return out


def configure_reflection_lm(
    reflection_lm: str,
    api_base: str,
    api_key: Optional[str],
    skip_server_check: bool,
) -> Tuple[str, List[str], Optional[str], Optional[str]]:
    reflection_lm = str(reflection_lm).strip()
    if not reflection_lm:
        raise ValueError("reflection_lm must not be empty.")

    normalized_base = normalize_openai_compatible_api_base(api_base)
    base_root = root_api_base(api_base)
    base_v1 = v1_api_base(api_base)
    model_ids: List[str] = []

    if reflection_lm.startswith("openai/"):
        os.environ["OPENAI_API_BASE"] = base_v1
        if api_key is not None:
            os.environ["OPENAI_API_KEY"] = str(api_key)
        if not skip_server_check:
            model_ids = probe_openai_compatible_models(base_root, api_key)
        return reflection_lm, model_ids, os.environ.get("OPENAI_API_BASE"), os.environ.get("OPENAI_API_KEY")

    if reflection_lm.startswith("lm_studio/"):
        os.environ["LM_STUDIO_API_BASE"] = base_v1
        if api_key is not None:
            os.environ["LM_STUDIO_API_KEY"] = str(api_key)
        if not skip_server_check:
            model_ids = probe_openai_compatible_models(base_root, api_key)
        return reflection_lm, model_ids, os.environ.get("LM_STUDIO_API_BASE"), os.environ.get("LM_STUDIO_API_KEY")

    return reflection_lm, [], None, None


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_idx} of {path}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected a JSON object on line {line_idx} of {path}, got {type(row).__name__}.")
            rows.append(row)
    return rows


def _coerce_content(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
                else:
                    parts.append(json.dumps(item, ensure_ascii=False, sort_keys=True))
            else:
                parts.append(str(item))
        return "\n".join(x.strip() for x in parts if str(x).strip()).strip()
    return str(content).strip()


def extract_messages_example(obj: Dict[str, Any], idx: int, path: Path) -> QAExample:
    messages = obj.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"{path} example {idx} does not contain a non-empty `messages` list.")

    system_parts: List[str] = []
    user_parts: List[str] = []
    assistant_parts: List[str] = []

    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role", "")).strip().lower()
        content = _coerce_content(msg.get("content", ""))
        if not content:
            continue
        if role == "system":
            system_parts.append(content)
        elif role == "user":
            user_parts.append(content)
        elif role == "assistant":
            assistant_parts.append(content)

    if not user_parts or not assistant_parts:
        raise ValueError(
            f"{path} example {idx} must contain at least one user message and one assistant message."
        )

    system_prompt = "\n\n".join(system_parts).strip()
    question = "\n\n".join(user_parts).strip()
    answer = "\n\n".join(assistant_parts).strip()

    if not question or not answer:
        raise ValueError(f"{path} example {idx} has an empty question or answer after parsing.")

    return QAExample(
        ex_id=str(obj.get("id", idx)),
        question=question,
        answer=answer,
        source_system_prompt=system_prompt,
    )


def load_chat_dataset(path: str) -> List[QAExample]:
    p = Path(path)
    rows = read_jsonl(p)
    return [extract_messages_example(row, idx=i, path=p) for i, row in enumerate(rows)]


def maybe_limit_examples(examples: Sequence[QAExample], max_examples: Optional[int], seed: int) -> List[QAExample]:
    examples = list(examples)
    if max_examples is None or max_examples <= 0 or max_examples >= len(examples):
        return examples
    rng = random.Random(seed)
    indices = list(range(len(examples)))
    rng.shuffle(indices)
    chosen = sorted(indices[:max_examples])
    return [examples[i] for i in chosen]


def infer_seed_prompt(
    train_examples: Sequence[QAExample],
    valid_examples: Sequence[QAExample],
    seed_prompt_file: Optional[str],
    seed_prompt_inline: Optional[str],
) -> str:
    if seed_prompt_file:
        return Path(seed_prompt_file).read_text(encoding="utf-8").strip()
    if seed_prompt_inline:
        return seed_prompt_inline.strip()

    prompts = Counter()
    for ex in list(train_examples) + list(valid_examples):
        if ex.source_system_prompt.strip():
            prompts[ex.source_system_prompt.strip()] += 1

    if not prompts:
        return (
            "You answer questions factually and concisely. Do not invent information. "
            "If the answer is unknown, say so clearly."
        )
    return prompts.most_common(1)[0][0]


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = text.replace("€", " euros ")
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"/no_think\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[\"'`´]", "", text)
    text = re.sub(r"[^a-z0-9äöüß\.\,\-\+\s]", " ", text)
    text = re.sub(r"\b(one|a|an)\b", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def maybe_extract_final_answer(text: str) -> str:
    text = text.strip()
    for pattern in FINAL_ANSWER_PATTERNS:
        matches = pattern.findall(text)
        if matches:
            cand = matches[-1].strip()
            if cand:
                return cand
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return text
    short_lines = [line for line in lines if len(line.split()) <= 16]
    return short_lines[-1] if short_lines else lines[-1]


def parse_number(text: str) -> Optional[float]:
    cleaned = text.replace(",", "")
    m = re.search(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?(?![A-Za-z])", cleaned)
    if not m:
        return None
    try:
        return float(m.group(0))
    except Exception:
        return None


def token_f1(pred: str, gold: str) -> float:
    pred_tokens = normalize_text(pred).split()
    gold_tokens = normalize_text(gold).split()
    if not pred_tokens or not gold_tokens:
        return 0.0

    pred_counts: Dict[str, int] = {}
    gold_counts: Dict[str, int] = {}
    for tok in pred_tokens:
        pred_counts[tok] = pred_counts.get(tok, 0) + 1
    for tok in gold_tokens:
        gold_counts[tok] = gold_counts.get(tok, 0) + 1

    common = 0
    for tok, cnt in pred_counts.items():
        common += min(cnt, gold_counts.get(tok, 0))

    if common == 0:
        return 0.0

    precision = common / len(pred_tokens)
    recall = common / len(gold_tokens)
    return 2.0 * precision * recall / (precision + recall)


def brevity_penalty(pred_text: str, gold_text: str) -> float:
    pred_len = max(1, len(pred_text.split()))
    gold_len = max(1, len(gold_text.split()))
    allowed = max(18, int(2.2 * gold_len + 4))
    if pred_len <= allowed:
        return 0.0
    overflow = pred_len - allowed
    return min(0.08, 0.004 * overflow)


def score_prediction(pred_text: str, gold_text: str) -> Tuple[float, bool, str]:
    pred_final = maybe_extract_final_answer(pred_text)
    pred_norm = normalize_text(pred_final)
    gold_norm = normalize_text(gold_text)

    if not pred_norm:
        return 0.0, False, "Empty answer."

    if any(p in pred_norm for p in UNKNOWN_ANSWER_PATTERNS) and not any(p in gold_norm for p in UNKNOWN_ANSWER_PATTERNS):
        return 0.0, False, f"Model said the answer was unknown, but the gold answer is {gold_text!r}."

    if pred_norm == gold_norm:
        penalty = brevity_penalty(pred_final, gold_text)
        return max(0.0, 1.0 - penalty), True, "Exact normalized match."

    pred_num = parse_number(pred_final)
    gold_num = parse_number(gold_text)
    if pred_num is not None and gold_num is not None and math.isclose(pred_num, gold_num, rel_tol=1e-9, abs_tol=1e-9):
        penalty = brevity_penalty(pred_final, gold_text)
        return max(0.0, 1.0 - penalty), True, f"Numeric match: predicted {pred_num}, expected {gold_num}."

    if gold_norm and gold_norm in pred_norm:
        penalty = brevity_penalty(pred_final, gold_text)
        return max(0.0, 0.94 - penalty), False, "Gold answer is contained in the prediction."
    if pred_norm and len(pred_norm) >= 5 and pred_norm in gold_norm:
        return 0.88, False, "Prediction is a concise subset of the gold answer."

    f1 = token_f1(pred_final, gold_text)
    penalty = brevity_penalty(pred_final, gold_text)

    if f1 >= 0.85:
        score = min(0.92, 0.84 + 0.08 * (f1 - 0.85) / 0.15)
        return max(0.0, score - penalty), False, f"Very strong overlap with gold answer (token F1={f1:.3f})."
    if f1 >= 0.65:
        score = 0.65 + 0.20 * (f1 - 0.65) / 0.20
        return max(0.0, score - penalty), False, f"Good partial overlap with gold answer (token F1={f1:.3f})."
    if f1 >= 0.40:
        score = 0.35 + 0.20 * (f1 - 0.40) / 0.25
        return max(0.0, score - penalty), False, f"Limited overlap with gold answer (token F1={f1:.3f})."

    return 0.0, False, f"Wrong answer. Predicted {pred_final!r}, expected {gold_text!r}."


def prepare_question(text: str, strip_no_think: bool) -> str:
    out = text.strip()
    if strip_no_think:
        out = re.sub(r"\s*/no_think\b", "", out, flags=re.IGNORECASE).strip()
    return out


def evaluate_prompt_on_example(
    model: LocalQwenChat,
    candidate_prompt: str,
    example: QAExample,
    strip_no_think: bool,
) -> Tuple[float, Dict[str, Any]]:
    user_prompt = prepare_question(example.question, strip_no_think=strip_no_think)
    raw_output = model.generate(candidate_prompt, user_prompt)
    score, exact, feedback = score_prediction(raw_output, example.answer)

    side_info = {
        "example_id": example.ex_id,
        "question": user_prompt,
        "gold_answer": example.answer,
        "predicted_text": raw_output,
        "predicted_final_answer": maybe_extract_final_answer(raw_output),
        "score": score,
        "exact_match": exact,
        "execution_feedback": feedback,
        "candidate_prompt": candidate_prompt,
        "source_system_prompt": example.source_system_prompt,
    }
    return score, side_info


def aggregate_eval(
    model: LocalQwenChat,
    prompt: str,
    dataset: Sequence[QAExample],
    strip_no_think: bool,
) -> EvalResult:
    total = 0.0
    exact = 0.0
    count = 0
    for ex in dataset:
        score, info = evaluate_prompt_on_example(model, prompt, ex, strip_no_think=strip_no_think)
        total += score
        exact += 1.0 if info["exact_match"] else 0.0
        count += 1
    n = max(1, count)
    return EvalResult(prompt=prompt, score=total / n, exact_match=exact / n, count=count)


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_background(train_examples: Sequence[QAExample], valid_examples: Sequence[QAExample]) -> str:
    all_examples = list(train_examples) + list(valid_examples)
    if not all_examples:
        return (
            "This is short-form factual question answering. Answers should be direct, concise, and non-speculative."
        )

    sample_questions = [ex.question for ex in all_examples[:6]]
    has_no_think = sum(1 for ex in all_examples if "/no_think" in ex.question)
    avg_answer_len = sum(len(ex.answer.split()) for ex in all_examples) / max(1, len(all_examples))

    topic_hint = (
        "The dataset is FAQ-style and domain-specific, with many questions about studying, living, visas, "
        "insurance, and bureaucracy in Germany, plus some local Nuremberg information."
    )
    style_hint = (
        f"Answers are typically short. The average reference answer length is about {avg_answer_len:.1f} words. "
        "Many user questions literally contain the suffix `/no_think`."
        if has_no_think
        else f"Answers are typically short. The average reference answer length is about {avg_answer_len:.1f} words."
    )

    examples_hint = "Example user questions include: " + " | ".join(q.replace("\n", " ") for q in sample_questions[:4])

    return " ".join([topic_hint, style_hint, examples_hint]).strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optimize the system prompt of a local fine-tuned Qwen model with GEPA using train and validation JSONL chat datasets."
    )
    parser.add_argument("--train-dataset", type=str, default=None, help="Path to train.jsonl.")
    parser.add_argument("--valid-dataset", type=str, default=None, help="Path to valid.jsonl.")
    parser.add_argument("--dataset", type=str, default=None, help="Deprecated alias for --train-dataset.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the local fine-tuned Qwen model.")
    parser.add_argument("--seed-prompt-file", type=str, default=None, help="Optional file containing the starting system prompt.")
    parser.add_argument("--seed-prompt", type=str, default=None, help="Optional inline starting system prompt.")
    parser.add_argument("--reflection-lm", type=str, default=os.environ.get("GEPA_REFLECTION_LM", "openai/qwen3.5-9b"), help="Reflection LM for GEPA. Preferred for OpenAI-compatible server: openai/<served-model-id> using its OpenAI-compatible API. The older lm_studio/<served-model-id> form is still accepted.",)
    parser.add_argument("--api-base", type=str, default=os.environ.get("OPENAI_API_BASE", os.environ.get("LM_STUDIO_API_BASE", "http://localhost:1234/v1")), help="Base URL of the OpenAI-compatible server, usually OpenAI-compatible server. You can pass either the root URL or one ending in /v1.",)
    parser.add_argument("--api-key", type=str, default=os.environ.get("OPENAI_API_KEY", os.environ.get("LM_STUDIO_API_KEY", "lm-studio")), help="Optional API key for the OpenAI-compatible reflection server. A dummy key is fine for OpenAI-compatible server if required by the client.",)
    parser.add_argument("--skip-server-check", action="store_true", help="Skip probing the OpenAI-compatible server /v1/models endpoint before starting.")
    parser.add_argument("--max-metric-calls", type=int, default=120, help="GEPA evaluation budget.")
    parser.add_argument("--reflection-minibatch-size", type=int, default=3, help="Number of training examples shown to the reflection model per GEPA step.")
    parser.add_argument("--max-new-tokens", type=int, default=96, help="Max tokens generated by the local Qwen model per example.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Task-model decoding temperature. Keep 0.0 for stable prompt search.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used by GEPA and any dataset subsampling.")
    parser.add_argument("--run-dir", type=str, default="./runs/gepa_qwen_prompt_opt", help="Resumable GEPA run directory.")
    parser.add_argument("--max-train-examples", type=int, default=None, help="Optional cap on training examples for faster experiments.")
    parser.add_argument("--max-valid-examples", type=int, default=None, help="Deprecated alias for --search-valid-examples.")
    parser.add_argument("--search-valid-examples", type=int, default=24, help="Number of validation examples used inside GEPA during prompt search. Keep this small so reflection starts quickly.")
    parser.add_argument("--final-valid-examples", type=int, default=None, help="Optional cap for the final post-search validation report. Defaults to the full provided validation set.")
    parser.add_argument("--skip-final-full-valid-eval", action="store_true", help="Skip the expensive final evaluation on the full validation set and only report the search subset.")
    parser.add_argument("--parallel", action="store_true", help="Enable parallel GEPA evaluation. Off by default for a single local GPU.")
    parser.add_argument("--max-workers", type=int, default=4, help="Maximum GEPA workers when --parallel is enabled.")
    parser.add_argument("--strip-no-think", action="store_true", help="Remove a trailing /no_think marker from user questions before evaluation.")
    parser.add_argument("--trust-remote-code", action="store_true", default=True, help="Pass trust_remote_code=True when loading the model/tokenizer.")
    parser.add_argument("--no-trust-remote-code", dest="trust_remote_code", action="store_false", help="Disable trust_remote_code.")
    parser.add_argument("--time-budget-minutes", type=float, default=None, help="Optional wall-clock time budget for GEPA.")
    args = parser.parse_args()

    train_dataset = args.train_dataset or args.dataset
    if not train_dataset:
        raise ValueError("Provide --train-dataset (or the deprecated --dataset alias).")
    if not args.valid_dataset:
        raise ValueError("Provide --valid-dataset. This script expects a separate validation set and does not create an internal test split.")

    resolved_reflection_lm, server_available_models, configured_api_base, configured_api_key = configure_reflection_lm(
        reflection_lm=args.reflection_lm,
        api_base=args.api_base,
        api_key=args.api_key,
        skip_server_check=args.skip_server_check,
    )

    if resolved_reflection_lm.startswith(("openai/", "lm_studio/")):
        requested_model_id = resolved_reflection_lm.split("/", 1)[1]
        normalized_api_base = root_api_base(args.api_base)
        print(f"Reflection LM:     {resolved_reflection_lm}")
        print(f"Server base:       {normalized_api_base}")
        if server_available_models:
            print(f"Server models:     {', '.join(server_available_models)}")
            if requested_model_id not in server_available_models:
                print(
                    "WARNING: The requested reflection model id was not returned by the server /v1/models endpoint. "
                    "Make sure `--reflection-lm` matches one of the served model ids exactly."
                )

    train_examples = load_chat_dataset(train_dataset)
    valid_examples_full = load_chat_dataset(args.valid_dataset)

    train_examples = maybe_limit_examples(train_examples, args.max_train_examples, seed=args.seed)

    search_valid_cap = args.search_valid_examples
    if args.max_valid_examples is not None:
        search_valid_cap = args.max_valid_examples
    valid_examples_search = maybe_limit_examples(valid_examples_full, search_valid_cap, seed=args.seed + 1)

    final_valid_cap = None if args.skip_final_full_valid_eval else args.final_valid_examples
    valid_examples_final = maybe_limit_examples(valid_examples_full, final_valid_cap, seed=args.seed + 2)

    if not train_examples:
        raise ValueError("Training dataset is empty after loading/subsampling.")
    if not valid_examples_search:
        raise ValueError("Validation search subset is empty after loading/subsampling.")
    if not args.skip_final_full_valid_eval and not valid_examples_final:
        raise ValueError("Final validation subset is empty after loading/subsampling.")

    seed_prompt = infer_seed_prompt(
        train_examples=train_examples,
        valid_examples=valid_examples_full,
        seed_prompt_file=args.seed_prompt_file,
        seed_prompt_inline=args.seed_prompt,
    )

    print(f"Train examples:           {len(train_examples)}")
    print(f"Valid examples total:     {len(valid_examples_full)}")
    print(f"Valid examples for GEPA:  {len(valid_examples_search)}")
    if not args.skip_final_full_valid_eval:
        print(f"Valid examples final:     {len(valid_examples_final)}")
    print(f"Seed prompt chars:        {len(seed_prompt)}")

    if args.max_metric_calls <= len(valid_examples_search):
        raise ValueError(
            f"max_metric_calls={args.max_metric_calls} is too small for the chosen GEPA validation subset size ({len(valid_examples_search)}). "
            "GEPA uses full validation on its provided valset by default, so reflection may never start. "
            "Reduce --search-valid-examples or increase --max-metric-calls."
        )

    print("Loading local task model...")
    model = LocalQwenChat(
        model_path=args.model_path,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        trust_remote_code=args.trust_remote_code,
    )

    def evaluator(candidate: str, example: QAExample) -> Tuple[float, Dict[str, Any]]:
        return evaluate_prompt_on_example(model, candidate, example, strip_no_think=args.strip_no_think)

    background = build_background(train_examples, valid_examples_full)
    objective = (
        "Optimize only the system prompt for this fine-tuned Qwen chat model. "
        "Primary objective: maximize validation accuracy on this short-form factual QA task. "
        "Secondary objective: keep answers concise and direct, usually one or two short sentences, with no invented facts. "
        "The prompt should generalize across unseen validation questions in the same domain and should not overfit to individual examples."
    )

    stop_callbacks = [SignalStopper()]
    if args.time_budget_minutes is not None:
        stop_callbacks.append(TimeoutStopCondition(timeout_seconds=args.time_budget_minutes * 60))

    config = GEPAConfig(
        engine=EngineConfig(
            max_metric_calls=args.max_metric_calls,
            cache_evaluation=True,
            run_dir=args.run_dir,
            parallel=bool(args.parallel),
            max_workers=args.max_workers if args.parallel else None,
            seed=args.seed,
            val_evaluation_policy="full_eval",
            display_progress_bar=True,
            track_best_outputs=True,
        ),
        reflection=ReflectionConfig(
            reflection_lm=resolved_reflection_lm,
            reflection_minibatch_size=args.reflection_minibatch_size,
            skip_perfect_score=True,
            perfect_score=1.0,
        ),
        stop_callbacks=stop_callbacks or None,
    )

    print(
        "Starting GEPA on the reduced validation subset. "
        "LM Studio is only contacted once GEPA reaches a reflection/proposal step after the initial local evaluations."
    )

    result = optimize_anything(
        seed_candidate=seed_prompt,
        evaluator=evaluator,
        dataset=train_examples,
        valset=valid_examples_search,
        objective=objective,
        background=background,
        config=config,
    )

    best_prompt = result.best_candidate if isinstance(result.best_candidate, str) else str(result.best_candidate)

    baseline_search_valid = aggregate_eval(
        model=model,
        prompt=seed_prompt,
        dataset=valid_examples_search,
        strip_no_think=args.strip_no_think,
    )
    optimized_search_valid = aggregate_eval(
        model=model,
        prompt=best_prompt,
        dataset=valid_examples_search,
        strip_no_think=args.strip_no_think,
    )

    baseline_final_valid = None
    optimized_final_valid = None
    if not args.skip_final_full_valid_eval:
        print("Running final validation report...")
        baseline_final_valid = aggregate_eval(
            model=model,
            prompt=seed_prompt,
            dataset=valid_examples_final,
            strip_no_think=args.strip_no_think,
        )
        optimized_final_valid = aggregate_eval(
            model=model,
            prompt=best_prompt,
            dataset=valid_examples_final,
            strip_no_think=args.strip_no_think,
        )

    out_dir = Path(args.run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_prompt_path = out_dir / "best_system_prompt.txt"
    best_prompt_path.write_text(best_prompt + "\n", encoding="utf-8")

    summary = {
        "model_path": args.model_path,
        "train_dataset": str(Path(train_dataset).resolve()),
        "valid_dataset": str(Path(args.valid_dataset).resolve()),
        "reflection_lm": resolved_reflection_lm,
        "api_base": v1_api_base(args.api_base) if resolved_reflection_lm.startswith(("openai/", "lm_studio/")) else None,
        "configured_api_base": configured_api_base,
        "server_available_models": server_available_models,
        "seed": args.seed,
        "max_metric_calls": args.max_metric_calls,
        "reflection_minibatch_size": args.reflection_minibatch_size,
        "parallel": bool(args.parallel),
        "max_workers": args.max_workers if args.parallel else None,
        "strip_no_think": bool(args.strip_no_think),
        "seed_prompt": seed_prompt,
        "best_prompt": best_prompt,
        "sizes": {
            "train": len(train_examples),
            "valid_total": len(valid_examples_full),
            "valid_search": len(valid_examples_search),
            "valid_final": 0 if args.skip_final_full_valid_eval else len(valid_examples_final),
        },
        "baseline_search_valid": baseline_search_valid.__dict__,
        "optimized_search_valid": optimized_search_valid.__dict__,
        "baseline_final_valid": None if baseline_final_valid is None else baseline_final_valid.__dict__,
        "optimized_final_valid": None if optimized_final_valid is None else optimized_final_valid.__dict__,
        "best_prompt_path": str(best_prompt_path.resolve()),
    }
    save_json(out_dir / "summary.json", summary)

    print("\n=== GEPA prompt optimization finished ===")
    print(f"Run dir:                 {out_dir}")
    print(f"Best prompt file:        {best_prompt_path}")
    print(f"Summary file:            {out_dir / 'summary.json'}")
    print(f"Baseline search score:   {baseline_search_valid.score:.4f} | exact: {baseline_search_valid.exact_match:.4f}")
    print(f"Optimized search score:  {optimized_search_valid.score:.4f} | exact: {optimized_search_valid.exact_match:.4f}")
    if baseline_final_valid is not None and optimized_final_valid is not None:
        print(f"Baseline final score:    {baseline_final_valid.score:.4f} | exact: {baseline_final_valid.exact_match:.4f}")
        print(f"Optimized final score:   {optimized_final_valid.score:.4f} | exact: {optimized_final_valid.exact_match:.4f}")
    print("\n=== Best system prompt ===\n")
    print(best_prompt)


if __name__ == "__main__":
    main()
