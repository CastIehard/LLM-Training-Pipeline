#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import os
import random
import re
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from tqdm import tqdm

try:
    import requests
except Exception:
    requests = None

NO_ANSWER_STRINGS = {
    "unknown",
    "not specified",
    "not mentioned",
    "not provided",
    "not available",
    "unclear",
    "n/a",
    "none",
}

RELATION_ALIASES = {
    "is located in": "located_in",
    "located in": "located_in",
    "is in": "located_in",
    "address": "has_address",
    "location": "located_in",
    "starts on": "starts_on",
    "starts at": "starts_at",
    "begins on": "starts_on",
    "begins at": "starts_at",
    "begins": "starts",
    "starts": "starts",
    "ends on": "ends_on",
    "ends at": "ends_at",
    "ends": "ends",
    "cost": "costs",
    "costs": "costs",
    "price": "costs",
    "fee": "fee",
    "fees": "fee",
    "maximum size": "maximum_size",
    "max size": "maximum_size",
    "maximum number": "maximum_number",
    "max number": "maximum_number",
    "number of": "number_of",
    "duration": "duration",
    "deadline": "deadline",
    "requirement": "requires",
    "requirements": "requires",
    "responsible for": "responsible_for",
    "contact email": "contact_email",
    "email": "contact_email",
    "phone": "contact_phone",
    "telephone": "contact_phone",
    "website": "website",
    "url": "website",
    "new chancellor": "has_chancellor",
    "chancellor": "has_chancellor",
    "professor leading": "led_by",
    "led by": "led_by",
    "appointed as": "appointed_as",
    "can work up to": "maximum_work_limit",
}

SKIP_SUBJECTS = {
    "it",
    "they",
    "this",
    "that",
    "these",
    "those",
    "he",
    "she",
    "we",
    "you",
    "one",
    "someone",
    "people",
    "students",
}


def eprint(*args: Any) -> None:
    print(*args, file=sys.stderr)


@dataclass
class Example:
    example_id: int
    question: str
    answer: str
    system: str = ""


@dataclass
class Triple:
    example_id: int
    subject: str
    relation: str
    object: str
    question: str
    answer: str
    extractor: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "example_id": self.example_id,
            "subject": self.subject,
            "relation": self.relation,
            "object": self.object,
            "question": self.question,
            "answer": self.answer,
            "extractor": self.extractor,
        }


def clean_text(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_question(text: str) -> str:
    text = clean_text(text)
    text = re.sub(r"\s*/no_think\s*$", "", text, flags=re.IGNORECASE)
    return text.strip()


def normalize_relation_text(text: str) -> str:
    text = clean_text(text)
    text = text.strip(" .,:;!?\"'`()[]{}")
    text = unicodedata.normalize("NFKC", text).lower()
    text = text.replace("/", " ")
    text = re.sub(r"\bis\b", "", text)
    text = re.sub(r"\bare\b", "", text)
    text = re.sub(r"\bthe\b", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    if text in RELATION_ALIASES:
        return RELATION_ALIASES[text]
    text = text.replace("-", " ")
    text = re.sub(r"[^a-z0-9 ]+", "", text)
    text = re.sub(r"\s+", "_", text).strip("_")
    if text in RELATION_ALIASES:
        return RELATION_ALIASES[text]
    return text or "related_to"


def normalize_entity(text: str) -> str:
    text = clean_text(text)
    text = text.strip(" \t\n\r\"'`()[]{}")
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return text


def is_bad_value(text: str) -> bool:
    t = normalize_entity(text)
    if not t:
        return True
    if t.lower() in NO_ANSWER_STRINGS:
        return True
    if len(t) < 2:
        return True
    return False


def parse_chat_jsonl(path: Path, limit: Optional[int] = None, seed: int = 0, shuffle: bool = False) -> List[Example]:
    rows: List[Example] = []
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            obj = json.loads(line)
            msgs = obj.get("messages", [])
            system = ""
            question = ""
            answer = ""
            for m in msgs:
                role = m.get("role", "")
                content = m.get("content", "")
                if role == "system" and not system:
                    system = clean_text(content)
                elif role == "user":
                    question = clean_question(content)
                elif role == "assistant":
                    answer = clean_text(content)
            if not question or not answer:
                continue
            rows.append(Example(example_id=idx, question=question, answer=answer, system=system))
    if shuffle:
        rnd = random.Random(seed)
        rnd.shuffle(rows)
    if limit is not None:
        rows = rows[:limit]
    return rows


def json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=False)


def extract_json_block(text: str) -> Optional[Any]:
    text = text.strip()
    candidates = [text]
    if "```" in text:
        fenced = re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
        candidates = fenced + candidates
    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except Exception:
            pass
        first_obj = candidate.find("{")
        last_obj = candidate.rfind("}")
        if first_obj != -1 and last_obj != -1 and last_obj > first_obj:
            try:
                return json.loads(candidate[first_obj:last_obj + 1])
            except Exception:
                pass
        first_arr = candidate.find("[")
        last_arr = candidate.rfind("]")
        if first_arr != -1 and last_arr != -1 and last_arr > first_arr:
            try:
                return json.loads(candidate[first_arr:last_arr + 1])
            except Exception:
                pass
    return None


def heuristic_extract(example: Example) -> List[Dict[str, str]]:
    q = example.question.rstrip(" ?")
    a = normalize_entity(example.answer)
    triples: List[Dict[str, str]] = []

    patterns: List[Tuple[str, Any]] = [
        (r"^What is the address of (.+)$", lambda m: (m.group(1), "has_address", a)),
        (r"^Where is (.+?) located$", lambda m: (m.group(1), "located_in", a)),
        (r"^Where is (.+)$", lambda m: (m.group(1), "located_in", a)),
        (r"^When does (.+?) begin(?: and at what time)?$", lambda m: (m.group(1), "starts", a)),
        (r"^When does (.+?) start(?: and at what time)?$", lambda m: (m.group(1), "starts", a)),
        (r"^When does (.+?) end$", lambda m: (m.group(1), "ends", a)),
        (r"^Who is the professor leading (.+)$", lambda m: (m.group(1), "led_by", a)),
        (r"^Who has been appointed as the new chancellor of (.+)$", lambda m: (m.group(1), "has_chancellor", a)),
        (r"^Who is responsible for (.+)$", lambda m: (m.group(1), "responsible_for", a)),
        (r"^What is the maximum size of (.+)$", lambda m: (m.group(1), "maximum_size", a)),
        (r"^What is the maximum number of (.+)$", lambda m: (m.group(1), "maximum_number", a)),
        (r"^How many (.+?) does (.+)$", lambda m: (m.group(2), f"number_of_{normalize_relation_text(m.group(1))}", a)),
        (r"^Which city is specifically mentioned as a location where (.+)$", lambda m: (m.group(1), "mentioned_location", a)),
    ]

    for pattern, fn in patterns:
        m = re.match(pattern, q, flags=re.IGNORECASE)
        if m:
            subject, relation, obj = fn(m)
            triples.append({
                "subject": normalize_entity(subject),
                "relation": normalize_relation_text(relation),
                "object": normalize_entity(obj),
            })
            return triples

    wh_map = [
        ("when", "answer_for_when"),
        ("where", "answer_for_where"),
        ("who", "answer_for_who"),
        ("what", "answer_for_what"),
        ("which", "answer_for_which"),
        ("how", "answer_for_how"),
        ("can", "answer_for_can"),
        ("does", "answer_for_does"),
        ("is", "answer_for_is"),
        ("are", "answer_for_are"),
    ]
    q_lower = q.lower()
    relation = "answer"
    for prefix, rel in wh_map:
        if q_lower.startswith(prefix + " "):
            relation = rel
            break
    triples.append({
        "subject": normalize_entity(q),
        "relation": normalize_relation_text(relation),
        "object": normalize_entity(a),
    })
    return triples


def build_messages(example: Example) -> List[Dict[str, str]]:
    system_prompt = (
        "You extract atomic factual triples from closed-book QA pairs. "
        "Return only JSON. Use the answer as the factual source of truth. "
        "Keep only self-contained, meaningful facts. "
        "Prefer one named entity or one clearly defined concept as the subject. "
        "Use a short normalized relation in snake_case. "
        "Use the smallest answer span that preserves the fact. "
        "If one QA contains multiple independent facts, output multiple triples. "
        "If the QA is comparative or too vague for a reliable triple, output an empty list."
    )
    user_prompt = f"""
Extract atomic factual triples from this QA pair.

Rules:
1. Use only facts supported by the answer.
2. Keep each triple self-contained.
3. Avoid pronouns as subjects.
4. Normalize relations to short snake_case labels.
5. Keep subjects and objects concise, but preserve exact facts like dates, numbers, addresses, names, and limits.
6. You may use concepts as objects, not only named entities, because this dataset contains dates, amounts, conditions, and definitions.
7. Output JSON with exactly this schema:
{{"triples": [{{"subject": "...", "relation": "...", "object": "..."}}]}}

Examples:
Q: What is the address of Stab Wohnen?
A: Stab Wohnen is located at Marienstraße 6, 90402 Nürnberg.
JSON:
{{"triples": [{{"subject": "Stab Wohnen", "relation": "has_address", "object": "Marienstraße 6, 90402 Nürnberg"}}]}}

Q: When does the student mandatory insurance end?
A: It generally ends with the semester in which the student turns 30, unless exceptions apply.
JSON:
{{"triples": [{{"subject": "student mandatory insurance", "relation": "ends", "object": "the semester in which the student turns 30"}}]}}

Q: What is the main difference between family insurance and student mandatory insurance?
A: Family insurance is free and based on income limits, while student mandatory insurance is a specific statutory insurance for students under 30 or under certain conditions.
JSON:
{{"triples": [
  {{"subject": "family insurance", "relation": "cost", "object": "free"}},
  {{"subject": "family insurance", "relation": "based_on", "object": "income limits"}},
  {{"subject": "student mandatory insurance", "relation": "is", "object": "a specific statutory insurance for students under 30 or under certain conditions"}}
]}}

Now process this pair.
Q: {example.question}
A: {example.answer}
""".strip()
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def call_openai_compatible(base_url: str, api_key: str, model: str, messages: List[Dict[str, str]], timeout: int, max_tokens: int, temperature: float) -> str:
    if requests is None:
        raise RuntimeError("The 'requests' package is required for LLM extraction.")
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    response = requests.post(url, headers=headers, json=payload, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]


def normalize_triplet_obj(obj: Dict[str, Any]) -> Optional[Dict[str, str]]:
    subject = normalize_entity(str(obj.get("subject", "")))
    relation = normalize_relation_text(str(obj.get("relation", "")))
    out = normalize_entity(str(obj.get("object", "")))
    if is_bad_value(subject) or is_bad_value(out) or not relation:
        return None
    if subject.lower() in SKIP_SUBJECTS:
        return None
    if relation == "related_to":
        return None
    if subject.lower() == out.lower():
        return None
    return {"subject": subject, "relation": relation, "object": out}


def extract_with_llm(example: Example, *, base_url: str, api_key: str, model: str, timeout: int, max_tokens: int, temperature: float, retries: int) -> Tuple[List[Dict[str, str]], str]:
    last_err = ""
    for attempt in range(retries + 1):
        try:
            raw = call_openai_compatible(
                base_url=base_url,
                api_key=api_key,
                model=model,
                messages=build_messages(example),
                timeout=timeout,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            parsed = extract_json_block(raw)
            if parsed is None:
                raise ValueError(f"Could not parse JSON from model output: {raw[:400]}")
            triples = parsed.get("triples", parsed if isinstance(parsed, list) else [])
            if not isinstance(triples, list):
                raise ValueError(f"Expected list of triples, got: {type(triples).__name__}")
            cleaned: List[Dict[str, str]] = []
            seen = set()
            for item in triples:
                if not isinstance(item, dict):
                    continue
                t = normalize_triplet_obj(item)
                if not t:
                    continue
                key = (t["subject"].lower(), t["relation"], t["object"].lower())
                if key in seen:
                    continue
                seen.add(key)
                cleaned.append(t)
            return cleaned, raw
        except Exception as exc:
            last_err = str(exc)
            time.sleep(min(2.0 * (attempt + 1), 5.0))
    raise RuntimeError(last_err)


def extract_one(example: Example, args: argparse.Namespace) -> Tuple[Example, List[Triple], Dict[str, Any]]:
    meta: Dict[str, Any] = {"example_id": example.example_id, "question": example.question, "answer": example.answer}
    triples: List[Dict[str, str]] = []
    raw_output = None
    extractor = "heuristic"

    if args.use_llm:
        try:
            triples, raw_output = extract_with_llm(
                example,
                base_url=args.base_url,
                api_key=args.api_key,
                model=args.model,
                timeout=args.timeout,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                retries=args.retries,
            )
            extractor = "llm"
        except Exception as exc:
            meta["llm_error"] = str(exc)
            if not args.allow_heuristic_fallback:
                return example, [], meta

    if not triples:
        triples = [t for t in heuristic_extract(example) if normalize_triplet_obj(t)]
        extractor = "heuristic"

    out: List[Triple] = []
    seen = set()
    for item in triples:
        t = normalize_triplet_obj(item)
        if not t:
            continue
        key = (t["subject"].lower(), t["relation"], t["object"].lower())
        if key in seen:
            continue
        seen.add(key)
        out.append(
            Triple(
                example_id=example.example_id,
                subject=t["subject"],
                relation=t["relation"],
                object=t["object"],
                question=example.question,
                answer=example.answer,
                extractor=extractor,
            )
        )

    meta["extractor"] = extractor
    meta["num_triples"] = len(out)
    if raw_output is not None:
        meta["raw_model_output"] = raw_output
    return example, out, meta


def verbalize_relation(rel: str) -> str:
    return rel.replace("_", " ").strip()


def triple_to_sentence(subject: str, relation: str, obj: str) -> str:
    rel = verbalize_relation(relation)
    sentence = f"{subject} {rel} {obj}."
    sentence = re.sub(r"\s+", " ", sentence).strip()
    return sentence


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json_dumps(row) + "\n")


def build_outputs(examples: List[Example], extracted: List[Triple], metas: List[Dict[str, Any]], out_dir: Path, prefix: str) -> Dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    qa_clean_path = out_dir / f"{prefix}qa_clean.jsonl"
    triples_raw_path = out_dir / f"{prefix}triples_raw.jsonl"
    triples_dedup_path = out_dir / f"{prefix}triples_dedup.jsonl"
    cpt_pipe_path = out_dir / f"{prefix}cpt_triples_pipe.jsonl"
    cpt_sentence_path = out_dir / f"{prefix}cpt_triples_sentences.jsonl"
    qa2triple_path = out_dir / f"{prefix}qa_to_triples_debug.jsonl"
    rel_stats_path = out_dir / f"{prefix}relation_stats.json"
    subj_rel_objects_path = out_dir / f"{prefix}subject_relation_to_objects.jsonl"
    summary_path = out_dir / f"{prefix}summary.json"

    write_jsonl(
        qa_clean_path,
        ({"example_id": ex.example_id, "question": ex.question, "answer": ex.answer, "system": ex.system} for ex in examples),
    )
    write_jsonl(triples_raw_path, (t.as_dict() for t in extracted))
    write_jsonl(qa2triple_path, metas)

    grouped: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for t in extracted:
        key = (t.subject, t.relation, t.object)
        if key not in grouped:
            grouped[key] = {
                "subject": t.subject,
                "relation": t.relation,
                "object": t.object,
                "support_count": 0,
                "example_ids": [],
                "questions": [],
                "answers": [],
                "extractors": Counter(),
            }
        entry = grouped[key]
        entry["support_count"] += 1
        entry["example_ids"].append(t.example_id)
        if len(entry["questions"]) < 5:
            entry["questions"].append(t.question)
        if len(entry["answers"]) < 5:
            entry["answers"].append(t.answer)
        entry["extractors"][t.extractor] += 1

    dedup_rows: List[Dict[str, Any]] = []
    for entry in grouped.values():
        row = dict(entry)
        row["extractors"] = dict(row["extractors"])
        dedup_rows.append(row)
    dedup_rows.sort(key=lambda x: (-x["support_count"], x["relation"], x["subject"], x["object"]))
    write_jsonl(triples_dedup_path, dedup_rows)

    cpt_pipe_rows = []
    cpt_sentence_rows = []
    for row in dedup_rows:
        s, r, o = row["subject"], row["relation"], row["object"]
        cpt_pipe_rows.append({
            "text": f"{s} | {r} | {o}",
            "subject": s,
            "relation": r,
            "object": o,
            "support_count": row["support_count"],
        })
        cpt_sentence_rows.append({
            "text": triple_to_sentence(s, r, o),
            "subject": s,
            "relation": r,
            "object": o,
            "support_count": row["support_count"],
        })
    write_jsonl(cpt_pipe_path, cpt_pipe_rows)
    write_jsonl(cpt_sentence_path, cpt_sentence_rows)

    sr_to_objects: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    for row in dedup_rows:
        sr_to_objects[(row["subject"], row["relation"])].append(row["object"])
    sr_rows = []
    for (subject, relation), objects in sr_to_objects.items():
        sr_rows.append({
            "subject": subject,
            "relation": relation,
            "objects": sorted(set(objects)),
            "object_count": len(set(objects)),
        })
    sr_rows.sort(key=lambda x: (-x["object_count"], x["relation"], x["subject"]))
    write_jsonl(subj_rel_objects_path, sr_rows)

    relation_counter = Counter(t.relation for t in extracted)
    relation_subject_counter = Counter((t.subject, t.relation) for t in extracted)
    relation_stats = {
        "relations": relation_counter.most_common(),
        "top_subject_relation_pairs": [
            {"subject": s, "relation": r, "count": c} for (s, r), c in relation_subject_counter.most_common(200)
        ],
    }
    rel_stats_path.write_text(json.dumps(relation_stats, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "num_examples": len(examples),
        "num_extracted_triples_raw": len(extracted),
        "num_unique_triples": len(dedup_rows),
        "num_subject_relation_pairs": len(sr_rows),
        "num_relations": len(relation_counter),
        "avg_triples_per_example": (len(extracted) / len(examples)) if examples else 0.0,
        "files": {
            "qa_clean": str(qa_clean_path),
            "triples_raw": str(triples_raw_path),
            "triples_dedup": str(triples_dedup_path),
            "cpt_triples_pipe": str(cpt_pipe_path),
            "cpt_triples_sentences": str(cpt_sentence_path),
            "qa_to_triples_debug": str(qa2triple_path),
            "relation_stats": str(rel_stats_path),
            "subject_relation_to_objects": str(subj_rel_objects_path),
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "qa_clean": qa_clean_path,
        "triples_raw": triples_raw_path,
        "triples_dedup": triples_dedup_path,
        "cpt_triples_pipe": cpt_pipe_path,
        "cpt_triples_sentences": cpt_sentence_path,
        "qa_to_triples_debug": qa2triple_path,
        "relation_stats": rel_stats_path,
        "subject_relation_to_objects": subj_rel_objects_path,
        "summary": summary_path,
    }


def run(args: argparse.Namespace) -> Dict[str, Path]:
    examples = parse_chat_jsonl(INPUT_PATH, limit=args.limit, seed=args.seed, shuffle=args.shuffle)
    eprint(f"Loaded {len(examples)} examples from {INPUT_PATH}")

    extracted: List[Triple] = []
    metas: List[Dict[str, Any]] = []

    if args.workers <= 1:
        for idx, ex in tqdm(enumerate(examples, start=1), total=len(examples), desc="Tuple creation"):
            _, triples, meta = extract_one(ex, args)
            extracted.extend(triples)
            metas.append(meta)
    else:
        with futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_map = {executor.submit(extract_one, ex, args): ex for ex in examples}
            done = 0
            for future in futures.as_completed(future_map):
                _, triples, meta = future.result()
                extracted.extend(triples)
                metas.append(meta)
                done += 1
                if done % max(args.progress_every, 1) == 0:
                    eprint(f"Processed {done}/{len(examples)} examples | raw triples so far: {len(extracted)}")

    paths = build_outputs(examples, extracted, metas, args.output_dir, args.output_prefix)
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    eprint(json.dumps(summary, ensure_ascii=False, indent=2))
    return paths


INPUT_PATH = Path(__file__).resolve().parent.parent / "5_sft_on_qna_peft" / "data" / "train.jsonl"


def main() -> int:
    p = argparse.ArgumentParser(description="Convert chat-style QA JSONL into condensed triple datasets for CPT.")
    p.add_argument("--output-dir", type=Path, default="./data/tuples", help="Directory for output files.")
    p.add_argument("--output-prefix", type=str, default="train_", help="Optional file prefix, e.g. 'train_'.")
    p.add_argument("--limit", type=int, default=25, help="Optional max number of examples.")
    p.add_argument("--shuffle", action="store_true", help="Shuffle before applying --limit.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=1, help="Parallel workers. Start with 1 for LM Studio.")
    p.add_argument("--progress-every", type=int, default=100)
    p.add_argument("--use-llm" ,action="store_true", help="Use an OpenAI-compatible local model for extraction.")
    p.add_argument("--base-url", type=str, default=os.environ.get("LM_STUDIO_BASE_URL", "http://127.0.0.1:1234/v1"))
    p.add_argument("--api-key", type=str, default=os.environ.get("LM_STUDIO_API_KEY", "lm-studio"))
    p.add_argument("--model", type=str, default=os.environ.get("LM_STUDIO_MODEL", "qwen/qwen3.5-9b"))
    p.add_argument("--timeout", type=int, default=120)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--retries", type=int, default=1)
    p.add_argument("--allow-heuristic-fallback", action="store_true", help="Fall back to simple regex extraction if LLM extraction fails.")
    args = p.parse_args()
    if args.use_llm and requests is None:
        eprint("ERROR: requests is not installed, but --use-llm was set.")
        return 2
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
