# QA → condensed triples pipeline

This pipeline reads your chat-style `train.jsonl` file and produces:

- `qa_clean.jsonl`: cleaned SFT data with `/no_think` removed
- `triples_raw.jsonl`: one extracted triple per row, with source QA attached
- `triples_dedup.jsonl`: exact-deduplicated triples with support counts
- `cpt_triples_pipe.jsonl`: CPT text in `subject | relation | object` format
- `cpt_triples_sentences.jsonl`: CPT text in sentence form
- `subject_relation_to_objects.jsonl`: useful for spotting statistical relations with multiple objects
- `relation_stats.json`: relation frequency report
- `qa_to_triples_debug.jsonl`: extraction metadata and any LLM errors

## Input format

The script expects the same format as your uploaded train file:

```json
{"messages": [
  {"role": "system", "content": "..."},
  {"role": "user", "content": "Question ... /no_think"},
  {"role": "assistant", "content": "Answer ..."}
]}
```

## Recommended mode

Use your local Qwen3.5-9B model in LM Studio as the extractor.

Example:

```bash
python qa_to_triples_pipeline.py \
  --input /path/to/train.jsonl \
  --output-dir /path/to/out_triples \
  --output-prefix train_ \
  --use-llm \
  --base-url http://127.0.0.1:1234/v1 \
  --api-key lm-studio \
  --model qwen/qwen3.5-9b-instruct \
  --workers 1 \
  --max-tokens 256 \
  --temperature 0.0 \
  --retries 1 \
  --allow-heuristic-fallback
```

Start with `--workers 1` in LM Studio. Only increase it if the server handles queued requests cleanly.

## Quick test on a small slice

```bash
python qa_to_triples_pipeline.py \
  --input /path/to/train.jsonl \
  --output-dir /path/to/out_small \
  --limit 100 \
  --use-llm \
  --allow-heuristic-fallback
```

Then inspect:

- `triples_raw.jsonl`
- `triples_dedup.jsonl`
- `relation_stats.json`

If the relation names are messy, edit `RELATION_ALIASES` inside the script and rerun.

## Heuristic-only mode

If you do not want to use a local model, the script can run without `--use-llm`.
That mode is only a fallback and is much worse for your dataset.

## Suggested training use

For CPT, start with the pipe format first:

```json
{"text": "subject | relation | object"}
```

For SFT, keep your original QA format, but use the cleaned `qa_clean.jsonl` or your original file with `/no_think` stripped.

## Practical workflow

1. Run the extractor on 100 examples.
2. Check relation quality and obvious errors.
3. Tighten `RELATION_ALIASES` if needed.
4. Run on the full train split.
5. CPT on `cpt_triples_pipe.jsonl`.
6. SFT on the cleaned QA data.

## Notes

- The extractor prompt allows dates, amounts, addresses, and limits as objects.
- Comparative or vague QAs may produce zero triples.
- Multi-fact answers may produce several triples.
- The best files to train on are usually the deduplicated CPT outputs, not the raw triples.
