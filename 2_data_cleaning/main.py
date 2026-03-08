"""
Data Cleaning Module for UTN Project.

This script cleans scraped markdown documents by applying the following steps:
1. Text normalization (punctuation spacing, boilerplate removal, custom regex)
2. Remove harmful text (keyword-based filtering)
3. Remove offensive language & slangs (keyword-based filtering)
4. Remove PII (regex-based redaction of emails, phones, SSNs, etc.)
5. Filter by length (remove too short or too long documents)
6. Deduplicate (MinHash or Bloom filter based near-duplicate detection)

Reads from data/raw_md/, writes cleaned files to data/cleaned_md/,
and updates data/index.json with cleaning metadata.
"""

import hashlib
import json
import math
import os
import re
import shutil
from pathlib import Path

import yaml
from tqdm.auto import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def load_config(config_path: str = "2_data_cleaning/config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Text Normalization
# ---------------------------------------------------------------------------


def normalize_text(text: str, config: dict) -> str:
    """Apply text normalization: punctuation fixes, boilerplate removal, custom regex."""
    section = config.get("text_normalization", {})
    if not section.get("enabled", False):
        return text

    # Fix punctuation spacing (e.g. "word , word" -> "word, word")
    if section.get("fix_punctuation_spacing", False):
        text = re.sub(r"\s+([,\.;:!?])", r"\1", text)

    # Apply custom regex replacements (MULTILINE so ^ and $ match each line)
    for rule in section.get("custom_regex_replacements", []):
        pattern = rule.get("pattern", "")
        replacement = rule.get("replacement", "")
        if pattern:
            text = re.sub(pattern, replacement, text, flags=re.MULTILINE)

    # Collapse runs of blank lines left after boilerplate removal
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# ---------------------------------------------------------------------------
# Markdown Cleaning (moved from 1_webscraping pipeline)
# ---------------------------------------------------------------------------


def clean_markdown(text: str, config: dict) -> str:
    """Clean raw markdown: remove URLs, emails, junk patterns, nav menus, normalise whitespace."""
    section = config.get("markdown_cleaning", {})
    if not section.get("enabled", False):
        return text

    # Remove URLs: markdown links → keep link text, then strip standalone URLs
    if section.get("remove_urls", False):
        text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
        text = re.sub(r"https?://[^\s\)]+", "", text)
        text = re.sub(r"www\.[^\s]+", "", text)

    # Remove emails
    if section.get("remove_emails", False):
        text = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "", text)

    # Remove junk patterns
    if section.get("remove_junk_patterns", False):
        for pattern in section.get("junk_patterns", []):
            text = re.sub(pattern, "", text)

    # Remove navigation-menu bullet lines (lines that are just '* text')
    if section.get("remove_navigation_menus", False):
        lines = text.split("\n")
        nav_pattern = re.compile(r"^\* .+$")
        lines = [line for line in lines if not nav_pattern.match(line.strip())]
        text = "\n".join(lines)

    # Normalise whitespace first so separated image lines collapse
    # (e.g. "Finden\n\n\n![](...)" becomes "Finden\n\n![](...)")
    if section.get("normalize_whitespace", False):
        # Multiple spaces → single space
        text = re.sub(r" +", " ", text)
        # Multiple blank lines → max two newlines
        text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
        # Trailing whitespace per line
        text = re.sub(r"[ \t]+$", "", text, flags=re.MULTILINE)

        # Strip leading whitespace but preserve code blocks
        lines = text.split("\n")
        cleaned_lines = []
        in_code_block = False
        for line in lines:
            if line.strip().startswith("```"):
                in_code_block = not in_code_block
                cleaned_lines.append(line)
            elif in_code_block:
                cleaned_lines.append(line)
            else:
                cleaned_lines.append(line.lstrip())
        text = "\n".join(cleaned_lines)
        text = text.strip()

    # Remove ALL markdown images (after whitespace normalisation so
    # lines like "Finden\n\n![](/ \"title\")" are already collapsed)
    if section.get("remove_images", False):
        text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)

    # Final cleanup: collapse any blank lines left by image removal
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    return text


# ---------------------------------------------------------------------------
# PII Removal
# ---------------------------------------------------------------------------

PII_PATTERNS: dict[str, re.Pattern] = {
    "email": re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),
    "phone": re.compile(
        r"(?<!\d)"
        r"(?:\+?\d{1,3}[\s\-]?)?"
        r"(?:\(?\d{2,5}\)?[\s\-]?)?"
        r"\d{3,4}[\s\-]?\d{3,5}"
        r"(?!\d)"
    ),
    "ip_address": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    "iban": re.compile(
        r"\b[A-Z]{2}\d{2}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{0,4}\s?\d{0,2}\b"
    ),
    "credit_card": re.compile(r"\b(?:\d[ \-]?){13,19}\b"),
    "social_security_number": re.compile(r"\b\d{3}[\s\-]?\d{2}[\s\-]?\d{4}\b"),
    "passport_number": re.compile(r"\b[A-Z]{1,2}\d{6,9}\b"),
    "physical_address": re.compile(
        r"\b\d{1,5}\s[A-Z][a-z]+(?:\s[A-Z][a-z]+){0,3}"
        r"\s(?:St(?:reet|\.)?|Ave(?:nue|\.)?|Blvd|Rd|Road|Dr(?:ive|\.)?|Ln|Lane"
        r"|Way|Ct|Pl(?:ace|\.)?|Straße|Str\.|Weg|Platz|Gasse)\b"
    ),
}

PII_REPLACEMENTS: dict[str, str] = {
    "email": "[EMAIL]",
    "phone": "[PHONE]",
    "ip_address": "[IP_ADDRESS]",
    "iban": "[IBAN]",
    "credit_card": "[CREDIT_CARD]",
    "social_security_number": "[SSN]",
    "passport_number": "[PASSPORT]",
    "physical_address": "[ADDRESS]",
}


def remove_pii(text: str, config: dict) -> tuple[str, int]:
    """
    Redact PII from text based on enabled patterns in config.

    Returns (cleaned_text, count_of_redactions).
    """
    pii_config = config.get("pii_removal", {})
    if not pii_config.get("enabled", False):
        return text, 0

    total_redactions = 0
    patterns = pii_config.get("patterns", {})

    for pii_type, enabled in patterns.items():
        if not enabled:
            continue
        pattern = PII_PATTERNS.get(pii_type)
        if pattern is None:
            continue
        replacement = PII_REPLACEMENTS.get(pii_type, "[REDACTED]")
        text, count = pattern.subn(replacement, text)
        total_redactions += count

    return text, total_redactions


# ---------------------------------------------------------------------------
# Harmful / Offensive Content Detection
# ---------------------------------------------------------------------------


def contains_keywords(text: str, keywords: list[str]) -> str | None:
    """
    Check if text contains any of the keywords (case-insensitive).

    Returns the first matched keyword or None.
    """
    text_lower = text.lower()
    for kw in keywords:
        if kw.lower() in text_lower:
            return kw
    return None


def check_harmful(text: str, config: dict) -> str | None:
    """Return matched keyword if harmful content detected, else None."""
    section = config.get("harmful_content", {})
    if not section.get("enabled", False):
        return None
    return contains_keywords(text, section.get("keywords", []))


def check_offensive(text: str, config: dict) -> str | None:
    """Return matched keyword if offensive language or slang detected, else None."""
    section = config.get("offensive_language", {})
    if not section.get("enabled", False):
        return None

    # Check offensive keywords
    match = contains_keywords(text, section.get("keywords", []))
    if match:
        return match

    # Check slangs (whole-word match to reduce false positives)
    if section.get("filter_slangs", False):
        text_lower = text.lower()
        for slang in section.get("slang_keywords", []):
            pattern = r"\b" + re.escape(slang.lower()) + r"\b"
            if re.search(pattern, text_lower):
                return f"slang: {slang}"

    return None


# ---------------------------------------------------------------------------
# Length Filtering
# ---------------------------------------------------------------------------


def check_length(text: str, config: dict) -> str | None:
    """
    Return a reason string if the document fails length checks, else None.
    """
    section = config.get("length_filter", {})
    if not section.get("enabled", False):
        return None

    min_len = section.get("min_length", 0)
    max_len = section.get("max_length", float("inf"))
    doc_len = len(text)

    if doc_len < min_len:
        return f"too_short ({doc_len} < {min_len})"
    if doc_len > max_len:
        return f"too_long ({doc_len} > {max_len})"
    return None


# ---------------------------------------------------------------------------
# Deduplication – MinHash
# ---------------------------------------------------------------------------


def _text_to_shingles(text: str, k: int = 5) -> set[str]:
    """Convert text into a set of word-level k-shingles."""
    words = text.lower().split()
    if len(words) < k:
        return {" ".join(words)}
    return {" ".join(words[i : i + k]) for i in range(len(words) - k + 1)}


def _minhash_signature(shingles: set[str], num_perm: int) -> list[int]:
    """
    Compute a MinHash signature for a set of shingles.

    Uses the format: hash_i(x) = (a_i * hash(x) + b_i) mod p
    with a large prime p and random a, b per permutation.
    """
    import random

    max_hash = (1 << 32) - 1
    prime = 4294967311  # prime > 2^32

    # Deterministic seed so signatures are reproducible across runs
    rng = random.Random(42)
    coeffs = [
        (rng.randint(1, max_hash), rng.randint(0, max_hash)) for _ in range(num_perm)
    ]

    # Pre-hash all shingles once
    hashed_shingles = [hash(s) & max_hash for s in shingles]

    signature = []
    for a, b in coeffs:
        min_val = max_hash + 1
        for h in hashed_shingles:
            val = ((a * h + b) % prime) & max_hash
            if val < min_val:
                min_val = val
        signature.append(min_val)

    return signature


def _jaccard_minhash(sig_a: list[int], sig_b: list[int]) -> float:
    """Estimate Jaccard similarity from two MinHash signatures."""
    if len(sig_a) != len(sig_b):
        raise ValueError("Signatures must have equal length")
    return sum(1 for a, b in zip(sig_a, sig_b) if a == b) / len(sig_a)


def deduplicate_minhash(documents: dict[str, str], config: dict) -> set[str]:
    """
    Find near-duplicate documents using MinHash.

    Args:
        documents: mapping of hash -> text content
        config: deduplication config section

    Returns:
        Set of document hashes that are duplicates (to be removed).
    """
    mh_cfg = config.get("deduplication", {}).get("minhash", {})
    num_perm = mh_cfg.get("num_perm", 128)
    threshold = mh_cfg.get("threshold", 0.8)
    shingle_size = mh_cfg.get("shingle_size", 5)

    # Build signatures
    sigs: dict[str, list[int]] = {}
    doc_hashes = list(documents.keys())

    print("  Building MinHash signatures ...")
    for doc_hash in tqdm(doc_hashes, desc="  Signatures", leave=False):
        shingles = _text_to_shingles(documents[doc_hash], k=shingle_size)
        if shingles:
            sigs[doc_hash] = _minhash_signature(shingles, num_perm)

    # Compare pairs
    duplicates: set[str] = set()
    keys = list(sigs.keys())
    total_comparisons = len(keys) * (len(keys) - 1) // 2

    print(f"  Comparing {len(keys)} documents ({total_comparisons} pairs) ...")
    for i in tqdm(range(len(keys)), desc="  Dedup", leave=False):
        if keys[i] in duplicates:
            continue
        for j in range(i + 1, len(keys)):
            if keys[j] in duplicates:
                continue
            sim = _jaccard_minhash(sigs[keys[i]], sigs[keys[j]])
            if sim >= threshold:
                duplicates.add(keys[j])

    return duplicates


# ---------------------------------------------------------------------------
# Deduplication – Bloom Filter
# ---------------------------------------------------------------------------


class BloomFilter:
    """Simple Bloom filter for deduplication of text shingles."""

    def __init__(self, expected_items: int, fp_rate: float):
        self.size = self._optimal_size(expected_items, fp_rate)
        self.num_hashes = self._optimal_hashes(self.size, expected_items)
        self.bit_array = bytearray(math.ceil(self.size / 8))

    @staticmethod
    def _optimal_size(n: int, p: float) -> int:
        return int(-n * math.log(p) / (math.log(2) ** 2))

    @staticmethod
    def _optimal_hashes(m: int, n: int) -> int:
        return max(1, int((m / n) * math.log(2)))

    def _get_positions(self, item: str) -> list[int]:
        positions = []
        for i in range(self.num_hashes):
            digest = hashlib.md5(f"{i}:{item}".encode("utf-8")).hexdigest()
            positions.append(int(digest, 16) % self.size)
        return positions

    def add(self, item: str) -> None:
        for pos in self._get_positions(item):
            byte_idx, bit_idx = divmod(pos, 8)
            self.bit_array[byte_idx] |= 1 << bit_idx

    def __contains__(self, item: str) -> bool:
        return all(
            (self.bit_array[pos // 8] >> (pos % 8)) & 1
            for pos in self._get_positions(item)
        )


def deduplicate_bloom(documents: dict[str, str], config: dict) -> set[str]:
    """
    Find near-duplicate documents using a Bloom filter on shingles.

    For each document, all shingles are checked against the Bloom filter.
    If the fraction of already-seen shingles exceeds the threshold the
    document is considered a duplicate.

    Returns set of duplicate document hashes.
    """
    bloom_cfg = config.get("deduplication", {}).get("bloom", {})
    threshold = config.get("deduplication", {}).get("minhash", {}).get("threshold", 0.8)
    expected = bloom_cfg.get("expected_items", 10000)
    fp_rate = bloom_cfg.get("false_positive_rate", 0.01)
    shingle_size = bloom_cfg.get("shingle_size", 5)

    bf = BloomFilter(expected, fp_rate)
    duplicates: set[str] = set()
    doc_hashes = list(documents.keys())

    print("  Running Bloom filter deduplication ...")
    for doc_hash in tqdm(doc_hashes, desc="  Bloom dedup", leave=False):
        shingles = _text_to_shingles(documents[doc_hash], k=shingle_size)
        if not shingles:
            continue

        seen_count = sum(1 for s in shingles if s in bf)
        overlap = seen_count / len(shingles)

        if overlap >= threshold:
            duplicates.add(doc_hash)
        else:
            for s in shingles:
                bf.add(s)

    return duplicates


# ---------------------------------------------------------------------------
# I/O Helpers
# ---------------------------------------------------------------------------


def load_index(index_path: str) -> list[dict]:
    """Load the index.json file."""
    with open(index_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_index(index_path: str, data: list[dict]) -> None:
    """Save the index.json file."""
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_document(raw_md_dir: str, filename: str) -> str:
    """Load a markdown document."""
    filepath = Path(raw_md_dir) / filename
    if not filepath.exists():
        return ""
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------


def main():
    """Run the data cleaning pipeline."""
    print("=" * 60)
    print("Data Cleaning Pipeline")
    print("=" * 60)

    config = load_config()

    index_path = config["index_file"]
    raw_md_dir = config["raw_md_dir"]
    cleaned_md_dir = config["cleaned_md_dir"]

    # Create output directory
    Path(cleaned_md_dir).mkdir(parents=True, exist_ok=True)

    # Load index
    index = load_index(index_path)
    print(f"\nLoaded {len(index)} documents from index.")

    # --- Phase 1: Per-document cleaning ---
    print("\n--- Phase 1: Per-document filtering, normalization & PII removal ---")

    kept_documents: dict[str, str] = {}  # hash -> cleaned content
    removal_stats = {
        "harmful": 0,
        "offensive": 0,
        "too_short": 0,
        "too_long": 0,
        "file_missing": 0,
    }
    pii_total = 0
    norm_count = 0

    for entry in tqdm(index, desc="Cleaning"):
        doc_hash = entry["hash"]
        filename = entry["filename"]

        # Load content
        content = load_document(raw_md_dir, filename)
        if not content:
            entry["cleaning_status"] = "removed"
            entry["cleaning_reason"] = "file_missing"
            removal_stats["file_missing"] += 1
            continue

        # Text normalization (punctuation, boilerplate, custom regex)
        original_len = len(content)
        content = normalize_text(content, config)

        # Markdown cleaning (URLs, emails, junk, nav menus, whitespace)
        content = clean_markdown(content, config)
        if len(content) != original_len:
            norm_count += 1

        # Check harmful content
        match = check_harmful(content, config)
        if match:
            entry["cleaning_status"] = "removed"
            entry["cleaning_reason"] = f"harmful: {match}"
            removal_stats["harmful"] += 1
            continue

        # Check offensive language & slangs
        match = check_offensive(content, config)
        if match:
            entry["cleaning_status"] = "removed"
            entry["cleaning_reason"] = f"offensive: {match}"
            removal_stats["offensive"] += 1
            continue

        # Remove PII
        content, redactions = remove_pii(content, config)
        pii_total += redactions

        # Check length
        length_reason = check_length(content, config)
        if length_reason:
            entry["cleaning_status"] = "removed"
            entry["cleaning_reason"] = length_reason
            if "too_short" in length_reason:
                removal_stats["too_short"] += 1
            else:
                removal_stats["too_long"] += 1
            continue

        # Document passes all checks
        kept_documents[doc_hash] = content
        entry["pii_redactions"] = redactions

    print(
        f"\n  Documents after per-doc filtering: {len(kept_documents)} / {len(index)}"
    )
    print(f"  Text normalized: {norm_count} documents")
    print(f"  PII redactions applied: {pii_total}")
    for reason, count in removal_stats.items():
        if count:
            print(f"  Removed ({reason}): {count}")

    # --- Phase 2: Deduplication ---
    dedup_config = config.get("deduplication", {})
    duplicates: set[str] = set()

    if dedup_config.get("enabled", False) and len(kept_documents) > 1:
        method = dedup_config.get("method", "minhash")
        print(f"\n--- Phase 2: Deduplication ({method}) ---")

        if method == "minhash":
            duplicates = deduplicate_minhash(kept_documents, config)
        elif method == "bloom":
            duplicates = deduplicate_bloom(kept_documents, config)
        else:
            print(f"  Unknown deduplication method: {method}, skipping.")

        print(f"  Duplicates found: {len(duplicates)}")

        # Mark duplicates in index
        for entry in index:
            if entry["hash"] in duplicates:
                entry["cleaning_status"] = "removed"
                entry["cleaning_reason"] = f"duplicate ({method})"

        # Remove duplicates from kept set
        for dup in duplicates:
            kept_documents.pop(dup, None)
    else:
        print("\n--- Phase 2: Deduplication (skipped) ---")

    # --- Phase 3: Write cleaned documents ---
    print(f"\n--- Phase 3: Writing {len(kept_documents)} cleaned documents ---")

    for entry in index:
        doc_hash = entry["hash"]
        if doc_hash in kept_documents:
            entry["cleaning_status"] = "kept"
            out_path = Path(cleaned_md_dir) / entry["filename"]
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(kept_documents[doc_hash])

    # Save updated index
    save_index(index_path, index)

    # --- Summary ---
    kept = sum(1 for e in index if e.get("cleaning_status") == "kept")
    removed = sum(1 for e in index if e.get("cleaning_status") == "removed")

    print("\n" + "=" * 60)
    print("Cleaning Summary")
    print("=" * 60)
    print(f"  Total documents:   {len(index)}")
    print(f"  Kept:              {kept}")
    print(f"  Removed:           {removed}")
    print(f"    - Harmful:       {removal_stats['harmful']}")
    print(f"    - Offensive:     {removal_stats['offensive']}")
    print(f"    - Too short:     {removal_stats['too_short']}")
    print(f"    - Too long:      {removal_stats['too_long']}")
    print(f"    - Missing file:  {removal_stats['file_missing']}")
    print(f"    - Duplicates:    {len(duplicates)}")
    print(f"  Text normalized:   {norm_count} documents")
    print(f"  PII redactions:    {pii_total}")
    print(f"\n  Cleaned files saved to: {cleaned_md_dir}/")
    print(f"  Index updated: {index_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
