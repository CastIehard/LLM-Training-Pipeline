"""
One-time script to add category field to existing JSONL entries.
Looks up the hash in index.json and adds the corresponding category.

DELETE THIS FILE AFTER RUNNING.
"""

import json
from pathlib import Path


def main():
    index_file = "data/index.json"
    jsonl_file = "data/llm_qna.jsonl"
    
    # Load index and create hash -> category mapping
    with open(index_file, "r", encoding="utf-8") as f:
        index_data = json.load(f)
    
    hash_to_category = {entry["hash"]: entry.get("category", "") for entry in index_data}
    print(f"Loaded {len(hash_to_category)} entries from index.json")
    
    # Read existing JSONL
    entries = []
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    
    print(f"Loaded {len(entries)} entries from llm_qna.jsonl")
    
    # Add category to each entry
    updated = 0
    for entry in entries:
        if "category" not in entry or not entry["category"]:
            hash_id = entry.get("hash", "")
            category = hash_to_category.get(hash_id, "")
            entry["category"] = category
            updated += 1
    
    print(f"Updated {updated} entries with category")
    
    # Write back
    with open(jsonl_file, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    
    print(f"Saved to {jsonl_file}")
    print("\nDone! You can delete this script now.")


if __name__ == "__main__":
    main()
