#!/usr/bin/env python3
"""Download Tatoeba bitext-mining test set and save as benchmark.json."""

import json
from datasets import load_dataset

# Language configs: (short_code, full_name, tatoeba_config)
LANGUAGES = [
    ("de", "German", "deu-eng"),
    ("es", "Spanish", "spa-eng"),
    ("fr", "French", "fra-eng"),
    ("ja", "Japanese", "jpn-eng"),
    ("zh", "Chinese", "cmn-eng"),
    ("ar", "Arabic", "ara-eng"),
    ("hi", "Hindi", "hin-eng"),
    ("ru", "Russian", "rus-eng"),
    ("ko", "Korean", "kor-eng"),
]

OUTPUT_FILE = "benchmark.json"


def main():
    benchmark = {
        "metadata": {
            "source": "mteb/tatoeba-bitext-mining",
            "languages": [code for code, _, _ in LANGUAGES],
            "description": "Tatoeba bitext mining pairs for multilingual embedding evaluation",
        },
        "languages": {},
        "pairs": {},
    }

    for code, name, config in LANGUAGES:
        print(f"Loading {name} ({config})...")
        ds = load_dataset("mteb/tatoeba-bitext-mining", config, split="test")

        pairs = []
        for row in ds:
            pairs.append({"en": row["sentence2"], "other": row["sentence1"]})

        benchmark["languages"][code] = {
            "name": name,
            "config": config,
            "count": len(pairs),
        }
        benchmark["pairs"][code] = pairs
        print(f"  {len(pairs)} pairs")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(benchmark, f, ensure_ascii=False, indent=1)

    total = sum(info["count"] for info in benchmark["languages"].values())
    print(f"\nSaved {OUTPUT_FILE}: {total} pairs across {len(LANGUAGES)} languages")


if __name__ == "__main__":
    main()
