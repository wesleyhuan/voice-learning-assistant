"""
convert_hw3.py
--------------
One-time script to convert HW3 output → preloaded_corpus.jsonl

HW3 schema:  {"url": "...", "extracted_at": "...", "content": "..."}
Target:      {"source": "arXiv:XXXXXX", "url": "...", "content": "..."}

Usage:
    python convert_hw3.py
    python convert_hw3.py --input my_corpus.jsonl --output preloaded_corpus.jsonl
"""

import json
import argparse
from urllib.parse import urlparse


def extract_arxiv_id(url: str) -> str:
    """Extract arXiv ID from URL, e.g. https://arxiv.org/abs/2305.10403 → 2305.10403"""
    path = urlparse(url).path.strip("/")
    return path.split("/")[-1]


def convert(input_path: str, output_path: str):
    skipped = 0
    count   = 0

    with open(input_path, encoding="utf-8") as f_in, \
         open(output_path, "w", encoding="utf-8") as f_out:

        for line in f_in:
            line = line.strip()
            if not line:
                continue

            item = json.loads(line)
            content = item.get("content", "").strip()

            if not content:
                skipped += 1
                continue

            arxiv_id = extract_arxiv_id(item.get("url", ""))
            source   = f"arXiv:{arxiv_id}" if arxiv_id else "Unknown"

            out = {
                "source":  source,
                "url":     item.get("url", ""),
                "content": content
            }
            f_out.write(json.dumps(out, ensure_ascii=False) + "\n")
            count += 1

    print(f"✅  Converted {count} documents → {output_path}")
    if skipped:
        print(f"⚠️   Skipped {skipped} empty documents")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert HW3 JSONL to preloaded corpus")
    parser.add_argument("--input",  default="arxiv_corpus.jsonl",  help="HW3 output JSONL")
    parser.add_argument("--output", default="preloaded_corpus.jsonl", help="Target file")
    args = parser.parse_args()
    convert(args.input, args.output)
