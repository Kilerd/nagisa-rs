#!/usr/bin/env python3
"""Regenerate word and POS fixtures from JSONL texts using the original Python nagisa."""
import argparse
import json
from pathlib import Path
import sys
import unicodedata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help='JSONL records containing "text" and optional "cat"')
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if sys.implementation.name != "cpython" or sys.version_info[:2] != (3, 12):
        raise SystemExit("use CPython 3.12 to match the port's Unicode 15.0.0 tables")
    if unicodedata.unidata_version != "15.0.0":
        raise SystemExit("expected Unicode 15.0.0")
    import nagisa
    if nagisa.__version__ != "0.2.11":
        raise SystemExit("expected nagisa 0.2.11")
    # Read before opening output so in-place regeneration is safe.
    with args.input.open(encoding="utf-8") as source:
        cases = [json.loads(line) for line in source if line.strip()]
    if not cases:
        raise SystemExit("no input cases")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output:
        for case in cases:
            tagged = nagisa.tagging(case["text"])
            result = {"cat": case.get("cat", "custom"), "text": case["text"],
                      "words": tagged.words, "postags": tagged.postags}
            output.write(json.dumps(result, ensure_ascii=False) + "\n")
    print(f"wrote {len(cases)} references to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
