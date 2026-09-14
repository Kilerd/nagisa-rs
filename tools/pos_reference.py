#!/usr/bin/env python3
"""Generate model-free candidate-tag features and pre-tokenized POS references."""
import argparse
import json
from pathlib import Path
import sys
import unicodedata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, nargs="?", default=Path("tests/fixtures"))
    args = parser.parse_args()
    if (sys.implementation.name, sys.version_info[:2], unicodedata.unidata_version) != ("cpython", (3, 12), "15.0.0"):
        raise SystemExit("use CPython 3.12 / Unicode 15.0.0")
    import nagisa
    if nagisa.__version__ != "0.3.0":
        raise SystemExit("expected nagisa 0.3.0")
    sequences = {tuple(tags) for tags in nagisa.tagger._word2postags.values()}
    # Also exercise collisions, duplicates, removal/reinsertion and resizing.
    sequences.update((a, b, c) for a in range(24) for b in range(24) for c in (0, 2, 9, 17, 23))
    sequences.update(((), (0,), tuple(range(24)), tuple(reversed(range(24)))))
    cases = []
    for tags in sorted(sequences):
        for noun in (False, True):
            actual = set(tags)
            if noun:
                actual.discard(0)
                actual.add(2)
            cases.append({"tags": tags, "noun": noun, "ordered": list(actual)})
    inputs = [
        [], ["Python", "で", "簡単", "に", "使える", "ツール", "です"],
        ["Python", "PYTHON", "python", "OpenAI", "東京", "123", "🙂"],
        [" ", "　", "ＡＢＣ", "①", "ﾊﾝｶｸ", "İstanbul"],
        ["今日", "は", "走る", "。"], ["走る", "こと", "が", "好き", "です"],
        ["2026", "年", "9", "月", "14", "日"], ["x\x01y", "ΑΣ", "ᾳ", "\U00011f02"],
        ["東京\t", "猫\n", "日本語", "\u00bd", "abc_", "\u0345"],
    ]
    tokens = [{"words": words, "postags": nagisa.postagging(words)} for words in inputs]
    args.output.mkdir(parents=True, exist_ok=True)
    for name, data in (("pos_candidates.json", cases), ("pos_tokens.json", tokens),
                       ("pos_labels.json", nagisa.tagger.postags)):
        (args.output / name).write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
        print(f"wrote {len(data)} references to {args.output / name}")


if __name__ == "__main__":
    main()
