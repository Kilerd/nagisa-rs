#!/usr/bin/env python3
"""Generate stopword and decoding regressions from nagisa 0.3.0."""
import argparse
import json
from pathlib import Path
import sys
import unicodedata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if (sys.implementation.name, sys.version_info[:2], unicodedata.unidata_version) != (
        "cpython", (3, 12), "15.0.0"
    ):
        raise SystemExit("use CPython 3.12 / Unicode 15.0.0")
    import nagisa
    import nagisa_utils
    if nagisa.__version__ != "0.3.0":
        raise SystemExit("expected nagisa 0.3.0")
    inputs = [("", []), ("東", [0]), ("東", [1]), ("東", [2]), ("東", [3]),
              ("東京", [0, 3]), ("東京", [1, 3]), ("東京", [0, 1]),
              ("東京へ", [0, 2, 3]), ("東京へ", [0, 3, 1]),
              ("あいうえお", [0, 1, 3, 0, 1]), ("東京", [4, 5])]
    bmes = [{"text": text, "tags": tags,
             "words": nagisa_utils.segmenter_for_bmes(text, tags)} for text, tags in inputs]
    regressions = []
    for text in ["いい写真やちゃ", "http://www.", "ちなみに公式ホームページ(http://www.",
                 "互いにライフを1000ポイント払わないと通常召喚できない。"]:
        tagged = nagisa.tagging(text)
        regressions.append({"cat": "upstream_030", "text": text,
                            "words": tagged.words, "postags": tagged.postags})
    result = {"nagisa": nagisa.__version__, "stopwords": nagisa.stopwords,
              "bmes": bmes, "regressions": regressions}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(nagisa.stopwords)} stopwords, {len(bmes)} BMES cases and "
          f"{len(regressions)} upstream regressions to {args.output}")


if __name__ == "__main__":
    main()
