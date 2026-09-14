#!/usr/bin/env python3
"""Generate nagisa 0.2.11 references for casing, dictionaries and POS selection."""
import argparse
import json
from pathlib import Path
import sys
import unicodedata

ROOT = Path(__file__).resolve().parents[1]


def tagged(result):
    return {"words": result.words, "postags": result.postags}


def selection(tagger, text, lower, labels):
    return {
        "text": text, "lower": lower, "labels": labels,
        **tagged(tagger.tagging(text, lower=lower)),
        "filtered": tagged(tagger.filter(text, lower=lower, filter_postags=labels)),
        "extracted": tagged(tagger.extract(text, lower=lower, extract_postags=labels)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if (sys.implementation.name, sys.version_info[:2], unicodedata.unidata_version) != ("cpython", (3, 12), "15.0.0"):
        raise SystemExit("use CPython 3.12 / Unicode 15.0.0")
    import nagisa
    if nagisa.__version__ != "0.2.11":
        raise SystemExit("expected nagisa 0.2.11")

    texts = [json.loads(line)["text"] for line in
             (ROOT / "tests/fixtures/ja_parity_subset.jsonl").read_text(encoding="utf-8").split("\n") if line.strip()]
    texts += ["ΑΣB ΑΣ. ΣΑ", "ＡＢＣとİSTANBULとPython", "\t \n", ""]
    lowercase = [{"text": text, **tagged(nagisa.tagging(text, lower=True))} for text in texts]

    examples = ["", "\t \n", "東京都渋谷区で東京都民が暮らす。", "東京都東京都渋谷区",
                "東京大学と京都大学で研究する。", "Pythonで簡単に使えるツールです",
                "PYTHONとPythonとpython。", "ＡＢＣとABCDEF。", "ΑΣBとΑΣ。",
                "猫と(猫)と（猫）です。", "アイスクリームとｱｲｽｸﾘｰﾑ", "A BCとＡ ＢＣ",
                "ABABA", "猫東京大学院へ行く。"]
    # These entries have literal semantics in upstream, including escaped parentheses.
    # Regex metacharacters and empty normalized entries have separate Rust tests.
    dictionaries = []
    for entries in [[], ["", "東"], ["東京", "東京都", "京都", "渋谷区"],
                    ["東京大学", "東京大学院", "京都大学", "大学"],
                    ["Python", "ABC", "ＡＢＣＤ", "ΑΣ", "ΑΣB"],
                    ["(猫)", "アイスクリーム", "Ａ ＢＣ"],
                    ["Ａ ", "ＢＣ", "AB", "ABA", "BAB", "ABA"]]:
        tagger = nagisa.Tagger(single_word_list=entries)
        cases = [selection(tagger, text, lower, ["名詞", "助詞"])
                 for lower in (False, True) for text in examples]
        dictionaries.append({"entries": entries, "cases": cases})

    selections = [selection(nagisa.tagger, text, lower, labels)
                  for text in ["", examples[5], examples[2], "走ることが好きです。"]
                  for lower in (False, True)
                  for labels in [[], ["unknown"], ["名詞", "名詞", "unknown"],
                                 ["助詞", "助動詞", "補助記号"], nagisa.tagger.postags]]
    tokens = json.loads((ROOT / "tests/fixtures/pos_tokens.json").read_text(encoding="utf-8"))
    inputs = [case["words"] for case in tokens] + [["ΑΣ", "B", "ΑΣB", "ΣΑ", "İSTANBUL"]]
    postagging = [{"words": words, "lower": lower, "postags": nagisa.postagging(words, lower=lower)}
                 for lower in (False, True) for words in inputs]
    result = {"lowercase": lowercase, "dictionaries": dictionaries,
              "selections": selections, "postagging": postagging}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"wrote {len(lowercase)} lowercase, {sum(len(group['cases']) for group in dictionaries)} dictionary, "
          f"{len(selections)} POS selection, and {len(postagging)} pre-tokenized cases to {args.output}")


if __name__ == "__main__":
    main()
