# Fixture provenance

These fixtures were imported unchanged from `aligner-rust-ja` in
[fishaudio/apex-fish-inference PR #84](https://github.com/fishaudio/apex-fish-inference/pull/84),
commit `46879e4b0d8224019f1a6f39e62baa3d599a8de4`.

- `ja_parity_subset.jsonl`: 1,713 word-segmentation cases. Inputs include
  synthetic Unicode/whitespace/control probes, dictionary-derived samples,
  and Japanese Wikipedia excerpts. Outputs target nagisa 0.2.11. All outputs
  were independently regenerated under CPython 3.12.14 and matched on
  2026-09-14.
- `prepro_cases.jsonl`: 407 preprocessing/lowercase/character-type cases
  targeting CPython 3.12 / Unicode 15.0.0, independent of model weights.
- `wikipedia-sources.json`: source article links for all 336 `wiki`, 18
  `wiki_multi`, and one Wikipedia-derived `long` record, indexed by the
  one-based line number in `ja_parity_subset.jsonl`. The other four `long`
  records are synthetic. Sources were recovered from the original corpus's
  cached article extracts; multiple links are retained for duplicate text.

Wikipedia text is credited to the contributors to the linked articles and
is available under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/),
subject to the [Wikimedia Terms of Use](https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use).
The fixture transformation extracts sentences, strips surrounding whitespace,
and sometimes concatenates consecutive sentences. Article histories linked
from each page identify contributors. These text excerpts retain their
separate license; the crate's Apache-2.0 license does not relicense them.

To regenerate word outputs, run `tools/reference.py` as described in the root
README. To add your own cases, supply JSONL records with `text` and optional
`cat`. Preserve article attribution when adding external corpus text.
