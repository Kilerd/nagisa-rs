# Maintenance

The library lives at the repository root. Runtime dependencies are
`thiserror`, `flate2` (using the Rust decompressor), and the optional
`nagisa-rs-model` data crate enabled by default. `serde` and
`serde_json` are development dependencies for examples and fixture readers.

The segmentation algorithm uses a three-character window of unigram, bigram and
character-type embeddings, plus dictionary-word features, to form a
200-dimensional input. A bidirectional LSTM with 50 units per direction
feeds a BMES CRF/Viterbi decoder. `JaSegmenter` retains only segmentation parameters. `Tagger` adds the
POS candidate dictionary, tag embeddings, character BiLSTM, token BiLSTM
and 24-label output projection, sharing the segmentation embeddings. `src/pickle.rs` reads the vocabulary
portion of the original gzip/pickle file, and `src/dynet.rs` reads DyNet text
parameters and the bundled binary representation. Neither loader executes
Python pickle instructions.

## Bundled model and release packaging

The default `bundled-model` feature exposes `JaSegmenter::new()` and
`Tagger::new()`. The original dictionary is embedded from `data/`; weights
are embedded by the same-workspace `nagisa-rs-model` crate in `model/`.
Both loaders decode in memory without runtime paths, caches, extraction,
network access, build-time downloads or Python. Disabling default features
removes the embedded data from the executable and the model dependency;
the main crate's source archive still contains its dictionary.

The weights use lossless little-endian f32 storage; every bit is checked
against the original text loader. [Data documentation](../data/README.md)
describes the format, licenses, pinned hashes and regeneration commands.
Use `python3 tools/bundle_model.py --check` after fetching the original model
to verify that the committed assets have not drifted.

Each compressed package must remain below 10 MiB, crates.io's default
upload limit. Verify both archives and an offline consumer with:

```sh
cargo package --workspace --locked
python3 tools/check_package.py
```

The consumer builds from the normalized Cargo archives with no network,
then runs from an empty directory after the extracted package sources are
deleted. The check also enforces package sizes and bundled license files.

When making a crates.io release, publish `nagisa-rs-model` first, then
`nagisa-rs`. The parent pins the model crate's exact version. Keep that
dependency version in sync when changing model data or storage format.
Ordinary clients only add the main crate; Cargo installs the data dependency.

## Test data configuration

| Variable | Data |
|---|---|
| `NAGISA_RS_MODEL_DIR` | Optional original model directory; unset to test bundled loading |
| `NAGISA_RS_FIXTURES` | Word/POS reference JSONL file or directory |
| `NAGISA_RS_UNICODE_REF` | Unicode scalar reference TSV file |
| `NAGISA_RS_UNICODE_SWEEPS` | Directory containing Unicode sequence references |

Cargo's build script controls which data-dependent tests are enabled.
It does not fetch data.

## Unicode tables

Unicode behavior is pinned to CPython 3.12 / Unicode 15.0.0, independent of
the Rust compiler's Unicode release. `src/nfkd_tables.rs` supplies NFKD,
combining classes and composition pairs. `src/unicode_tables.rs` supplies
lowercase mappings and contextual final-sigma properties.
`src/pos_unicode_tables.rs` pins Python's `str.isalnum()` for the default
noun heuristic; Rust's current Unicode tables are not substituted. `src/nfkc.rs` and
`src/prepro.rs` consume these tables.

Use CPython 3.12 (no third-party Python packages needed):

```sh
.venv/bin/python tools/unicode.py --check
.venv/bin/python tools/unicode.py --write
cargo fmt --all
```

`--check` compares every stored table value against the interpreter;
`--write` regenerates them. Both reject incompatible Python or Unicode
versions. A Unicode-version upgrade is a compatibility change and requires
updating the reference version and fixtures together, then rerunning parity.

Generate the exhaustive audit files and run all tests:

```sh
.venv/bin/python tools/unicode.py --check --references results/unicode --sweeps
# Also set NAGISA_RS_MODEL_DIR to check original-file loading and all weight bits.
export NAGISA_RS_UNICODE_REF="$PWD/results/unicode/singles.tsv"
export NAGISA_RS_UNICODE_SWEEPS="$PWD/results/unicode"
cargo test --release --locked
```

The scalar audit checks all **1,112,064 Unicode scalar values** (surrogates
excluded). The sequence audit checks **1,112,064 × 85** (scalar, composing
mark) pairs and **922 × 922** combining-mark orderings. Reference generation
can take a few minutes. The full current suite contains fourteen unit tests, eleven integration tests and
two documentation tests. Both `JaSegmenter` and `Tagger` have model parity
and 8-thread reentrancy coverage.

U+0130 is replaced with `I` before lowercase feature extraction; its two-code-point
full lowercase mapping is not used. Final sigma requires string context,
which a per-character lowercase operation cannot reproduce.

## POS inference and reference data

`Tagger::tagging()` segments and labels normalized text. `Tagger::postagging()`
normalizes supplied words, preserves single-space tokens, and returns an
`EmptyToken` error for empty normalized words. By default POS features use
the original case of each token, independently of segmentation's lowercase
features. `TextOptions { lower: true }` also lowercases POS input tokens.

The original `char_seq_model.transduce(chars)[-1]` selects the last original
character position. Its forward state covers the whole word; its backward
state has consumed only the last character. Using the final reverse state
instead would implement a different encoder.

Candidate tag IDs come from the complete `word2postags` dictionary. nagisa
builds a Python integer set, removes OOV and adds the noun label for alphanumeric
tokens, and sums tag embeddings in set iteration order. `src/pos.rs` reproduces
that order for the supported 24-label model, including small-table collisions
and resize behavior. This avoids differences in f32 embedding sums. Unknown
word IDs and candidate tag ID 0 contribute zero vectors in the POS network.

Regenerate fixtures using the pinned reference environment:

```sh
.venv/bin/python tools/reference.py \
  tests/fixtures/ja_parity_subset.jsonl results/tagging.jsonl
.venv/bin/python tools/pos_reference.py results/pos
.venv/bin/python tools/options_reference.py results/options.json
cmp tests/fixtures/options.json results/options.json
NAGISA_RS_FIXTURES="$PWD/results/tagging.jsonl" \
  cargo test --release --locked --test parity --test tagging
```

Unset `NAGISA_RS_MODEL_DIR` to check the bundled model, or set it to compare
the original-file loader. Candidate-feature and pre-tokenized POS
fixtures are committed and regenerated in CI. The 1,713 text fixtures now
contain both `words` and `postags`; the original inputs and word outputs
are preserved. Full POS inference, like segmentation, runs without Python
or native numerical libraries.

## Casing and dictionary behavior

Segmentation lowercases the full normalized sentence before cutting words,
so contextual final sigma must not be computed separately on each output
token. `postagging_with_options()` instead lowercases each supplied token,
matching upstream's pre-segmented API.

`src/dictionary.rs` uses a literal trie to choose longest, leftmost,
non-overlapping matches on the normalized text with its original casing.
It forces BMES tags and repairs adjacent boundaries before words are cut.
The dictionary builder replaces existing entries and inference only reads
the trie. Regex syntax is not interpreted and empty normalized entries are
ignored; these deliberate differences from nagisa 0.2.11 are documented
in the README. Filtering and extraction run POS inference on the full
sentence before selecting word/tag pairs.

`tests/options.rs` checks casing against the entire text corpus, plus
dictionary boundaries, overlap, normalization, POS selection, pre-segmented
input and shared-instance concurrency. `tools/options_reference.py`
regenerates all expected outputs from the pinned Python reference in CI.

## Benchmark maintenance

Run `tools/benchmark.py` with the pinned reference requirements. It builds
`examples/bench.rs`, checks token equality, records individual timings and
source/model SHA-256 hashes, and prints a Markdown table. Keep benchmarks
separate from builds and tests to avoid competing CPU load. Commit a dated
report under `docs/benchmarks/` when updating README figures, and include
hardware, versions, inputs, warmup, repetitions and exclusions.

The committed benchmark and current benchmark tool measure word segmentation
only. Loading and inference have different scopes. Python loads the full
upstream package/model, while the benchmark's Rust `JaSegmenter` loads only
segmentation parameters. `Tagger` also loads POS data. Do not compare
those load times as if they measured identical operations. Segmentation
microbenchmarks do not measure application throughput.
