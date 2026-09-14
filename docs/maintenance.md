# Maintenance

The library lives at the repository root. Runtime dependencies are
`thiserror`, `flate2` (using the Rust decompressor), and the optional
`ragisa-model` data crate enabled by default. `serde` and
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
are embedded by the same-workspace `ragisa-model` crate in `model/`.
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

When making a crates.io release, publish `ragisa-model` first, then
`ragisa`. The parent pins the model crate's exact version. Keep that
dependency version in sync when changing model data or storage format.
Ordinary clients only add the main crate; Cargo installs the data dependency.

## Test data configuration

| Variable | Data |
|---|---|
| `RAGISA_MODEL_DIR` | Optional original model directory; unset to test bundled loading |
| `RAGISA_FIXTURES` | Word/POS reference JSONL file or directory |
| `RAGISA_UNICODE_REF` | Unicode scalar reference TSV file |
| `RAGISA_UNICODE_SWEEPS` | Directory containing Unicode sequence references |

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
# Also set RAGISA_MODEL_DIR to check original-file loading and all weight bits.
export RAGISA_UNICODE_REF="$PWD/results/unicode/singles.tsv"
export RAGISA_UNICODE_SWEEPS="$PWD/results/unicode"
cargo test --release --locked
```

The scalar audit checks all **1,112,064 Unicode scalar values** (surrogates
excluded). The sequence audit checks **1,112,064 × 85** (scalar, composing
mark) pairs and **922 × 922** combining-mark orderings. Reference generation
can take a few minutes. The full current suite contains nineteen unit tests, eleven integration tests and
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
Character vectors are cached per call by character-ID sequence, matching
nagisa 0.3.0: repeated words reuse the same vector, with no global mutable state.

POS outputs and selection inputs use `PosTag`. Its 24 variants correspond
one-to-one to the original model's numeric IDs; model loading validates the
complete label-to-ID mapping before inference. `Oov` and `UnknownWord` are
separate labels. `as_str()` / `Display` preserve the original text labels,
and `FromStr` rejects unknown strings. JSON examples convert labels only at
the output boundary, outside benchmark timing.

`pos_labels.json` is generated from `nagisa.tagger.postags` alongside the POS
reference fixtures. Tests check all enum labels and their order against this
Python reference, then compare inferred enum values as text against the
unchanged Python word/POS and option fixtures. The selection fixture's
deliberately invalid `unknown` label is checked for a parse error and omitted
by the test adapter; upstream ignores it, while Rust requires typed inputs.

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
RAGISA_FIXTURES="$PWD/results/tagging.jsonl" \
  cargo test --release --locked --test parity --test tagging
```

Unset `RAGISA_MODEL_DIR` to check the bundled model, or set it to compare
the original-file loader. Candidate-feature and pre-tokenized POS
fixtures are committed and regenerated in CI. The 1,717 text fixtures
contain both `words` and `postags`, regenerated against nagisa 0.3.0,
including four upstream decoder regressions. Full POS inference runs without Python
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
ignored; these deliberate differences from nagisa 0.3.0 are documented
in the README. Filtering and extraction run POS inference on the full
sentence before selecting word/tag pairs.

`tests/options.rs` checks casing against the entire text corpus, plus
dictionary boundaries, overlap, normalization, POS selection, pre-segmented
input and shared-instance concurrency. `tools/options_reference.py`
regenerates all expected outputs from the pinned Python reference in CI.

## Benchmark maintenance

Run `tools/benchmark.py` with the pinned reference requirements. It builds
`examples/bench.rs`, checks word/POS equality, records individual timings and
source/model SHA-256 hashes, and prints a Markdown table. Keep benchmarks
separate from builds and tests to avoid competing CPU load. Commit a dated
report under `docs/benchmarks/` when updating README figures, and include
hardware, versions, inputs, warmup, repetitions and exclusions.

Use `--mode words` for `JaSegmenter::words()` and `--mode tagging` for
`Tagger::tagging()`. The Python worker reads `.words` in both modes and also
reads the lazy `.postags` property in tagging mode. Each timed call creates a
fresh result; preprocessing and output construction are included. Model
loading, startup, I/O and returned-result destruction are excluded.

Rust loads the bundled model by default. `--external-model` selects the
Python package's original files instead. Reports include hashes of both
model formats, the benchmark driver, worker, implementation and Cargo files.
Compare those hashes when measuring the same implementation on another CPU.

On Linux, `--cpu N` pins the driver and both workers to an allowed logical CPU
after compilation, before loading the models. Reports record affinity,
cgroup v2 CPU/memory constraints and CPU accounting before/after the runs.
Check the change in `nr_throttled` and `throttled_usec` before interpreting
container results. Affinity does not reserve a physical core; shared-host
contention can still affect timing. Keep infrastructure identifiers out of
public reports.

Historical nagisa 0.2.11 reports retain their original source hashes. New
reports explicitly identify the 0.3.0 NumPy/Cython backend and current Rust
sources; do not present the older timings as measurements of later code.

Loading and inference have different scopes. Python imports the full
upstream package/model; Rust initializes a `JaSegmenter` or `Tagger` from
bundled data. Do not compare their `load_ms` as if they measured identical
operations. These synthetic microbenchmarks do not measure application
throughput.
