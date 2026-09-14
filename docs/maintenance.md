# Maintenance

The library lives at the repository root. Runtime dependencies are only
`thiserror` and `flate2` (using the Rust decompressor). `serde` and
`serde_json` are development dependencies for examples and fixture readers.

The extracted algorithm uses a three-character window of unigram, bigram and
character-type embeddings, plus dictionary-word features, to form a
200-dimensional input. A bidirectional LSTM with 50 units per direction
feeds a BMES CRF/Viterbi decoder. Only segmentation parameters are loaded;
the separate POS network is skipped. `src/pickle.rs` reads the vocabulary
portion of the original gzip/pickle file, and `src/dynet.rs` reads DyNet text
parameters without executing Python pickle instructions.

## Source history

The source snapshot is `aligner-rust/crates/aligner-rust-ja` at
`fishaudio/apex-fish-inference@46879e4b0d8224019f1a6f39e62baa3d599a8de4`
(PR #84). The standalone extraction changes Cargo metadata, Rust import
names, environment-variable names, examples, documentation, and tooling.
It retains `JaSegmenter` and `JaError` as public names.

| Old variable | Standalone variable |
|---|---|
| `ALIGNER_RUST_JA_NAGISA_DIR` | `NAGISA_RS_MODEL_DIR` |
| `ALIGNER_RUST_JA_FIXTURES` | `NAGISA_RS_FIXTURES` |
| `ALIGNER_RUST_JA_UNICODE_REF` | `NAGISA_RS_UNICODE_REF` |
| `ALIGNER_RUST_JA_UNICODE_SWEEPS` | `NAGISA_RS_UNICODE_SWEEPS` |

The upstream workspace, CUDA build, deployment configuration and service
benchmarks are not dependencies of this project. Nothing is fetched by
Cargo's build script; it only controls which data-dependent tests are enabled.

## Unicode tables

Unicode behavior is pinned to CPython 3.12 / Unicode 15.0.0, independent of
the Rust compiler's Unicode release. `src/nfkd_tables.rs` supplies NFKD,
combining classes and composition pairs. `src/unicode_tables.rs` supplies
lowercase mappings and contextual final-sigma properties. `src/nfkc.rs` and
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
export NAGISA_RS_MODEL_DIR="$PWD/models/nagisa-0.2.11"
export NAGISA_RS_UNICODE_REF="$PWD/results/unicode/singles.tsv"
export NAGISA_RS_UNICODE_SWEEPS="$PWD/results/unicode"
cargo test --release --locked
```

The scalar audit checks all **1,112,064 Unicode scalar values** (surrogates
excluded). The sequence audit checks **1,112,064 × 85** (scalar, composing
mark) pairs and **922 × 922** combining-mark orderings. Reference generation
can take a few minutes. All six unit tests, three integration tests and one
documentation test passed locally with these inputs on 2026-09-14.

U+0130 is replaced with `I` before lowercase feature extraction; its two-code-point
full lowercase mapping is not used. Final sigma requires string context,
which a per-character lowercase operation cannot reproduce.

## Benchmark maintenance

Run `tools/benchmark.py` with the pinned reference requirements. It builds
`examples/bench.rs`, checks token equality, records individual timings and
source/model SHA-256 hashes, and prints a Markdown table. Keep benchmarks
separate from builds and tests to avoid competing CPU load. Commit a dated
report under `docs/benchmarks/` when updating README figures, and include
hardware, versions, inputs, warmup, repetitions and exclusions.

Loading and inference have different scopes. Python loads the full upstream
package/model, while Rust loads only segmentation parameters. Do not compare
those load times as if they measured identical operations, or substitute
forced-aligner service throughput for segmentation speed.
