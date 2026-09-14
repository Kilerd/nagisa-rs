# nagisa-rs

Pure Rust Japanese word segmentation using the pretrained model from
[**nagisa**](https://github.com/taishi-i/nagisa), the Python/Cython tokenizer
created by **Taishi Ikeda (taishi-i) and contributors**. This project ports
nagisa's word-segmentation inference; the original algorithm, training code,
and pretrained model come from that upstream project.

The compatibility target is `nagisa.tagging(text).words` from
[nagisa 0.2.11](https://pypi.org/project/nagisa/0.2.11/), running under CPython
3.12 (Unicode 15.0.0). It reads the original model files directly and needs
**no Python, DyNet, BLAS, GPU, or native inference library at runtime**.

This implementation started as `aligner-rust-ja` in
[fishaudio/apex-fish-inference PR #84](https://github.com/fishaudio/apex-fish-inference/pull/84),
and is maintained here as a standalone Cargo package. Extraction is pinned to
commit [`46879e4`](https://github.com/fishaudio/apex-fish-inference/tree/46879e4b0d8224019f1a6f39e62baa3d599a8de4/aligner-rust/crates/aligner-rust-ja).
Access to that repository is not needed to build, test, or use this one.

## Scope

- Japanese word segmentation with nagisa's original `nagisa_v001` dictionary and weights.
- CPython 3.12 compatible preprocessing: trailing whitespace removal, NFKC,
  `İ` → `I`, and ASCII space → ideographic space (`U+3000`). Tokens refer to
  this normalized text, not byte offsets in the original input.
- Immutable `JaSegmenter`: `Send + Sync`, with reentrant `words(&self, text)`.
  Load once and share it between threads.
- Pure Rust BiLSTM/CRF inference; `unsafe_code` is forbidden.

POS tagging, training, custom model architectures, `single_word_list`,
`lower=True`, and nagisa's POS-based filtering APIs are not implemented.
Output parity is tested on the bundled corpus; it is not a guarantee for
all possible text, model versions, or floating-point platforms.

## Quick start

The project is tested with Rust **1.97.1**. Use a recent stable Rust toolchain.

Download the original data with a standard-library-only Python script:

```sh
python3 tools/download_model.py
```

This verifies the SHA-256 of the pinned nagisa 0.2.11 source distribution and
extracts `nagisa_v001.dict`, `nagisa_v001.model`, and its license into
`models/nagisa-0.2.11/`. It does not install nagisa or execute its setup code.
The dictionary and model total approximately 46.4 MB uncompressed. If nagisa
0.2.11 is already installed, you can instead use its `nagisa/data/` directory.
Python is only needed for this setup helper or comparison tools, not inference.

Use this checkout as a dependency:

```toml
[dependencies]
nagisa-rs = { path = "../nagisa-rs" }
```

```rust
use nagisa_rs::JaSegmenter;

fn main() -> Result<(), nagisa_rs::JaError> {
    let segmenter = JaSegmenter::from_nagisa_dir("models/nagisa-0.2.11")?;
    let words = segmenter.words("Pythonで簡単に使えるツールです");
    assert_eq!(words, ["Python", "で", "簡単", "に", "使える", "ツール", "です"]);
    Ok(())
}
```

The `segment` example reads one text per line and writes one JSON token array
per line:

```sh
printf '%s\n' 'Pythonで簡単に使えるツールです' |
  cargo run --release --locked --example segment -- models/nagisa-0.2.11
# ["Python","で","簡単","に","使える","ツール","です"]
```

Only `nagisa_v001.dict` and `nagisa_v001.model` are needed. Model dimensions
are fixed to the shipped nagisa 0.2.11 model; `.hp` is not read. Model files
are distributed separately and are not embedded in the crate.

## Performance against Python

| Characters | Python median / p95 (µs) | Rust median / p95 (µs) | Median speedup |
|---:|---:|---:|---:|
| 20 | 362.6 / 408.1 | 89.4 / 106.0 | **4.06×** |
| 100 | 1772.7 / 1900.0 | 459.3 / 516.7 | **3.86×** |
| 400 | 7320.5 / 8370.1 | 1795.4 / 1960.9 | **4.08×** |

Measured on **Apple M4, 32 GB RAM, macOS 26.3 (arm64)** on 2026-09-14, using
Rust 1.97.1 with the repository's release profile, CPython 3.12.14,
nagisa 0.2.11, DyNet38 2.2, and NumPy 2.5.3. See the
[raw samples, inputs, versions, source hashes and model hashes](docs/benchmarks/apple-m4-2026-09-14.json).

Each row uses exactly the same input and original weights in both engines:
the sentence `令和6年4月1日から、東京都渋谷区で新しいサービスが始まります。`
is repeated and truncated to 20, 100, or 400 Unicode characters. Six runs
alternate Rust/Python and Python/Rust order, with separate processes, 10
warmup calls and 400 timed calls per length per run (2,400 samples per cell).
Native thread-count environment variables are set to one. The table reports
pooled median and p95 latency; speedup is Python median / Rust median.

Timing covers `nagisa.tagging(text).words` versus `JaSegmenter::words(text)`,
including preprocessing and token creation. Python's `.words` property
lazily performs segmentation; neither side computes POS tags. Model loading,
process startup, JSON I/O, and destruction of the returned token list are
outside the timed region. Both sides' tokens must match before a result is
reported. The raw `load_ms` fields measure different initialization paths
(Python import/initialization versus Rust segmentation-model loading), so they
are not presented as a startup-speed comparison.

These are warmed, single-thread microbenchmarks on three synthetic inputs,
not an application-throughput or GPU benchmark. Input mix, CPU and build
settings affect results. The original PR's 2.65× forced-aligner throughput
figure covers a separate end-to-end GPU service and does not measure this crate.

Reproduce the comparison with a CPython 3.12 environment:

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r tools/requirements-reference.txt
.venv/bin/python tools/benchmark.py --output results/benchmark.json
```

`uv` is optional; a regular Python 3.12 `venv` and `pip` work too. The benchmark
builds the Rust example in release mode and uses the Python package's own
model directory for both engines.

## Validation

```sh
cargo fmt --all -- --check
cargo clippy --all-targets --locked -- -D warnings
cargo test --release --locked
```

Without external data, four unit tests and the documentation example run;
three model tests and two exhaustive Unicode audits are explicitly marked
**ignored** with setup instructions. The ordinary unit tests include **407**
committed preprocessing/lowercase/character-type cases.

To execute the model tests, including **1,713** reference texts and an
8-thread reentrancy test:

```sh
export NAGISA_RS_MODEL_DIR="$PWD/models/nagisa-0.2.11"
cargo test --release --locked
```

To regenerate the reference outputs with Python and compare against them:

```sh
.venv/bin/python tools/reference.py \
  tests/fixtures/ja_parity_subset.jsonl results/reference.jsonl
NAGISA_RS_FIXTURES="$PWD/results/reference.jsonl" cargo test --release --locked --test parity
```

`NAGISA_RS_MODEL_DIR` must remain set. `NAGISA_RS_FIXTURES` accepts either a
JSONL file or a directory of word fixtures. Each record contains `text`,
`words`, and an optional `cat`; `prepro_cases.jsonl` is reserved for the
separate preprocessing test. See [fixture provenance](tests/fixtures/README.md).

The standalone extraction was verified locally with **0 / 1,713 word
mismatches**, all model tests, and both exhaustive Unicode audits passing.
The original PR additionally reports 23,662 sentences without mismatches;
that larger corpus is not bundled here, so it is a historical result rather
than this repository's default test count.

CI runs on Linux and macOS, downloads the model, verifies the Unicode tables,
runs the scalar Unicode audit, regenerates Python references, and checks the
packaged crate. The slower combining-sequence sweep is available locally.
See [maintenance and Unicode audits](docs/maintenance.md).

## License and acknowledgements

The Rust implementation retains the original crate's **Apache-2.0** license;
see [LICENSE](LICENSE) and [NOTICE](NOTICE). Upstream nagisa is **MIT** licensed;
its copyright and permission notice are preserved in
[licenses/nagisa-MIT.txt](licenses/nagisa-MIT.txt). Pretrained files remain
upstream nagisa assets. Generated Unicode tables carry the
[Unicode data license](licenses/Unicode-3.0.txt). Wikipedia-derived fixture
text has [separate attribution and terms](tests/fixtures/README.md).

Please credit the original [nagisa project](https://github.com/taishi-i/nagisa)
when using this port; this repository does not replace or claim authorship
of nagisa's original model or Python implementation.
