# ragisa

Pure Rust Japanese word segmentation and POS tagging using the pretrained model from
[**nagisa**](https://github.com/taishi-i/nagisa), the Python/Cython tokenizer
created by **Taishi Ikeda (taishi-i) and contributors**. This project ports
nagisa's inference APIs; the original algorithm, training code, and pretrained
model come from that upstream project. The checklist below tracks which
upstream features are available here.

The compatibility target is the default word and POS output of `nagisa.tagging(text)` from
[nagisa 0.2.11](https://pypi.org/project/nagisa/0.2.11/), running under CPython
3.12 (Unicode 15.0.0). The pretrained model is **bundled by default**: no
separate model download, model path, Python, DyNet, BLAS, GPU, or native
inference library is needed at runtime.

## Upstream feature checklist

Checked items are implemented and tested against **nagisa 0.2.11 / CPython
3.12**. Unchecked items are not yet implemented; this is not a claim of full
API compatibility with every nagisa release.

- [x] **Word segmentation** — equivalent to `nagisa.wakati(text)` and
  `nagisa.tagging(text).words`, via `JaSegmenter::words()` or `Tagger::words()`.
- [x] **Word segmentation + POS tagging** — `Tagger::tagging()` returns
  `TaggedText { words, postags }`, targeting both fields of `nagisa.tagging(text)`.
- [x] **POS tagging of pre-segmented words** — `Tagger::postagging()` targets
  `nagisa.postagging(words)` / `nagisa.decode(words)`, including normalization
  and the default noun heuristic. Tokens that become empty return a Rust error.
- [x] **Available POS labels** — `Tagger::postags()` exposes the original
  model's 24 labels in numeric ID order, including `oov`.
- [x] **Original pretrained model** — the bundled dictionary and lossless
  f32 weights are available through `new()`. `from_nagisa_dir()` also loads
  the original `nagisa_v001.dict` and `nagisa_v001.model` without conversion.
- [x] **Default preprocessing** — CPython 3.12 whitespace removal, NFKC,
  `İ` → `I`, ASCII space → `U+3000`, and contextual lowercase features.
- [x] **Lowercase output option** — `TextOptions { lower: true }` in the
  `*_with_options()` methods targets the upstream `lower=True` argument.
- [x] **User dictionary / forced single words** — `with_single_word_list()`
  targets `Tagger(single_word_list=...)`, with literal matching as described below.
- [x] **Filter by POS** — `Tagger::filter()` targets
  `nagisa.filter(..., filter_postags=...)`.
- [x] **Extract by POS** — `Tagger::extract()` targets
  `nagisa.extract(..., extract_postags=...)`.
- [ ] **Model training** — `nagisa.fit(...)` and training-data workflows.
- [ ] **Custom trained models and hyperparameters** — custom `vocabs`, `params`
  and `hp` configurations; only the original 0.2.11 architecture is supported.

Both Rust types are immutable, `Send + Sync`, and reentrant. Load once and
share between threads. `JaSegmenter` retains only segmentation data; `Tagger`
additionally retains the POS dictionary and neural networks. Inference is pure
Rust and `unsafe_code` is forbidden. Tokens refer to normalized text rather
than byte offsets in the original input. Parity is tested on the bundled
corpus, not guaranteed for all text or floating-point platforms.

## Quick start

The project is tested with Rust **1.97.1**. Use a recent stable Rust toolchain.

Use the repository as a dependency until the first crates.io release:

```toml
[dependencies]
ragisa = { git = "https://github.com/Kilerd/ragisa" }
```

```rust
use ragisa::JaSegmenter;

fn main() -> Result<(), ragisa::JaError> {
    let segmenter = JaSegmenter::new()?;
    let words = segmenter.words("Pythonで簡単に使えるツールです");
    assert_eq!(words, ["Python", "で", "簡単", "に", "使える", "ツール", "です"]);
    Ok(())
}
```

For word segmentation **and** POS tags:

```rust
use ragisa::Tagger;

fn main() -> Result<(), ragisa::JaError> {
    let tagger = Tagger::new()?;
    let result = tagger.tagging("Pythonで簡単に使えるツールです");
    assert_eq!(result.postags, ["名詞", "助詞", "形状詞", "助動詞", "動詞", "名詞", "助動詞"]);
    assert_eq!(tagger.postagging(&result.words)?, result.postags);
    Ok(())
}
```

`postagging()` returns `JaError::EmptyToken { index }` for an input token that
normalizes to an empty string, where Python's character encoder raises an
exception. An empty token list and an empty text are supported. A single
ASCII or ideographic space is preserved when labeling pre-segmented words.

The `segment` example reads one text per line and writes one JSON token array
per line:

```sh
printf '%s\n' 'Pythonで簡単に使えるツールです' |
  cargo run --release --locked --example segment
# ["Python","で","簡単","に","使える","ツール","です"]
```

To emit both `words` and `postags` as JSON, use `--example tag` with the same
arguments and line-based input.

The default `bundled-model` feature embeds the dictionary and weights in the
application. Cargo automatically installs the `ragisa-model` data dependency;
each crate stays below crates.io's default 10 MiB upload limit. All assets
are committed in this repository and load entirely in memory. The original
MIT license and checksums are included. See [bundled data](data/README.md).

### Optional external model files

To manage model files yourself, disable default features and use
`JaSegmenter::from_nagisa_dir(dir)` or `Tagger::from_nagisa_dir(dir)`. The
external loaders are also available with default features enabled.

```toml
[dependencies]
ragisa = { git = "https://github.com/Kilerd/ragisa", default-features = false }
```

`python3 tools/download_model.py` is an optional maintainer/reference helper
that verifies the pinned source archive's SHA-256 and extracts the original
files to `models/nagisa-0.2.11/`. Only `nagisa_v001.dict` and
`nagisa_v001.model` are needed by the external loaders. Model dimensions are
fixed to nagisa 0.2.11's shipped architecture; `.hp` is not read. Passing a
data-directory argument to either CLI example selects the external loader.

### Casing, user dictionaries and POS selection

```rust
use ragisa::{Tagger, TextOptions};

fn main() -> Result<(), ragisa::JaError> {
    let tagger = Tagger::new()?
        .with_single_word_list(["東京大学", "C++"]);
    let lower = TextOptions { lower: true };
    assert_eq!(tagger.words_with_options("Python", lower), ["python"]);
    assert_eq!(tagger.words("東京大学"), ["東京大学"]);

    let nouns = tagger.extract("東京大学でPythonを学ぶ。", &["名詞"]);
    let content = tagger.filter("東京大学でPythonを学ぶ。", &["助詞", "補助記号"]);
    println!("{nouns:?}\n{content:?}");
    Ok(())
}
```

`TextOptions` defaults to `lower: false`. Both types support
`words_with_options()`. `Tagger` also provides `tagging_with_options()`,
`postagging_with_options()`, `filter_with_options()` and
`extract_with_options()`. POS selection keeps the labels inferred with full
sentence context and preserves word/tag alignment. Empty or unknown label
lists remove nothing in `filter()` and retain nothing in `extract()`.

Both types support the consuming `with_single_word_list()` builder; calling
it again replaces the dictionary. Matching is case-sensitive on normalized
text, before optional lowercasing, and takes the longest non-overlapping
match at each position from left to right. Entries of at most one character
before normalization are ignored, as in nagisa 0.2.11.

**Dictionary compatibility:** Rust treats every entry literally, including
regex metacharacters such as `+`, `.`, `[` and `|`. Upstream 0.2.11 only
escapes parentheses before building a regex, so those metacharacters can
behave differently. Rust also ignores entries that normalize to empty text.
The dictionary parity fixtures cover literal entries, normalization,
overlaps, boundaries, casing, and combinations with POS selection.

Training and arbitrary custom models remain unimplemented. Training needs
backpropagation, optimizers and data workflows; custom configurations need
dynamic network dimensions and vocabulary/label handling beyond the shipped
architecture.

## Performance against Python

### Word segmentation

These figures measure word segmentation via `JaSegmenter::words()`.
They do not measure `Tagger::tagging()`. The raw report records the source
hashes of the measured implementation at
[`ccb6232`](https://github.com/Kilerd/ragisa/commit/ccb6232abebd726b14d2c1edd0eb96c039f3a807),
before bundled loading was added; model loading is excluded from these timings.

| Characters | Python median / p95 (µs) | Rust median / p95 (µs) | Median speedup |
|---:|---:|---:|---:|
| 20 | 367.6 / 415.0 | 88.8 / 108.7 | **4.14×** |
| 100 | 1806.9 / 1913.3 | 457.0 / 612.6 | **3.95×** |
| 400 | 7400.7 / 9113.0 | 1796.5 / 1980.2 | **4.12×** |

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
settings affect results.

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
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo test --workspace --release --locked
```

With default features, all eleven inference tests and both executable
documentation examples run using bundled data, without downloading anything.
Only the original-file bitwise comparison and two exhaustive Unicode audits
are **ignored** until their reference data is supplied. Unit tests include **407**
committed preprocessing/lowercase/character-type cases and **6,402** Python
reference cases for candidate-POS set ordering.

The bundled tests cover **1,713** reference texts with both word and POS
outputs, **1,717** lowercase cases, **196** dictionary cases, **40** POS
selection cases, **20** pre-segmented casing cases, and 8-thread reentrancy.
To check the original-file loader as well as every bundled weight's f32 bits:

```sh
python3 tools/download_model.py
python3 tools/bundle_model.py --check
export RAGISA_MODEL_DIR="$PWD/models/nagisa-0.2.11"
cargo test --release --locked
```

To regenerate both word and POS reference outputs with Python and compare against them:

```sh
.venv/bin/python tools/reference.py \
  tests/fixtures/ja_parity_subset.jsonl results/reference.jsonl
RAGISA_FIXTURES="$PWD/results/reference.jsonl" cargo test --release --locked --test parity --test tagging
```

Unset `RAGISA_MODEL_DIR` to test bundled loading. `RAGISA_FIXTURES` accepts either a
JSONL file or a directory of word fixtures. Each record contains `text`,
`words`, `postags`, and an optional `cat`; `prepro_cases.jsonl` is reserved for the
separate preprocessing test. See [fixture provenance](tests/fixtures/README.md).

The current implementation was verified locally with **0 / 1,713 word
mismatches and 0 / 1,713 POS mismatches**, including both types' 8-thread tests.
All **27 tests** pass locally with the original model and exhaustive Unicode audit data enabled.

CI runs on Linux and macOS, tests bundled loading before any model download,
then verifies the original data, all f32 weight bits and the Unicode tables,
runs the scalar Unicode audit, regenerates word/POS, inference-option and candidate-feature
Python references, and checks the
packaged crates. It also checks the external-only build and offline use of
the distributed packages. The slower combining-sequence sweep is available locally.
See [maintenance and Unicode audits](docs/maintenance.md).

## License and acknowledgements

The Rust implementation is licensed under **Apache-2.0**;
see [LICENSE](LICENSE) and [NOTICE](NOTICE). Upstream nagisa is **MIT** licensed;
its copyright and permission notice are preserved in
[licenses/nagisa-MIT.txt](licenses/nagisa-MIT.txt). Pretrained files remain
upstream nagisa assets. Generated Unicode tables carry the
[Unicode data license](licenses/Unicode-3.0.txt). CPython set behavior is
attributed with its [PSF license](licenses/Python-2.0.txt). Wikipedia-derived fixture
text has [separate attribution and terms](tests/fixtures/README.md).

Please credit the original [nagisa project](https://github.com/taishi-i/nagisa)
when using this port; this repository does not replace or claim authorship
of nagisa's original model or Python implementation.
