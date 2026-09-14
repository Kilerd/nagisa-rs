# ragisa

Pure Rust Japanese word segmentation and POS tagging using the pretrained model from
[**nagisa**](https://github.com/taishi-i/nagisa), the Python/Cython tokenizer
created by **Taishi Ikeda (taishi-i) and contributors**. This project ports
nagisa's inference APIs; the original algorithm, training code, and pretrained
model come from that upstream project. The checklist below tracks which
upstream features are available here.

The compatibility target is the default word and POS output of `nagisa.tagging(text)` from
[nagisa 0.3.0](https://pypi.org/project/nagisa/0.3.0/), running under CPython
3.12 (Unicode 15.0.0). The pretrained model is **bundled by default**: no
separate model download, model path, Python, DyNet, BLAS, GPU, or native
inference library is needed at runtime.

## Upstream feature checklist

Checked items are implemented and tested against **nagisa 0.3.0 / CPython
3.12**. Unchecked items are not yet implemented; this is not a claim of full
API compatibility with every nagisa release.

- [x] **Word segmentation** — equivalent to `nagisa.wakati(text)` and
  `nagisa.tagging(text).words`, via `JaSegmenter::words()` or `Tagger::words()`.
- [x] **Word segmentation + POS tagging** — `Tagger::tagging()` returns
  `TaggedText { words, postags: Vec<PosTag> }`, targeting both fields of
  `nagisa.tagging(text)` with typed POS labels.
- [x] **POS tagging of pre-segmented words** — `Tagger::postagging()` targets
  `nagisa.postagging(words)` / `nagisa.decode(words)`, including normalization
  and the default noun heuristic. Tokens that become empty return a Rust error.
- [x] **Available POS labels** — `Tagger::postags()` exposes the original
  model's 24 `PosTag` variants in numeric ID order, including `PosTag::Oov`.
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
- [x] **Stopword list** — `STOPWORDS` contains all 135 entries from `nagisa.stopwords`,
  in upstream order. Stopword removal is opt-in.
- [x] **Concurrent inference** — share one immutable `JaSegmenter` or `Tagger`
  across threads; correctness and batch throughput are checked at 1, 2, 4 and
  8 threads, including comparison with nagisa on CPython 3.14t.
- [ ] **Model training** — `nagisa.fit(...)` and training-data workflows.
- [ ] **Custom trained models and hyperparameters** — custom `vocabs`, `params`
  and `hp` configurations; only the original 0.3.0 architecture is supported.

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
use ragisa::{PosTag, Tagger};

fn main() -> Result<(), ragisa::JaError> {
    let tagger = Tagger::new()?;
    let result = tagger.tagging("Pythonで簡単に使えるツールです");
    assert_eq!(result.postags, [
        PosTag::Noun,
        PosTag::Particle,
        PosTag::AdjectivalNoun,
        PosTag::AuxiliaryVerb,
        PosTag::Verb,
        PosTag::Noun,
        PosTag::AuxiliaryVerb,
    ]);
    assert_eq!(tagger.postagging(&result.words)?, result.postags);
    Ok(())
}
```

POS APIs use the `Copy + Eq + Hash` enum `PosTag`: `TaggedText::postags` and
`postagging()` return `Vec<PosTag>`, while `filter()` and `extract()` accept
`&[PosTag]`. Use variants such as `PosTag::Noun` in comparisons and `match`
expressions. `PosTag::ALL` lists all 24 labels in original model ID order.

`PosTag::Noun.as_str()` and `PosTag::Noun.to_string()` produce `"名詞"`.
`"助詞".parse::<PosTag>()` returns `Ok(PosTag::Particle)`; unrecognized labels
return `ParsePosTagError`. `PosTag::Oov` (`oov`) and `PosTag::UnknownWord`
(`未知語`) remain distinct. The JSON examples retain the original label strings.

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
files to `models/nagisa-0.3.0/`. Only `nagisa_v001.dict` and
`nagisa_v001.model` are needed by the external loaders. Model dimensions are
fixed to nagisa 0.3.0's shipped architecture; `.hp` is not read. Passing a
data-directory argument to either CLI example selects the external loader.

### Casing, user dictionaries and POS selection

```rust
use ragisa::{PosTag, Tagger, TextOptions};

fn main() -> Result<(), ragisa::JaError> {
    let tagger = Tagger::new()?
        .with_single_word_list(["東京大学", "C++"]);
    let lower = TextOptions { lower: true };
    assert_eq!(tagger.words_with_options("Python", lower), ["python"]);
    assert_eq!(tagger.words("東京大学"), ["東京大学"]);

    let nouns = tagger.extract("東京大学でPythonを学ぶ。", &[PosTag::Noun]);
    let content = tagger.filter(
        "東京大学でPythonを学ぶ。",
        &[PosTag::Particle, PosTag::SupplementarySymbol],
    );
    println!("{nouns:?}\n{content:?}");
    Ok(())
}
```

`TextOptions` defaults to `lower: false`. Both types support
`words_with_options()`. `Tagger` also provides `tagging_with_options()`,
`postagging_with_options()`, `filter_with_options()` and
`extract_with_options()`. POS selection keeps the labels inferred with full
sentence context and preserves word/tag alignment. Empty label lists remove
nothing in `filter()` and retain nothing in `extract()`. Unknown label strings
must be handled when parsing into `PosTag`; they cannot enter the typed API.

Both types support the consuming `with_single_word_list()` builder; calling
it again replaces the dictionary. Matching is case-sensitive on normalized
text, before optional lowercasing, and takes the longest non-overlapping
match at each position from left to right. Entries of at most one character
before normalization are ignored, as in nagisa 0.3.0.

**Dictionary compatibility:** Rust treats every entry literally, including
regex metacharacters such as `+`, `.`, `[` and `|`. Upstream 0.3.0 only
escapes parentheses before building a regex, so those metacharacters can
behave differently. Rust also ignores entries that normalize to empty text.
The dictionary parity fixtures cover literal entries, normalization,
overlaps, boundaries, casing, and combinations with POS selection.

### Stopwords

`STOPWORDS` is the original 135-entry list from `nagisa.stopwords`. For example:

```rust
use ragisa::{JaSegmenter, STOPWORDS};

fn main() -> Result<(), ragisa::JaError> {
    let mut words = JaSegmenter::new()?.words("このツールはPythonで使えます。");
    words.retain(|word| !STOPWORDS.contains(&word.as_str()));
    Ok(())
}
```

See [nagisa 0.3.0 compatibility and implementation changes](docs/nagisa-0.3.0.md)
for the upstream decoder fix, backend changes and verified feature coverage.

Training and arbitrary custom models remain unimplemented. Training needs
backpropagation, optimizers and data workflows; custom configurations need
dynamic network dimensions and vocabulary/label handling beyond the shipped
architecture.

## Performance against Python

Measured on **2026-09-14** with the default bundled model, Rust **1.97.1**
using the repository's release profile and no extra `RUSTFLAGS`, CPython
**3.12.14**, nagisa **0.3.0**, DyNet38 **2.2**, and NumPy **2.5.3**.
Both platforms used identical implementation, benchmark and model hashes.
Speedup compares Python and Rust on the same CPU. Python workers explicitly
verify the **NumPy/Cython backend** and record that DyNet was not imported;
DyNet38 is installed because upstream still declares it as a dependency.

### Apple M4

**32 GB RAM, macOS 26.3 (arm64)**, with default OS scheduling.

| Task | Characters | Python median / p95 (µs) | Rust median / p95 (µs) | Median speedup |
|---|---:|---:|---:|---:|
| Words | 20 | 139.7 / 159.1 | 88.9 / 105.7 | **1.57×** |
| Words | 100 | 685.8 / 732.7 | 449.3 / 493.8 | **1.53×** |
| Words | 400 | 2724.9 / 2920.7 | 1772.2 / 1854.3 | **1.54×** |
| Words + POS | 20 | 220.5 / 233.7 | 152.0 / 163.9 | **1.45×** |
| Words + POS | 100 | 961.7 / 1020.0 | 690.7 / 743.6 | **1.39×** |
| Words + POS | 400 | 3678.8 / 3810.5 | 2640.7 / 2784.2 | **1.39×** |

Raw inputs, samples, versions and hashes: [words](docs/benchmarks/apple-m4-words-nagisa-0.3.0-2026-09-14.json), [words + POS](docs/benchmarks/apple-m4-tagging-nagisa-0.3.0-2026-09-14.json).

### AMD EPYC 9V74

**Linux 6.8.0 (x86_64)**, in a container limited to **4 vCPU / 8 GiB**.
Both workers were pinned to logical CPU 16 on a shared host. No CPU quota
throttling occurred during either measurement (both `nr_throttled` and
`throttled_usec` deltas were zero). Affinity does not reserve a core.

| Task | Characters | Python median / p95 (µs) | Rust median / p95 (µs) | Median speedup |
|---|---:|---:|---:|---:|
| Words | 20 | 233.6 / 244.0 | 231.3 / 239.0 | **1.01×** |
| Words | 100 | 1154.7 / 1224.3 | 1169.0 / 1197.4 | **0.99×** |
| Words | 400 | 5904.4 / 5974.7 | 4684.2 / 4739.4 | **1.26×** |
| Words + POS | 20 | 421.6 / 437.8 | 394.1 / 401.7 | **1.07×** |
| Words + POS | 100 | 1888.8 / 1960.2 | 1796.2 / 1827.4 | **1.05×** |
| Words + POS | 400 | 8288.5 / 8483.4 | 6952.1 / 7025.8 | **1.19×** |

Raw inputs, samples, versions and hashes: [words](docs/benchmarks/amd-epyc-9v74-words-nagisa-0.3.0-2026-09-14.json), [words + POS](docs/benchmarks/amd-epyc-9v74-tagging-nagisa-0.3.0-2026-09-14.json).

### Method and reproduction

Each row uses the sentence
`令和6年4月1日から、東京都渋谷区で新しいサービスが始まります。`,
repeated and truncated to 20, 100, or 400 Unicode characters. Each task has
six runs alternating Rust/Python and Python/Rust order, using separate
processes, 10 warmup calls and 400 timed calls per length per run
(**2,400 samples per cell**). Native thread-count environment variables are
set to one. Tables report pooled median and p95 latency; speedup is Python
median / Rust median. All compared word and POS outputs matched.
A speedup below 1 means the Python median was lower.

Word segmentation times `JaSegmenter::words(text)` against
`nagisa.tagging(text).words`. Python's `.words` property lazily performs
segmentation, so neither side computes POS tags in this mode. Words + POS
times `Tagger::tagging(text)` against creating a fresh `nagisa.tagging(text)`
result and reading **both** `.words` and `.postags`. Rust returns typed
`PosTag` values; conversion to original labels for validation and JSON occurs
outside the timed region.

Timings include preprocessing and output construction, but exclude model
loading, process startup, JSON I/O and destruction of the returned results.
Rust initializes from the bundled dictionary and lossless f32 weights;
Python uses its package's original model. The raw `load_ms` fields cover
different initialization paths and are not a startup-speed comparison.

These are warmed, single-thread CPU microbenchmarks on three synthetic
inputs. Input mix, CPU scheduling, host contention and build settings affect
results; they do not measure application throughput or GPU performance.

Reproduce with a CPython 3.12 environment:

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r tools/requirements-reference.txt
.venv/bin/python tools/benchmark.py --mode words --output results/words.json
.venv/bin/python tools/benchmark.py --mode tagging --output results/tagging.json
```

`uv` is optional; a regular Python 3.12 `venv` and `pip` work too. The tool
builds the Rust worker in release mode. Add `--external-model` to use
the Python package's original model files, or `--cpu N` on Linux to
pin both workers to an allowed logical CPU. Reports record Linux affinity,
cgroup v2 CPU/memory limits and CPU accounting before and after the runs.
The earlier 0.2.11 reports remain in [the benchmark directory](docs/benchmarks/).

### Multithread throughput

The [thread benchmark](docs/thread-benchmarks.md) compares one shared model
at **1 / 2 / 4 / 8 worker threads** on Apple M4 and AMD EPYC 9V74, for both
word segmentation and words + POS. It includes nagisa 0.3.0 on CPython 3.12
and CPython 3.14t, with an explicit GIL-on/off control on the same 3.14t build.

**The nagisa 0.3.0 PyPI extension automatically re-enables the GIL on 3.14t.**
The no-GIL benchmark therefore explicitly uses `-X gil=0` and verifies the
actual GIL state before import, after import and after inference. Every
timed output is checked against the committed reference outside the timer.
See the linked report for results, raw measurements, charts and reproduction.

Eight-thread median throughput on the 4,000-line mixed corpus:

| CPU | Task | ragisa lines/s | nagisa 3.14t no-GIL lines/s | Rust / Python | ragisa 8 / 1 scaling |
|---|---|---:|---:|---:|---:|
| Apple M4 | Words | 19,231 | 8,549 | 2.25× | 4.75× |
| Apple M4 | Words + POS | 11,812 | 6,722 | 1.76× | 4.72× |
| AMD EPYC 9V74 | Words | 11,731 | 8,939 | 1.31× | 7.24× |
| AMD EPYC 9V74 | Words + POS | 7,354 | 5,922 | 1.24× | 7.26× |

These are four-run medians, with 1 / 2 / 4 / 8-thread results and observed
variation in the full report. M4 has 4 performance and 6 efficiency cores;
AMD uses eight distinct physical cores. This workload differs from the
three synthetic inputs in the single-thread latency tables above.

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

The bundled tests cover **1,717** reference texts with both word and POS
outputs, **1,721** lowercase cases, **196** dictionary cases, **40** POS
selection cases, **20** pre-segmented casing cases, and shared-model reentrancy
at 1, 2, 4 and 8 threads.
To check the original-file loader as well as every bundled weight's f32 bits:

```sh
python3 tools/download_model.py
python3 tools/bundle_model.py --check
export RAGISA_MODEL_DIR="$PWD/models/nagisa-0.3.0"
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

The current implementation was verified locally with **0 / 1,717 word
mismatches and 0 / 1,717 POS mismatches**, including both types' 8-thread tests.
All **32 tests** pass locally with the original model and exhaustive Unicode audit data enabled.

CI runs on Linux and macOS, tests bundled loading before any model download,
then verifies the original data, all f32 weight bits and the Unicode tables,
runs the scalar Unicode audit, regenerates word/POS, inference-option and candidate-feature
Python references, and checks the
packaged crates. It also checks the external-only build and offline use of
the distributed packages. A separate Linux/macOS job checks the thread benchmark
with CPython 3.12 and 3.14t, including both explicit GIL states. CI checks output
parity and the measurement protocol; it does not enforce timing thresholds on
shared runners. The slower combining-sequence sweep is available locally.
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
