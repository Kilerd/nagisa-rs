# Compatibility with nagisa 0.3.0

This upgrade compares upstream **0.2.11** with **0.3.0**, pinned to
[`4b7c9f5`](https://github.com/taishi-i/nagisa/tree/4b7c9f50060428063c46803fbbf772d725bc3b31).
The [0.3.0 release notes](https://github.com/taishi-i/nagisa/releases/tag/0.3.0)
claim roughly 3× single-thread inference speed and 25% less memory relative
to **0.2.12**. That is an upstream benchmark claim, not a measured ragisa
speedup. Our own comparisons against 0.3.0 are in the [README](../README.md#performance-against-python).

## User-visible changes

| Area | Upstream change since 0.2.11 | ragisa coverage |
|---|---|---|
| Segmentation, tagging, pre-segmented POS | Public inference methods are retained | `JaSegmenter` / `Tagger`; POS values use `PosTag` |
| BMES word assembly | 0.2.12 flushes incomplete words before a single-character token and at the end | Updated decoder; 12 direct Python reference cases |
| Stopwords | 0.2.12 adds the 135-entry `nagisa.stopwords` list | Public `STOPWORDS`, in identical order; callers choose whether to filter |
| Lowercase, forced words, POS selection | Existing arguments and behavior retained | Existing typed options and POS selection, regenerated against 0.3.0 |
| Preprocessing | Inference normalization and feature preprocessing unchanged | CPython 3.12 / Unicode 15.0.0 compatibility remains the tested target |
| Model training | `nagisa.fit` remains DyNet-based | Unimplemented, as recorded in the feature checklist |
| Custom models and hyperparameters | Upstream still supports its training/model workflow | Only the shipped architecture and original label IDs are supported |

The decoder change fixes real output differences in the original 1,713-text
corpus. For `注する療養氷枕ゆず`, the old decoder lost the final `ゆず`.
For `1997年 Funk This (Transvideo)`, it misplaced `1997` into a later word.
Both cases now match 0.3.0. Four upstream release regressions were appended,
bringing word/POS coverage to **1,717 texts**. These changes originated in
the [upstream word-loss fix](https://github.com/taishi-i/nagisa/commit/7f0e0578c3681433d8076575cfb2fb75aa670f93).

The forced-word dictionary still has a documented difference: ragisa treats
entries literally, whereas upstream escapes parentheses but leaves other
regex metacharacters active. Rust rejects empty normalized pre-segmented
tokens and unknown POS label strings with typed errors. The checklist is
specific to these inference semantics, not a claim that training or every
Python API is implemented.

## Inference implementation changes

Upstream switches `Tagger` from `nagisa.model.Model` to
[`nagisa.np_model.Model`](https://github.com/taishi-i/nagisa/blob/0.3.0/nagisa/np_model.py).
The network and pretrained parameters are retained, but inference no longer
constructs DyNet computation graphs. NumPy arrays hold the parameters and
features; a byte-count-aware [loader](https://github.com/taishi-i/nagisa/blob/0.3.0/nagisa/dynet_loader.py)
reads the original DyNet text dump directly.

The [Cython extension](https://github.com/taishi-i/nagisa/blob/0.3.0/nagisa/nagisa_utils.pyx)
adds C kernels for matrix projections and LSTM recurrence. It precomputes
input projections, fuses recurrent loops, batches character vectors and
reuses the vectors of identical character-ID sequences within one call.
Viterbi is rewritten as a C loop while retaining float64 scores and
first-index tie breaking. The numerical kernels release the GIL and keep
scratch state local. On supported Linux x86-64/GCC builds, target clones
select AVX2/FMA kernels at runtime. These changes replace repeated Python /
DyNet operations and BLAS dispatch in the inference path.

ragisa already uses Rust kernels, precomputed segmentation projections,
float64 Viterbi and per-call scratch data. This upgrade additionally adopts
per-call POS character-vector reuse and the repaired BMES assembly. Its
`Tagger` remains immutable, `Send + Sync`, and tested with concurrent callers.
It does not depend on NumPy, Cython or DyNet at runtime.

The model math is equivalent, but independently compiled float32 kernels
can round differently across CPUs and implementations. Compatibility tests
compare normalized words and POS labels, not bitwise-identical intermediate
activations or a guarantee for every possible input.

**Dependency detail:** inference imports no DyNet module in 0.3.0, but the
published [package metadata](https://github.com/taishi-i/nagisa/blob/0.3.0/pyproject.toml)
still lists DyNet38 as an installation dependency for Python ≥3.8. Our pinned
reference environment records that dependency; benchmark workers explicitly
verify `nagisa.np_model` and record whether DyNet was imported.

## Model and validation

The 0.3.0 source archive contains the exact dictionary, model and upstream
license bytes used previously. No new model download is needed by consumers,
and the bundled compressed assets are unchanged:

| Original file | SHA-256 |
|---|---|
| `nagisa_v001.dict` | `968ac9e6c7a53051ef24d8561673dd31de81b2feb9b5bff01b1d3b6b2473113c` |
| `nagisa_v001.model` | `9db9abc06a927c56e18af8d485e20a14908138752be459a83f0dc7ac85368c1b` |

The downloader now pins the 0.3.0 source archive. The existing checker still
compares all **2,517,042 f32 parameters**, original dictionary bytes, license
and bundled hashes. Model provenance is recorded in [the manifest](../data/manifest.json).

All word/POS and option fixtures were regenerated with nagisa 0.3.0 under
CPython 3.12.14. Candidate-POS set order, all 24 label IDs and pre-segmented
POS reference cases are unchanged. New compatibility fixtures cover the
stopword list, decoder behavior and upstream regression examples; CI
regenerates and compares these fixtures on Linux and macOS.

## Python baseline change observed in these runs

The earlier same-day runs used nagisa 0.2.11; the new runs use 0.3.0.
For the same three inputs, Python median latency improved by the ranges
below. These are separate warmed runs on the same hardware and thread
settings; the upstream release claim instead compares with 0.2.12.

| CPU | Word segmentation: Python 0.2.11 / 0.3.0 | Words + POS: Python 0.2.11 / 0.3.0 |
|---|---:|---:|
| Apple M4 | 2.59–2.67× | 2.52–2.89× |
| AMD EPYC 9V74 | 3.74–4.82× | 3.59–3.97× |

Raw data: [all benchmark reports](benchmarks/).
