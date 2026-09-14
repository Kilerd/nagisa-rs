# ragisa-model

Pretrained weights for [ragisa](https://github.com/Kilerd/ragisa), from
[nagisa 0.2.11](https://github.com/taishi-i/nagisa/tree/0.2.11) by Taishi Ikeda
and contributors. The original model and this data package are MIT licensed;
see [LICENSE](LICENSE).

This crate is installed automatically by ragisa's default `bundled-model`
feature. Applications should depend on `ragisa` and use `Tagger::new()`
or `JaSegmenter::new()`.

The data preserves every original f32 weight without quantization. Packaging
the weights separately keeps each crate below crates.io's default 10 MiB
upload limit. No build script, runtime download, or external model file is
needed. The format is versioned and internal to the matching ragisa
release. Its source hashes and regeneration instructions live in the main
repository's `data/` directory.
