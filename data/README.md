# Bundled pretrained data

The dictionary and all model parameters originate from
[nagisa 0.2.11](https://github.com/taishi-i/nagisa/tree/0.2.11), by Taishi Ikeda
and contributors. They are redistributed under the upstream
[MIT license](../licenses/nagisa-MIT.txt), also included in the model package.

- `nagisa_v001.dict` is the unmodified upstream gzip/pickle dictionary.
- `../model/data/nagisa_v001.bin.gz` contains every original f32 parameter,
  without quantization, pruning, retraining, or changes to vocabulary IDs.
- `manifest.json` records upstream and bundled SHA-256 hashes and asset sizes.

Both assets are committed as regular Git files, included in their Cargo
packages and embedded with `include_bytes!`. `nagisa-rs-model` carries the
weights and is an automatic dependency of the default `bundled-model`
feature. Each compressed crate fits under crates.io's default 10 MiB limit.
Loading requires no network, external paths, temporary files or writable cache.

To regenerate or verify assets (maintainers only):

```sh
python3 tools/download_model.py
python3 tools/bundle_model.py --check
# To rewrite from the same checksum-pinned source:
python3 tools/bundle_model.py
NAGISA_RS_MODEL_DIR="$PWD/models/nagisa-0.2.11" cargo test --release --locked
```

The checker compares the dictionary byte-for-byte and every binary model
record against the pinned source. The Rust test additionally compares all
2,517,042 parameters by `f32::to_bits()` against the original DyNet text reader.
Tests with no model-directory environment variable run all segmentation,
POS and inference-option fixtures using the bundled model.

The bundled storage format starts with eight magic/version bytes
`NAGISA\x00\x01`. Each original DyNet header line is followed by the number
of little-endian f32 values specified by its dimensions, with no separator
before the next header. The header's byte-count field still describes the
upstream text representation; binary readers use `product(dimensions) * 4`.
The whole stream is gzip-compressed with no filename and a zero timestamp.
The main crate pins the model crate's exact version so their formats agree.
