//! Pretrained nagisa 0.2.11 weights, distributed under the upstream MIT license.
//! Use the `ragisa` crate for segmentation and POS inference.

/// Versioned, gzip-compressed, lossless little-endian f32 model data.
/// This storage format is internal to the matching `ragisa` release.
pub static WEIGHTS_GZIP: &[u8] = include_bytes!("../data/nagisa_v001.bin.gz");
