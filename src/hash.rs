//! A tiny FxHash-style hasher.
//!
//! The segmenter does ~20 dictionary probes per input character, so the default
//! SipHash costs more than the model arithmetic on short inputs. This is the
//! same multiply-rotate mixer rustc uses; it is not DoS-resistant, which is
//! fine because the keys come from a fixed vocabulary file.

use std::hash::{BuildHasherDefault, Hasher};

pub(crate) type FxBuild = BuildHasherDefault<FxHasher>;
pub(crate) type FxMap<K, V> = std::collections::HashMap<K, V, FxBuild>;

const SEED: u64 = 0x51_7c_c1_b7_27_22_0a_95;

#[derive(Default)]
pub(crate) struct FxHasher {
    hash: u64,
}

impl FxHasher {
    #[inline]
    fn add(&mut self, word: u64) {
        self.hash = (self.hash.rotate_left(5) ^ word).wrapping_mul(SEED);
    }
}

impl Hasher for FxHasher {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        let mut rest = bytes;
        while let Some((chunk, tail)) = rest.split_first_chunk::<8>() {
            self.add(u64::from_ne_bytes(*chunk));
            rest = tail;
        }
        if let Some((chunk, tail)) = rest.split_first_chunk::<4>() {
            self.add(u64::from(u32::from_ne_bytes(*chunk)));
            rest = tail;
        }
        if let Some((chunk, tail)) = rest.split_first_chunk::<2>() {
            self.add(u64::from(u16::from_ne_bytes(*chunk)));
            rest = tail;
        }
        if let Some(&b) = rest.first() {
            self.add(u64::from(b));
        }
    }

    #[inline]
    fn write_u8(&mut self, i: u8) {
        self.add(u64::from(i));
    }

    #[inline]
    fn finish(&self) -> u64 {
        self.hash
    }
}
