//! Pure-Rust Japanese word segmentation and POS tagging with nagisa 0.2.11.
//! Targets `nagisa.tagging(text).words` and `.postags` under CPython 3.12.
//!
//! The original pretrained dictionary and lossless f32 weights are bundled
//! by default. `JaSegmenter::new()` and `Tagger::new()` load them in memory,
//! without Python, network access, model paths or temporary files.
//! [`JaSegmenter::from_nagisa_dir`] and [`Tagger::from_nagisa_dir`] also read
//! the original `nagisa_v001.dict` and DyNet text model files directly.
//!
//! [`JaSegmenter`] retains only segmentation data. [`Tagger`] also retains the POS
//! dictionary and networks and returns [`TaggedText`] with words and labels.
//!
//! ```
//! # #[cfg(feature = "bundled-model")]
//! # fn main() -> Result<(), ragisa::JaError> {
//! let seg = ragisa::JaSegmenter::new()?;
//! assert_eq!(seg.words("Pythonで簡単に使えるツールです"),
//!            ["Python", "で", "簡単", "に", "使える", "ツール", "です"]);
//! # Ok(())
//! # }
//! # #[cfg(not(feature = "bundled-model"))]
//! # fn main() {}
//! ```
//!
//! [`JaSegmenter`] is immutable after construction: [`JaSegmenter::words`] takes
//! `&self`, keeps all scratch state on the stack/heap of the calling thread and
//! touches no globals, so it is `Send + Sync` and reentrant. (The Python
//! original uses DyNet's process-global computation graph.)

mod dictionary;
mod dynet;
mod error;
mod hash;
mod infer;
mod nfkc;
mod nfkd_tables;
mod pickle;
mod pos;
mod pos_tag;
mod pos_unicode_tables;
mod prepro;
mod tagger;
mod unicode_tables;

use std::io::Read;
use std::path::Path;

pub use error::JaError;
pub use pos_tag::{ParsePosTagError, PosTag};
pub use tagger::{TaggedText, Tagger};

use infer::{
    DIM_BI, DIM_CTYPE, DIM_DIR, DIM_HIDDEN, DIM_INPUT, DIM_OUT, DIM_UNI, DIM_WORD, Lstm, N_CTYPE,
    Vocab, Weights,
};

const FWD_X: &str = "/birnn/vanilla-lstm-builder/_0";
const FWD_H: &str = "/birnn/vanilla-lstm-builder/_1";
const FWD_B: &str = "/birnn/vanilla-lstm-builder/_2";
const BWD_X: &str = "/birnn/vanilla-lstm-builder_1/_0";
const BWD_H: &str = "/birnn/vanilla-lstm-builder_1/_1";
const BWD_B: &str = "/birnn/vanilla-lstm-builder_1/_2";

/// Per-call options shared by segmentation, tagging and POS selection APIs.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct TextOptions {
    /// Return lowercase words and use lowercase tokens for POS features,
    /// matching nagisa's `lower=True`. Defaults to false.
    pub lower: bool,
}

/// nagisa's Japanese word segmenter.
pub struct JaSegmenter {
    vocab: Vocab,
    weights: Weights,
    dictionary: dictionary::Dictionary,
}

impl std::fmt::Debug for JaSegmenter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JaSegmenter")
            .field("unigrams", &self.vocab.uni2id.len())
            .field("bigrams", &self.vocab.bi2id.len())
            .field("words", &self.vocab.word2id.len())
            .finish()
    }
}

impl JaSegmenter {
    /// Load the bundled nagisa 0.2.11 model, without file or network access.
    /// Available with the default `bundled-model` feature. Load once and reuse.
    ///
    /// # Errors
    /// Returns [`JaError`] if the bundled model cannot be decoded or validated.
    #[cfg(feature = "bundled-model")]
    pub fn new() -> Result<Self, JaError> {
        let vocab = vocab_from(pickle::parse_vocabs(&bundled_dictionary()?)?)?;
        let weights = weights_from(
            &mut dynet::bundled()?,
            Path::new("bundled/nagisa_v001.bin.gz"),
        )?;
        Ok(Self {
            vocab,
            weights,
            dictionary: dictionary::Dictionary::default(),
        })
    }

    /// Load the segmenter from a nagisa `data/` directory.
    ///
    /// `dir` must contain `nagisa_v001.dict` and `nagisa_v001.model`, i.e. the
    /// layout of the installed `nagisa/data` package directory.
    ///
    /// # Errors
    /// Returns [`JaError`] if a file is missing or unreadable, if the pickle or
    /// the DyNet text format cannot be parsed, or if the parameters do not have
    /// the shapes nagisa 0.2.11 uses.
    pub fn from_nagisa_dir(dir: impl AsRef<Path>) -> Result<Self, JaError> {
        let dir = dir.as_ref();
        let vocab = load_vocab(dir)?;
        let weights = load_weights(dir)?;
        Ok(Self {
            vocab,
            weights,
            dictionary: dictionary::Dictionary::default(),
        })
    }

    /// Segment text, targeting `nagisa.tagging(text).words` for nagisa 0.2.11
    /// under CPython 3.12 (Unicode 15.0.0).
    ///
    /// The words are cut out of the *preprocessed* text (`str.rstrip()`, NFKC,
    /// `İ`->`I`, ASCII space -> U+3000), which is what nagisa returns; the
    /// caller is responsible for any further filtering.
    #[must_use]
    pub fn words(&self, text: &str) -> Vec<String> {
        self.words_with_options(text, TextOptions::default())
    }

    /// Segment with explicit output casing. Lowercase context is evaluated over
    /// the entire normalized text before words are cut, matching Python.
    #[must_use]
    pub fn words_with_options(&self, text: &str, options: TextOptions) -> Vec<String> {
        let chars = prepro::preprocess(text);
        infer::wakati(
            &self.vocab,
            &self.weights,
            &chars,
            options.lower,
            &self.dictionary,
        )
    }

    /// Replace the user dictionary with literal, case-sensitive forced words.
    ///
    /// Terms are normalized like input text. Entries of at most one character
    /// before normalization, and empty normalized entries, are ignored. At each
    /// position the longest match wins, with no overlaps. Unlike Python nagisa's
    /// partly escaped regex, regex metacharacters have their literal meaning.
    #[must_use]
    pub fn with_single_word_list<I, S>(mut self, words: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        self.dictionary = dictionary::Dictionary::new(words);
        self
    }

    /// Number of entries in the unigram, bigram and word vocabularies.
    #[must_use]
    pub fn vocab_sizes(&self) -> (usize, usize, usize) {
        (
            self.vocab.uni2id.len(),
            self.vocab.bi2id.len(),
            self.vocab.word2id.len(),
        )
    }
}

fn load_vocab(dir: &Path) -> Result<Vocab, JaError> {
    vocab_from(pickle::parse_vocabs(&read_dictionary(dir)?)?)
}

fn read_dictionary(dir: &Path) -> Result<Vec<u8>, JaError> {
    let path = dir.join("nagisa_v001.dict");
    let file = std::fs::File::open(&path).map_err(|source| JaError::Io {
        path: path.clone(),
        source,
    })?;
    inflate_dictionary(std::io::BufReader::with_capacity(1 << 20, file), &path)
}

#[cfg(feature = "bundled-model")]
fn bundled_dictionary() -> Result<Vec<u8>, JaError> {
    inflate_dictionary(
        &include_bytes!("../data/nagisa_v001.dict")[..],
        Path::new("bundled/nagisa_v001.dict"),
    )
}

fn inflate_dictionary(reader: impl Read, path: &Path) -> Result<Vec<u8>, JaError> {
    let mut raw = Vec::new();
    flate2::read::GzDecoder::new(reader)
        .read_to_end(&mut raw)
        .map_err(|source| JaError::Gzip {
            path: path.to_owned(),
            source,
        })?;
    Ok(raw)
}

fn vocab_from(v: pickle::Vocabs) -> Result<Vocab, JaError> {
    let oov = |m: &hash::FxMap<String, u32>, name: &'static str| -> Result<u32, JaError> {
        m.get("oov").copied().ok_or(JaError::MissingVocabKey {
            vocab: name,
            key: "oov",
        })
    };
    Ok(Vocab {
        uni_oov: oov(&v.uni2id, "uni2id")?,
        bi_oov: oov(&v.bi2id, "bi2id")?,
        word_oov: oov(&v.word2id, "word2id")?,
        uni2id: v.uni2id,
        bi2id: v.bi2id,
        word2id: v.word2id,
    })
}

fn load_weights(dir: &Path) -> Result<Weights, JaError> {
    let path = dynet::model_path(dir);
    let wanted = [
        "/_0", "/_1", "/_2", "/_3", "/_5", "/_6", "/_7", FWD_X, FWD_H, FWD_B, BWD_X, BWD_H, BWD_B,
    ];
    let mut p = dynet::load(&path, &wanted)?;
    weights_from(&mut p, &path)
}

fn weights_from(
    p: &mut std::collections::HashMap<String, dynet::RawParam>,
    path: &Path,
) -> Result<Weights, JaError> {
    // The lookup tables are `{dim, vocab_size}`; the vocabulary sizes come from
    // the file itself (3090 / 82114 / 59260 for nagisa_v001) so only the
    // embedding width is pinned here.
    let uni = dynet::take_lookup(p, path, "/_0", DIM_UNI)?;
    let bi = dynet::take_lookup(p, path, "/_1", DIM_BI)?;
    let word = dynet::take_lookup(p, path, "/_2", DIM_WORD)?;
    let ctype = dynet::take(p, path, "/_3", &[DIM_CTYPE, N_CTYPE])?;
    let w_ws = dynet::take(p, path, "/_5", &[DIM_OUT, DIM_HIDDEN])?;
    let b_ws = dynet::take(p, path, "/_6", &[DIM_OUT])?;
    let trans_flat = dynet::take(p, path, "/_7", &[DIM_OUT, DIM_OUT])?;

    let mut trans = [[0.0f64; DIM_OUT]; DIM_OUT];
    for (next, row) in trans.iter_mut().enumerate() {
        for (prev, cell) in row.iter_mut().enumerate() {
            *cell = f64::from(trans_flat[next * DIM_OUT + prev]);
        }
    }

    let h4 = DIM_DIR * 4;
    let fwd = Lstm {
        wx: dynet::take(p, path, FWD_X, &[h4, DIM_INPUT])?,
        wh: dynet::take(p, path, FWD_H, &[h4, DIM_DIR])?,
        b: dynet::take(p, path, FWD_B, &[h4])?,
    };
    let bwd = Lstm {
        wx: dynet::take(p, path, BWD_X, &[h4, DIM_INPUT])?,
        wh: dynet::take(p, path, BWD_H, &[h4, DIM_DIR])?,
        b: dynet::take(p, path, BWD_B, &[h4])?,
    };

    Ok(Weights {
        uni,
        bi,
        word,
        ctype,
        fwd,
        bwd,
        w_ws,
        b_ws,
        trans,
    })
}
