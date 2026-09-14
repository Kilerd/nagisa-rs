//! Pure-Rust Japanese word segmentation using nagisa 0.2.11's pretrained model.
//! Targets the output of `nagisa.tagging(text).words` under CPython 3.12.
//!
//! It loads nagisa's own data files -- `nagisa_v001.dict` (a gzipped pickle of
//! the vocabularies) and
//! `nagisa_v001.model` (DyNet's text parameter format) -- straight out of the
//! installed package's `data/` directory, so there is no conversion step and no
//! second copy of the weights to keep in sync.
//!
//! Only the word-segmentation half of the model is loaded; nagisa's POS tagger
//! is a separate BiLSTM whose output the word boundaries do not depend on.
//!
//! ```no_run
//! # fn main() -> Result<(), nagisa_rs::JaError> {
//! let seg = nagisa_rs::JaSegmenter::from_nagisa_dir("/venv/lib/python3.12/site-packages/nagisa/data")?;
//! assert_eq!(seg.words("Pythonで簡単に使えるツールです"),
//!            ["Python", "で", "簡単", "に", "使える", "ツール", "です"]);
//! # Ok(())
//! # }
//! ```
//!
//! [`JaSegmenter`] is immutable after construction: [`JaSegmenter::words`] takes
//! `&self`, keeps all scratch state on the stack/heap of the calling thread and
//! touches no globals, so it is `Send + Sync` and reentrant. (The Python
//! original uses DyNet's process-global computation graph.)

mod dynet;
mod error;
mod hash;
mod infer;
mod nfkc;
mod nfkd_tables;
mod pickle;
mod prepro;
mod unicode_tables;

use std::io::Read;
use std::path::Path;

pub use error::JaError;

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

/// nagisa's Japanese word segmenter.
pub struct JaSegmenter {
    vocab: Vocab,
    weights: Weights,
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
        Ok(Self { vocab, weights })
    }

    /// Segment text, targeting `nagisa.tagging(text).words` for nagisa 0.2.11
    /// under CPython 3.12 (Unicode 15.0.0).
    ///
    /// The words are cut out of the *preprocessed* text (`str.rstrip()`, NFKC,
    /// `İ`->`I`, ASCII space -> U+3000), which is what nagisa returns; the
    /// caller is responsible for any further filtering.
    #[must_use]
    pub fn words(&self, text: &str) -> Vec<String> {
        let chars = prepro::preprocess(text);
        infer::wakati(&self.vocab, &self.weights, &chars)
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
    let path = dir.join("nagisa_v001.dict");
    let file = std::fs::File::open(&path).map_err(|source| JaError::Io {
        path: path.clone(),
        source,
    })?;
    let mut raw = Vec::new();
    flate2::read::GzDecoder::new(std::io::BufReader::with_capacity(1 << 20, file))
        .read_to_end(&mut raw)
        .map_err(|source| JaError::Gzip {
            path: path.clone(),
            source,
        })?;
    let v = pickle::parse_vocabs(&raw)?;
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

    // The lookup tables are `{dim, vocab_size}`; the vocabulary sizes come from
    // the file itself (3090 / 82114 / 59260 for nagisa_v001) so only the
    // embedding width is pinned here.
    let uni = dynet::take_lookup(&mut p, &path, "/_0", DIM_UNI)?;
    let bi = dynet::take_lookup(&mut p, &path, "/_1", DIM_BI)?;
    let word = dynet::take_lookup(&mut p, &path, "/_2", DIM_WORD)?;
    let ctype = dynet::take(&mut p, &path, "/_3", &[DIM_CTYPE, N_CTYPE])?;
    let w_ws = dynet::take(&mut p, &path, "/_5", &[DIM_OUT, DIM_HIDDEN])?;
    let b_ws = dynet::take(&mut p, &path, "/_6", &[DIM_OUT])?;
    let trans_flat = dynet::take(&mut p, &path, "/_7", &[DIM_OUT, DIM_OUT])?;

    let mut trans = [[0.0f64; DIM_OUT]; DIM_OUT];
    for (next, row) in trans.iter_mut().enumerate() {
        for (prev, cell) in row.iter_mut().enumerate() {
            *cell = f64::from(trans_flat[next * DIM_OUT + prev]);
        }
    }

    let h4 = DIM_DIR * 4;
    let fwd = Lstm {
        wx: dynet::take(&mut p, &path, FWD_X, &[h4, DIM_INPUT])?,
        wh: dynet::take(&mut p, &path, FWD_H, &[h4, DIM_DIR])?,
        b: dynet::take(&mut p, &path, FWD_B, &[h4])?,
    };
    let bwd = Lstm {
        wx: dynet::take(&mut p, &path, BWD_X, &[h4, DIM_INPUT])?,
        wh: dynet::take(&mut p, &path, BWD_H, &[h4, DIM_DIR])?,
        b: dynet::take(&mut p, &path, BWD_B, &[h4])?,
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
