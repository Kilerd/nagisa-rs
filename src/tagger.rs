use crate::{JaError, JaSegmenter, dynet, infer::Lstm, pickle, pos};
use std::path::Path;

/// Normalized words and their corresponding part-of-speech labels.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TaggedText {
    /// Word segmentation, in text order.
    pub words: Vec<String>,
    /// One POS label for each word, in the same order.
    pub postags: Vec<String>,
}

/// Japanese segmentation and POS tagging with nagisa 0.2.11's original model.
///
/// Immutable, `Send + Sync`, and reentrant. Use [`JaSegmenter`] if only words
/// are needed; it does not load the POS dictionary or network parameters.
///
/// ```no_run
/// # fn main() -> Result<(), nagisa_rs::JaError> {
/// let tagger = nagisa_rs::Tagger::from_nagisa_dir("models/nagisa-0.2.11")?;
/// let result = tagger.tagging("Pythonで簡単に使えるツールです");
/// assert_eq!(result.postags, ["名詞", "助詞", "形状詞", "助動詞", "動詞", "名詞", "助動詞"]);
/// # Ok(())
/// # }
/// ```
pub struct Tagger {
    segmenter: JaSegmenter,
    vocab: pickle::PosVocabs,
    labels: Vec<String>,
    weights: pos::PosWeights,
}

impl std::fmt::Debug for Tagger {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tagger")
            .field("segmenter", &self.segmenter)
            .field("postags", &self.labels)
            .field("pos_dictionary_words", &self.vocab.word2postags.len())
            .finish()
    }
}

impl Tagger {
    /// Load segmentation and POS parameters from the original nagisa data directory.
    ///
    /// # Errors
    /// Returns [`JaError`] for missing, malformed or incompatible model data.
    pub fn from_nagisa_dir(dir: impl AsRef<Path>) -> Result<Self, JaError> {
        let dir = dir.as_ref();
        let (words, vocab) = pickle::parse_tagging_vocabs(&crate::read_dictionary(dir)?)?;
        let segmenter = JaSegmenter {
            vocab: crate::vocab_from(words)?,
            weights: crate::load_weights(dir)?,
        };
        let mut labels = vec![String::new(); pos::N_TAGS];
        for (label, &id) in &vocab.pos2id {
            let slot = labels
                .get_mut(id as usize)
                .ok_or(JaError::InvalidVocab("POS ID outside 0..24"))?;
            if !slot.is_empty() {
                return Err(JaError::InvalidVocab("duplicate POS ID"));
            }
            *slot = label.clone();
        }
        if labels.iter().any(String::is_empty) || labels[0] != "oov" || labels[2] != "名詞" {
            return Err(JaError::InvalidVocab(
                "expected nagisa 0.2.11's 24 POS labels",
            ));
        }
        if vocab
            .word2postags
            .values()
            .any(|ids| ids.is_empty() || ids.iter().any(|&id| id as usize >= pos::N_TAGS))
        {
            return Err(JaError::InvalidVocab("invalid candidate POS IDs"));
        }
        for (name, ids, size) in [
            (
                "unigram ID outside embedding table",
                &segmenter.vocab.uni2id,
                segmenter.weights.uni.len() / crate::infer::DIM_UNI,
            ),
            (
                "word ID outside embedding table",
                &segmenter.vocab.word2id,
                segmenter.weights.word.len() / crate::infer::DIM_WORD,
            ),
            (
                "bigram ID outside embedding table",
                &segmenter.vocab.bi2id,
                segmenter.weights.bi.len() / crate::infer::DIM_BI,
            ),
        ] {
            if ids.values().any(|&id| id as usize >= size) {
                return Err(JaError::InvalidVocab(name));
            }
        }
        let weights = load_weights(dir)?;
        Ok(Self {
            segmenter,
            vocab,
            labels,
            weights,
        })
    }

    /// Segment and label text, targeting `nagisa.tagging(text).words/postags`.
    #[must_use]
    pub fn tagging(&self, text: &str) -> TaggedText {
        let words = self.words(text);
        let postags = self.label_words(&words);
        TaggedText { words, postags }
    }

    /// Segment text without running the POS network.
    #[must_use]
    pub fn words(&self, text: &str) -> Vec<String> {
        self.segmenter.words(text)
    }

    /// Available POS labels, in upstream numeric ID order (including `oov`).
    #[must_use]
    pub fn postags(&self) -> &[String] {
        &self.labels
    }

    /// Normalize and label already-segmented words like `nagisa.postagging(words)`.
    /// A single ASCII or ideographic space is preserved as an ideographic space.
    ///
    /// # Errors
    /// Returns [`JaError::EmptyToken`] if a token is empty after normalization.
    /// An empty input list is valid and returns an empty list of labels.
    pub fn postagging<S: AsRef<str>>(&self, words: &[S]) -> Result<Vec<String>, JaError> {
        let mut normalized = Vec::with_capacity(words.len());
        for (index, word) in words.iter().enumerate() {
            let word = word.as_ref();
            let text = if word == " " || word == "\u{3000}" {
                "\u{3000}".to_owned()
            } else {
                crate::prepro::preprocess(word).into_iter().collect()
            };
            if text.is_empty() {
                return Err(JaError::EmptyToken { index });
            }
            normalized.push(text);
        }
        Ok(self.label_words(&normalized))
    }

    fn label_words(&self, words: &[String]) -> Vec<String> {
        pos::infer(
            &self.segmenter.vocab,
            &self.segmenter.weights,
            &self.weights,
            &self.vocab.word2postags,
            words,
        )
        .into_iter()
        .map(|id| self.labels[id].clone())
        .collect()
    }
}

fn load_weights(dir: &Path) -> Result<pos::PosWeights, JaError> {
    // Static names also appear in JaError::ModelMissing/ModelShape.
    const NAMES: [[&str; 3]; 4] = [
        [
            "/birnn_1/vanilla-lstm-builder/_0",
            "/birnn_1/vanilla-lstm-builder/_1",
            "/birnn_1/vanilla-lstm-builder/_2",
        ],
        [
            "/birnn_1/vanilla-lstm-builder_1/_0",
            "/birnn_1/vanilla-lstm-builder_1/_1",
            "/birnn_1/vanilla-lstm-builder_1/_2",
        ],
        [
            "/birnn_2/vanilla-lstm-builder/_0",
            "/birnn_2/vanilla-lstm-builder/_1",
            "/birnn_2/vanilla-lstm-builder/_2",
        ],
        [
            "/birnn_2/vanilla-lstm-builder_1/_0",
            "/birnn_2/vanilla-lstm-builder_1/_1",
            "/birnn_2/vanilla-lstm-builder_1/_2",
        ],
    ];
    let path = dynet::model_path(dir);
    let mut wanted = vec!["/_4", "/_8", "/_9"];
    wanted.extend(NAMES.iter().flatten().copied());
    let mut params = dynet::load(&path, &wanted)?;
    let mut lstm = |names: [&'static str; 3], input, hidden| -> Result<Lstm, JaError> {
        Ok(Lstm {
            wx: dynet::take(&mut params, &path, names[0], &[hidden * 4, input])?,
            wh: dynet::take(&mut params, &path, names[1], &[hidden * 4, hidden])?,
            b: dynet::take(&mut params, &path, names[2], &[hidden * 4])?,
        })
    };
    let fwd = lstm(NAMES[0], 64, 50)?;
    let bwd = lstm(NAMES[1], 64, 50)?;
    let char_fwd = lstm(NAMES[2], 32, 16)?;
    let char_bwd = lstm(NAMES[3], 32, 16)?;
    Ok(pos::PosWeights {
        tags: dynet::take(&mut params, &path, "/_4", &[16, pos::N_TAGS])?,
        output: dynet::take(&mut params, &path, "/_8", &[pos::N_TAGS, 100])?,
        bias: dynet::take(&mut params, &path, "/_9", &[pos::N_TAGS])?,
        fwd,
        bwd,
        char_fwd,
        char_bwd,
    })
}
