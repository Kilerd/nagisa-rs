use crate::{JaError, JaSegmenter, PosTag, TextOptions, dynet, infer::Lstm, pickle, pos};
use std::path::Path;

/// Normalized words and their corresponding part-of-speech labels.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TaggedText {
    /// Word segmentation, in text order.
    pub words: Vec<String>,
    /// One typed POS label for each word, in the same order.
    pub postags: Vec<PosTag>,
}

/// Japanese segmentation and POS tagging with nagisa 0.2.11's original model.
///
/// Immutable, `Send + Sync`, and reentrant. Use [`JaSegmenter`] if only words
/// are needed; it does not retain the POS dictionary or network parameters.
///
/// ```
/// # #[cfg(feature = "bundled-model")]
/// # fn main() -> Result<(), ragisa::JaError> {
/// use ragisa::PosTag::{AdjectivalNoun, AuxiliaryVerb, Noun, Particle, Verb};
/// let tagger = ragisa::Tagger::new()?;
/// let result = tagger.tagging("Pythonで簡単に使えるツールです");
/// assert_eq!(result.postags, [Noun, Particle, AdjectivalNoun, AuxiliaryVerb, Verb, Noun, AuxiliaryVerb]);
/// assert_eq!(tagger.extract("Pythonで簡単に使えるツールです", &[Noun]).words, ["Python", "ツール"]);
/// # Ok(())
/// # }
/// # #[cfg(not(feature = "bundled-model"))]
/// # fn main() {}
/// ```
pub struct Tagger {
    segmenter: JaSegmenter,
    vocab: pickle::PosVocabs,
    weights: pos::PosWeights,
}

impl std::fmt::Debug for Tagger {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tagger")
            .field("segmenter", &self.segmenter)
            .field("postags", &self.postags())
            .field("pos_dictionary_words", &self.vocab.word2postags.len())
            .finish()
    }
}

impl Tagger {
    /// Load bundled segmentation and POS data without file or network access.
    /// Available with the default `bundled-model` feature. Load once and reuse.
    ///
    /// # Errors
    /// Returns [`JaError`] if bundled data cannot be decoded or validated.
    #[cfg(feature = "bundled-model")]
    pub fn new() -> Result<Self, JaError> {
        let (words, vocab) = pickle::parse_tagging_vocabs(&crate::bundled_dictionary()?)?;
        let mut params = dynet::bundled()?;
        let path = Path::new("bundled/nagisa_v001.bin.gz");
        let segmenter = JaSegmenter {
            vocab: crate::vocab_from(words)?,
            weights: crate::weights_from(&mut params, path)?,
            dictionary: crate::dictionary::Dictionary::default(),
        };
        Self::from_parts(segmenter, vocab, weights_from(&mut params, path)?)
    }

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
            dictionary: crate::dictionary::Dictionary::default(),
        };
        Self::from_parts(segmenter, vocab, load_weights(dir)?)
    }

    fn from_parts(
        segmenter: JaSegmenter,
        vocab: pickle::PosVocabs,
        weights: pos::PosWeights,
    ) -> Result<Self, JaError> {
        validate_pos_labels(&vocab.pos2id)?;
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
        Ok(Self {
            segmenter,
            vocab,
            weights,
        })
    }

    /// Segment and label text, targeting `nagisa.tagging(text).words/postags`.
    #[must_use]
    pub fn tagging(&self, text: &str) -> TaggedText {
        self.tagging_with_options(text, TextOptions::default())
    }

    /// Segment text without running the POS network.
    #[must_use]
    pub fn words(&self, text: &str) -> Vec<String> {
        self.segmenter.words(text)
    }

    /// Segment with explicit output casing, without running POS inference.
    #[must_use]
    pub fn words_with_options(&self, text: &str, options: TextOptions) -> Vec<String> {
        self.segmenter.words_with_options(text, options)
    }

    /// Segment and label with explicit casing, matching `tagging(text, lower=...)`.
    #[must_use]
    pub fn tagging_with_options(&self, text: &str, options: TextOptions) -> TaggedText {
        let words = self.words_with_options(text, options);
        let postags = self.label_words(&words);
        TaggedText { words, postags }
    }

    /// Replace the user dictionary. See [`JaSegmenter::with_single_word_list`]
    /// for normalization, longest-match behavior and literal matching semantics.
    #[must_use]
    pub fn with_single_word_list<I, S>(mut self, words: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        self.segmenter = self.segmenter.with_single_word_list(words);
        self
    }

    /// Exclude words with any listed POS label, matching `nagisa.filter()`.
    /// An empty label list retains all words.
    #[must_use]
    pub fn filter(&self, text: &str, postags: &[PosTag]) -> TaggedText {
        self.filter_with_options(text, postags, TextOptions::default())
    }

    /// Exclude listed POS labels with explicit output casing.
    #[must_use]
    pub fn filter_with_options(
        &self,
        text: &str,
        postags: &[PosTag],
        options: TextOptions,
    ) -> TaggedText {
        select(self.tagging_with_options(text, options), postags, false)
    }

    /// Keep only words with a listed POS label, matching `nagisa.extract()`.
    /// An empty label list returns no words.
    #[must_use]
    pub fn extract(&self, text: &str, postags: &[PosTag]) -> TaggedText {
        self.extract_with_options(text, postags, TextOptions::default())
    }

    /// Keep only listed POS labels with explicit output casing.
    #[must_use]
    pub fn extract_with_options(
        &self,
        text: &str,
        postags: &[PosTag],
        options: TextOptions,
    ) -> TaggedText {
        select(self.tagging_with_options(text, options), postags, true)
    }

    /// Available POS labels, in upstream numeric ID order (including `oov`).
    #[must_use]
    pub fn postags(&self) -> &[PosTag] {
        &PosTag::ALL
    }

    /// Normalize and label already-segmented words like `nagisa.postagging(words)`.
    /// A single ASCII or ideographic space is preserved as an ideographic space.
    ///
    /// # Errors
    /// Returns [`JaError::EmptyToken`] if a token is empty after normalization.
    /// An empty input list is valid and returns an empty list of labels.
    pub fn postagging<S: AsRef<str>>(&self, words: &[S]) -> Result<Vec<PosTag>, JaError> {
        self.postagging_with_options(words, TextOptions::default())
    }

    /// Label pre-segmented words with explicit casing, matching
    /// `nagisa.postagging(words, lower=...)`. Lowercase context is per token.
    ///
    /// # Errors
    /// Returns [`JaError::EmptyToken`] for empty normalized input tokens.
    pub fn postagging_with_options<S: AsRef<str>>(
        &self,
        words: &[S],
        options: TextOptions,
    ) -> Result<Vec<PosTag>, JaError> {
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
            normalized.push(if options.lower {
                crate::prepro::lower(&text.chars().collect::<Vec<_>>())
                    .into_iter()
                    .collect()
            } else {
                text
            });
        }
        Ok(self.label_words(&normalized))
    }

    fn label_words(&self, words: &[String]) -> Vec<PosTag> {
        pos::infer(
            &self.segmenter.vocab,
            &self.segmenter.weights,
            &self.weights,
            &self.vocab.word2postags,
            words,
        )
        .into_iter()
        .map(|id| PosTag::ALL[id])
        .collect()
    }
}

// Select AFTER tagging the full sentence, preserving contextual POS labels.
fn select(tagged: TaggedText, labels: &[PosTag], keep: bool) -> TaggedText {
    let (words, postags) = tagged
        .words
        .into_iter()
        .zip(tagged.postags)
        .filter(|(_, tag)| labels.contains(tag) == keep)
        .unzip();
    TaggedText { words, postags }
}

fn validate_pos_labels(labels: &crate::hash::FxMap<String, u32>) -> Result<(), JaError> {
    if labels.len() != PosTag::ALL.len()
        || PosTag::ALL
            .iter()
            .enumerate()
            .any(|(id, tag)| labels.get(tag.as_str()) != Some(&(id as u32)))
    {
        return Err(JaError::InvalidVocab(
            "expected nagisa 0.2.11's 24 POS labels in original ID order",
        ));
    }
    Ok(())
}

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
fn load_weights(dir: &Path) -> Result<pos::PosWeights, JaError> {
    let path = dynet::model_path(dir);
    let mut wanted = vec!["/_4", "/_8", "/_9"];
    wanted.extend(NAMES.iter().flatten().copied());
    let mut params = dynet::load(&path, &wanted)?;
    weights_from(&mut params, &path)
}

fn weights_from(
    params: &mut std::collections::HashMap<String, dynet::RawParam>,
    path: &Path,
) -> Result<pos::PosWeights, JaError> {
    let mut lstm = |names: [&'static str; 3], input, hidden| -> Result<Lstm, JaError> {
        Ok(Lstm {
            wx: dynet::take(params, path, names[0], &[hidden * 4, input])?,
            wh: dynet::take(params, path, names[1], &[hidden * 4, hidden])?,
            b: dynet::take(params, path, names[2], &[hidden * 4])?,
        })
    };
    let fwd = lstm(NAMES[0], 64, 50)?;
    let bwd = lstm(NAMES[1], 64, 50)?;
    let char_fwd = lstm(NAMES[2], 32, 16)?;
    let char_bwd = lstm(NAMES[3], 32, 16)?;
    Ok(pos::PosWeights {
        tags: dynet::take(params, path, "/_4", &[16, pos::N_TAGS])?,
        output: dynet::take(params, path, "/_8", &[pos::N_TAGS, 100])?,
        bias: dynet::take(params, path, "/_9", &[pos::N_TAGS])?,
        fwd,
        bwd,
        char_fwd,
        char_bwd,
    })
}

#[cfg(test)]
mod label_tests {
    use super::*;

    #[test]
    fn rejects_incompatible_label_ids() {
        let original: Vec<String> =
            serde_json::from_str(include_str!("../tests/fixtures/pos_labels.json")).unwrap();
        let labels: crate::hash::FxMap<_, _> = original
            .into_iter()
            .enumerate()
            .map(|(id, label)| (label, id as u32))
            .collect();
        validate_pos_labels(&labels).unwrap();
        let mut unknown = labels.clone();
        unknown.remove("動詞");
        unknown.insert("unknown".into(), 6);
        assert!(validate_pos_labels(&unknown).is_err());
        let mut swapped = labels.clone();
        swapped.insert("動詞".into(), 4);
        swapped.insert("助詞".into(), 6);
        assert!(validate_pos_labels(&swapped).is_err());
        let mut duplicate = labels.clone();
        duplicate.insert("助詞".into(), 6);
        assert!(validate_pos_labels(&duplicate).is_err());
        let mut missing = labels;
        missing.remove("名詞");
        assert!(validate_pos_labels(&missing).is_err());
    }
}
