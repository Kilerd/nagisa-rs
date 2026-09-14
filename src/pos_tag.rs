use std::{fmt, str::FromStr};

/// A part-of-speech label from nagisa 0.2.11's original 24-label model.
///
/// Use variants for matching and POS selection. [`Self::as_str`] and
/// [`Display`](fmt::Display) return the original Japanese label (or `oov` /
/// `URL`). [`FromStr`] accepts those exact labels and rejects unknown strings.
/// `Oov` and `UnknownWord` are distinct upstream labels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum PosTag {
    /// `oov`
    Oov,
    /// `補助記号`
    SupplementarySymbol,
    /// `名詞`
    Noun,
    /// `空白`
    Whitespace,
    /// `助詞`
    Particle,
    /// `接尾辞`
    Suffix,
    /// `動詞`
    Verb,
    /// `連体詞`
    Adnominal,
    /// `助動詞`
    AuxiliaryVerb,
    /// `形容詞`
    Adjective,
    /// `感動詞`
    Interjection,
    /// `接頭辞`
    Prefix,
    /// `記号`
    Symbol,
    /// `接続詞`
    Conjunction,
    /// `副詞`
    Adverb,
    /// `代名詞`
    Pronoun,
    /// `形状詞`
    AdjectivalNoun,
    /// `web誤脱`
    WebTypo,
    /// `URL`
    Url,
    /// `英単語`
    EnglishWord,
    /// `漢文`
    ClassicalChinese,
    /// `未知語`
    UnknownWord,
    /// `言いよどみ`
    Hesitation,
    /// `ローマ字文`
    RomanizedText,
}

impl PosTag {
    /// All labels, in the original model's numeric ID order.
    pub const ALL: [Self; 24] = [
        Self::Oov,
        Self::SupplementarySymbol,
        Self::Noun,
        Self::Whitespace,
        Self::Particle,
        Self::Suffix,
        Self::Verb,
        Self::Adnominal,
        Self::AuxiliaryVerb,
        Self::Adjective,
        Self::Interjection,
        Self::Prefix,
        Self::Symbol,
        Self::Conjunction,
        Self::Adverb,
        Self::Pronoun,
        Self::AdjectivalNoun,
        Self::WebTypo,
        Self::Url,
        Self::EnglishWord,
        Self::ClassicalChinese,
        Self::UnknownWord,
        Self::Hesitation,
        Self::RomanizedText,
    ];

    /// The exact upstream label, without allocating a string.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Oov => "oov",
            Self::SupplementarySymbol => "補助記号",
            Self::Noun => "名詞",
            Self::Whitespace => "空白",
            Self::Particle => "助詞",
            Self::Suffix => "接尾辞",
            Self::Verb => "動詞",
            Self::Adnominal => "連体詞",
            Self::AuxiliaryVerb => "助動詞",
            Self::Adjective => "形容詞",
            Self::Interjection => "感動詞",
            Self::Prefix => "接頭辞",
            Self::Symbol => "記号",
            Self::Conjunction => "接続詞",
            Self::Adverb => "副詞",
            Self::Pronoun => "代名詞",
            Self::AdjectivalNoun => "形状詞",
            Self::WebTypo => "web誤脱",
            Self::Url => "URL",
            Self::EnglishWord => "英単語",
            Self::ClassicalChinese => "漢文",
            Self::UnknownWord => "未知語",
            Self::Hesitation => "言いよどみ",
            Self::RomanizedText => "ローマ字文",
        }
    }
}

impl AsRef<str> for PosTag {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl fmt::Display for PosTag {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// An unrecognized label passed to [`PosTag::from_str`].
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("unknown POS label: {0:?}")]
pub struct ParsePosTagError(String);

impl FromStr for PosTag {
    type Err = ParsePosTagError;

    fn from_str(label: &str) -> Result<Self, Self::Err> {
        Self::ALL
            .into_iter()
            .find(|tag| tag.as_str() == label)
            .ok_or_else(|| ParsePosTagError(label.to_owned()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn labels_match_upstream_ids_and_roundtrip() {
        let labels: Vec<String> =
            serde_json::from_str(include_str!("../tests/fixtures/pos_labels.json")).unwrap();
        assert_eq!(labels.len(), PosTag::ALL.len());
        for (tag, label) in PosTag::ALL.into_iter().zip(labels) {
            assert_eq!(tag.as_str(), label);
            assert_eq!(tag.as_ref(), label);
            assert_eq!(tag.to_string(), label);
            assert_eq!(label.parse::<PosTag>(), Ok(tag));
        }
        assert_ne!(PosTag::Oov, PosTag::UnknownWord);
    }

    #[test]
    fn unknown_labels_are_errors_not_oov() {
        for label in ["", "unknown", "noun", "名次", "名詞 ", "OOV", "url"] {
            let error = label.parse::<PosTag>().unwrap_err();
            assert_eq!(error.to_string(), format!("unknown POS label: {label:?}"));
        }
    }
}
