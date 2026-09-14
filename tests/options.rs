use nagisa_rs::{JaError, TaggedText, Tagger, TextOptions};
mod common;
use serde::Deserialize;
use std::sync::OnceLock;

#[derive(Deserialize)]
struct Expected {
    words: Vec<String>,
    postags: Vec<String>,
}

impl Expected {
    fn check(&self, actual: TaggedText, context: &str) {
        assert_eq!(actual.words, self.words, "words: {context:?}");
        assert_eq!(actual.postags, self.postags, "POS: {context:?}");
    }
}

#[derive(Deserialize)]
struct Lowercase {
    text: String,
    #[serde(flatten)]
    expected: Expected,
}

#[derive(Deserialize)]
struct Selection {
    text: String,
    lower: bool,
    labels: Vec<String>,
    #[serde(flatten)]
    expected: Expected,
    filtered: Expected,
    extracted: Expected,
}

#[derive(Deserialize)]
struct Dictionary {
    entries: Vec<String>,
    cases: Vec<Selection>,
}

#[derive(Deserialize)]
struct Tokens {
    words: Vec<String>,
    lower: bool,
    postags: Vec<String>,
}

#[derive(Deserialize)]
struct References {
    lowercase: Vec<Lowercase>,
    dictionaries: Vec<Dictionary>,
    selections: Vec<Selection>,
    postagging: Vec<Tokens>,
}

fn references() -> &'static References {
    static REFERENCES: OnceLock<References> = OnceLock::new();
    REFERENCES.get_or_init(|| serde_json::from_str(include_str!("fixtures/options.json")).unwrap())
}

fn tagger() -> &'static Tagger {
    static TAGGER: OnceLock<Tagger> = OnceLock::new();
    TAGGER.get_or_init(common::tagger)
}

#[test]
#[cfg_attr(
    not(any(have_nagisa_dir, feature = "bundled-model")),
    ignore = "set NAGISA_RS_MODEL_DIR for lowercase parity"
)]
fn lowercase_matches_python_words_and_postags() {
    let segmenter = common::segmenter();
    let options = TextOptions { lower: true };
    assert_eq!(references().lowercase.len(), 1717);
    for case in &references().lowercase {
        case.expected.check(
            tagger().tagging_with_options(&case.text, options),
            &case.text,
        );
        assert_eq!(
            segmenter.words_with_options(&case.text, options),
            case.expected.words,
            "{:?}",
            case.text
        );
    }
}

fn check_selection(tagger: &Tagger, case: &Selection) {
    let options = TextOptions { lower: case.lower };
    let labels: Vec<_> = case.labels.iter().map(String::as_str).collect();
    case.expected
        .check(tagger.tagging_with_options(&case.text, options), &case.text);
    case.filtered.check(
        tagger.filter_with_options(&case.text, &labels, options),
        &case.text,
    );
    case.extracted.check(
        tagger.extract_with_options(&case.text, &labels, options),
        &case.text,
    );
    if !case.lower {
        case.filtered
            .check(tagger.filter(&case.text, &labels), &case.text);
        case.extracted
            .check(tagger.extract(&case.text, &labels), &case.text);
    }
}

#[test]
#[cfg_attr(
    not(any(have_nagisa_dir, feature = "bundled-model")),
    ignore = "set NAGISA_RS_MODEL_DIR for dictionary parity"
)]
fn dictionaries_match_python_with_normalization_and_casing() {
    for group in &references().dictionaries {
        // Starting with another dictionary checks that the builder replaces it.
        let tagger = common::tagger()
            .with_single_word_list(["東京都民"])
            .with_single_word_list(&group.entries);
        let segmenter = common::segmenter().with_single_word_list(&group.entries);
        for case in &group.cases {
            check_selection(&tagger, case);
            assert_eq!(
                segmenter.words_with_options(&case.text, TextOptions { lower: case.lower }),
                case.expected.words,
                "{:?} dictionary: {:?}",
                case.text,
                group.entries
            );
        }
    }
    // Intentional literal semantics: upstream treats these as regex syntax.
    let literal = common::tagger().with_single_word_list(["Ｃ＋＋", "a.b", "[猫]", "  "]);
    for text in ["C++", "a.b", "[猫]"] {
        assert_eq!(literal.words(text), [text]);
    }
    assert_eq!(literal.words("  "), Vec::<String>::new());
}

#[test]
#[cfg_attr(
    not(any(have_nagisa_dir, feature = "bundled-model")),
    ignore = "set NAGISA_RS_MODEL_DIR for POS selection parity"
)]
fn filter_and_extract_match_python_including_empty_and_unknown_labels() {
    for case in &references().selections {
        check_selection(tagger(), case);
    }
}

#[test]
#[cfg_attr(
    not(any(have_nagisa_dir, feature = "bundled-model")),
    ignore = "set NAGISA_RS_MODEL_DIR for lowercase token POS parity"
)]
fn postagging_with_casing_matches_python() {
    for case in &references().postagging {
        assert_eq!(
            tagger()
                .postagging_with_options(&case.words, TextOptions { lower: case.lower })
                .unwrap(),
            case.postags,
            "{:?}, lower={}",
            case.words,
            case.lower
        );
    }
    assert!(matches!(
        tagger().postagging_with_options(&["猫", "\n"], TextOptions { lower: true }),
        Err(JaError::EmptyToken { index: 1 })
    ));
}

#[test]
#[cfg_attr(
    not(any(have_nagisa_dir, feature = "bundled-model")),
    ignore = "set NAGISA_RS_MODEL_DIR for concurrent option parity"
)]
fn dictionary_and_options_are_reentrant() {
    let group = &references().dictionaries[2];
    let tagger = common::tagger().with_single_word_list(&group.entries);
    std::thread::scope(|scope| {
        for _ in 0..8 {
            let tagger = &tagger;
            scope.spawn(move || {
                for case in &group.cases {
                    check_selection(tagger, case);
                }
            });
        }
    });
}
