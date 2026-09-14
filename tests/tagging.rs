use ragisa::{JaError, PosTag, Tagger};
use serde::Deserialize;
use std::path::{Path, PathBuf};

#[derive(Deserialize)]
struct Case {
    text: String,
    words: Vec<String>,
    postags: Vec<String>,
}

mod common;
use common::{labels, tagger};

fn cases() -> Vec<Case> {
    let root = std::env::var_os("RAGISA_FIXTURES").map_or_else(
        || Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures"),
        PathBuf::from,
    );
    let files = if root.is_dir() {
        let mut files: Vec<_> = std::fs::read_dir(root)
            .expect("fixture directory")
            .map(|entry| entry.expect("fixture entry").path())
            .filter(|path| {
                path.extension().is_some_and(|ext| ext == "jsonl")
                    && path
                        .file_name()
                        .is_none_or(|name| name != "prepro_cases.jsonl")
            })
            .collect();
        files.sort();
        files
    } else {
        vec![root]
    };
    let mut cases = Vec::new();
    for file in files {
        let text = std::fs::read_to_string(&file).expect("read fixtures");
        for line in text.lines().filter(|line| !line.trim().is_empty()) {
            cases.push(
                serde_json::from_str(line).expect("fixture must contain text, words and postags"),
            );
        }
    }
    assert!(!cases.is_empty(), "no tagging fixtures found");
    cases
}

#[test]
#[cfg_attr(
    not(any(have_nagisa_dir, feature = "bundled-model")),
    ignore = "set RAGISA_MODEL_DIR for POS parity"
)]
fn tagging_matches_python() {
    let tagger = tagger();
    let mut mismatches = Vec::new();
    let cases = cases();
    for case in &cases {
        let got = tagger.tagging(&case.text);
        assert_eq!(got.words, case.words, "word mismatch: {:?}", case.text);
        if labels(&got.postags) != case.postags {
            mismatches.push(format!(
                "{:?}\nwords: {:?}\nPython: {:?}\nRust: {:?}",
                case.text, case.words, case.postags, got.postags
            ));
        }
    }
    for message in mismatches.iter().take(20) {
        eprintln!("{message}");
    }
    eprintln!(
        "POS parity: {} cases, {} mismatches",
        cases.len(),
        mismatches.len()
    );
    assert!(mismatches.is_empty(), "{} POS mismatches", mismatches.len());
}

#[test]
#[cfg_attr(
    not(any(have_nagisa_dir, feature = "bundled-model")),
    ignore = "set RAGISA_MODEL_DIR for pre-tokenized POS parity"
)]
fn postagging_matches_python_and_rejects_empty_tokens() {
    #[derive(Deserialize)]
    struct Tokens {
        words: Vec<String>,
        postags: Vec<String>,
    }
    let cases: Vec<Tokens> =
        serde_json::from_str(include_str!("fixtures/pos_tokens.json")).unwrap();
    let tagger = tagger();
    for case in cases {
        assert_eq!(
            labels(&tagger.postagging(&case.words).unwrap()),
            case.postags,
            "{:?}",
            case.words
        );
    }
    for invalid in ["", "\t", "  ", "\n\r"] {
        assert!(matches!(
            tagger.postagging(&["東京", invalid]),
            Err(JaError::EmptyToken { index: 1 })
        ));
    }
    assert_eq!(tagger.postags().len(), 24);
    assert_eq!(tagger.postags(), PosTag::ALL);
    assert_eq!(tagger.postags()[2], PosTag::Noun);
}

#[test]
#[cfg_attr(
    not(any(have_nagisa_dir, feature = "bundled-model")),
    ignore = "set RAGISA_MODEL_DIR for concurrent POS parity"
)]
fn tagging_is_send_sync_and_reentrant() {
    fn send_sync<T: Send + Sync>() {}
    send_sync::<Tagger>();
    let tagger = tagger();
    let cases = cases();
    // Include Unicode, whitespace, mixed-script, and natural text categories.
    let sample: Vec<_> = cases.iter().step_by((cases.len() / 80).max(1)).collect();
    std::thread::scope(|scope| {
        for _ in 0..8 {
            let tagger = &tagger;
            let sample = &sample;
            scope.spawn(move || {
                for case in sample {
                    let got = tagger.tagging(&case.text);
                    assert_eq!(got.words, case.words);
                    assert_eq!(labels(&got.postags), case.postags);
                }
            });
        }
    });
}
