//! Differential test against `nagisa.tagging(text).words`.
//!
//! Fixtures are JSON lines of `{"cat": ..., "text": ..., "words": [...]}`
//! produced by `tools/reference.py` with nagisa 0.3.0 itself.
//!
//! Environment:
//! * `RAGISA_MODEL_DIR` -- nagisa's `data/` directory (the one that
//!   holds `nagisa_v001.model`). When unset, use the bundled model.
//! * `RAGISA_FIXTURES` -- a `.jsonl` file or a directory of them.
//!   Defaults to the committed subset in `tests/fixtures/`.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use ragisa::JaSegmenter;
use serde::Deserialize;

#[derive(Deserialize)]
struct Case {
    #[serde(default)]
    cat: String,
    text: String,
    words: Vec<String>,
}

mod common;

fn fixture_files() -> Vec<PathBuf> {
    let root = std::env::var_os("RAGISA_FIXTURES").map_or_else(
        || Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures"),
        PathBuf::from,
    );
    if root.is_dir() {
        let mut files: Vec<PathBuf> = std::fs::read_dir(&root)
            .expect("read fixture dir")
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| {
                p.extension().is_some_and(|e| e == "jsonl")
                    && p.file_name().is_none_or(|n| n != "prepro_cases.jsonl")
            })
            .collect();
        files.sort();
        files
    } else if root.is_file() {
        vec![root]
    } else {
        Vec::new()
    }
}

fn load_cases() -> Vec<Case> {
    let mut cases = Vec::new();
    for f in fixture_files() {
        let body = std::fs::read_to_string(&f).expect("read fixture file");
        for (n, line) in body.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            let case: Case = serde_json::from_str(line)
                .unwrap_or_else(|e| panic!("{}:{}: {e}", f.display(), n + 1));
            cases.push(case);
        }
    }
    cases
}

#[cfg_attr(
    not(any(have_nagisa_dir, feature = "bundled-model")),
    ignore = "needs nagisa's data/ dir: set RAGISA_MODEL_DIR (see README.md)"
)]
#[test]
fn words_match_nagisa_exactly() {
    let seg = common::segmenter();
    let cases = load_cases();
    assert!(!cases.is_empty(), "no fixtures found; set RAGISA_FIXTURES");

    let mut per_cat: BTreeMap<&str, (usize, usize)> = BTreeMap::new();
    let mut shown = 0usize;
    let mut bad = 0usize;
    for case in &cases {
        let got = seg.words(&case.text);
        let entry = per_cat.entry(case.cat.as_str()).or_default();
        entry.0 += 1;
        if got != case.words {
            bad += 1;
            entry.1 += 1;
            if shown < 20 {
                shown += 1;
                eprintln!(
                    "MISMATCH [{}] {:?}\n  nagisa: {:?}\n  rust  : {:?}",
                    case.cat, case.text, case.words, got
                );
            }
        }
    }
    eprintln!("parity: {} cases, {bad} mismatches", cases.len());
    for (cat, (n, b)) in &per_cat {
        eprintln!("  {cat:>16}: {n:>6} cases, {b} mismatches");
    }
    assert_eq!(bad, 0, "{bad}/{} sentences differ from nagisa", cases.len());
}

#[cfg_attr(
    not(any(have_nagisa_dir, feature = "bundled-model")),
    ignore = "needs nagisa's data/ dir: set RAGISA_MODEL_DIR (see README.md)"
)]
#[test]
fn probes() {
    let seg = common::segmenter();
    let expect: &[(&str, &[&str])] = &[
        ("", &[]),
        ("a", &["a"]),
        ("の", &["の"]),
        ("ノ", &["ノ"]),
        (
            "Pythonで簡単に使えるツールです",
            &["Python", "で", "簡単", "に", "使える", "ツール", "です"],
        ),
        ("東京に行きます", &["東京", "に", "行き", "ます"]),
        ("ラーメンたべたい", &["ラーメン", "たべ", "たい"]),
        (
            "すごい hello 中",
            &["すごい", "\u{3000}", "hello", "\u{3000}", "中"],
        ),
    ];
    for (text, want) in expect {
        assert_eq!(&seg.words(text), want, "input {text:?}");
    }
}

#[cfg_attr(
    not(any(have_nagisa_dir, feature = "bundled-model")),
    ignore = "needs nagisa's data/ dir: set RAGISA_MODEL_DIR (see README.md)"
)]
#[test]
fn is_send_sync_and_reentrant() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<JaSegmenter>();

    let seg = common::segmenter();
    let cases = load_cases();
    let sample: Vec<&Case> = cases.iter().take(400).collect();
    let expected: Vec<Vec<String>> = sample.iter().map(|c| c.words.clone()).collect();

    std::thread::scope(|scope| {
        for _ in 0..8 {
            let seg = &seg;
            let sample = &sample;
            let expected = &expected;
            scope.spawn(move || {
                for _ in 0..3 {
                    for (case, want) in sample.iter().zip(expected) {
                        assert_eq!(&seg.words(&case.text), want, "input {:?}", case.text);
                    }
                }
            });
        }
    });
}
