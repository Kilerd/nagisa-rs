//! Read one text per stdin line, print `nagisa.tagging(text).words` as JSON.
//!
//! ```text
//! cargo run --release -p nagisa-rs --example segment -- <nagisa/data dir> < in.txt
//! ```
//! Lines are read raw (no unescaping); use it for eyeballing, and `tests/parity.rs`
//! for the real differential check.

use std::io::{BufRead, Write};

fn main() {
    let dir = std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!("usage: segment <nagisa data dir>");
        std::process::exit(2);
    });
    let seg = nagisa_rs::JaSegmenter::from_nagisa_dir(&dir).expect("load nagisa model");
    let stdin = std::io::stdin();
    let mut out = std::io::BufWriter::new(std::io::stdout());
    for line in stdin.lock().lines() {
        let line = line.expect("read stdin");
        let words = seg.words(&line);
        serde_json::to_writer(&mut out, &words).expect("write JSON");
        writeln!(out).expect("write newline");
    }
}
