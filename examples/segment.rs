//! Read one text per stdin line, print `nagisa.tagging(text).words` as JSON.
//!
//! ```text
//! cargo run --release -p ragisa --example segment < in.txt
//! cargo run --release -p ragisa --example segment -- <nagisa/data dir> < in.txt
//! ```
//! Lines are read raw (no unescaping); use it for eyeballing, and `tests/parity.rs`
//! for the real differential check.

use std::io::{BufRead, Write};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let seg = match std::env::args().nth(1) {
        Some(dir) => ragisa::JaSegmenter::from_nagisa_dir(dir)?,
        #[cfg(feature = "bundled-model")]
        None => ragisa::JaSegmenter::new()?,
        #[cfg(not(feature = "bundled-model"))]
        None => return Err("usage: segment <nagisa data dir> (or enable bundled-model)".into()),
    };
    let stdin = std::io::stdin();
    let mut out = std::io::BufWriter::new(std::io::stdout());
    for line in stdin.lock().lines() {
        let line = line.expect("read stdin");
        let words = seg.words(&line);
        serde_json::to_writer(&mut out, &words).expect("write JSON");
        writeln!(out).expect("write newline");
    }
    Ok(())
}
