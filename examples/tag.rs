//! Read one text per stdin line and emit JSON words and POS labels.
use nagisa_rs::Tagger;
use std::io::{BufRead, Write};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = std::env::args()
        .nth(1)
        .ok_or("usage: tag <nagisa data dir>")?;
    let tagger = Tagger::from_nagisa_dir(dir)?;
    let mut out = std::io::BufWriter::new(std::io::stdout().lock());
    for line in std::io::stdin().lock().lines() {
        let result = tagger.tagging(&line?);
        serde_json::to_writer(
            &mut out,
            &serde_json::json!({"words": result.words, "postags": result.postags}),
        )?;
        writeln!(out)?;
    }
    Ok(())
}
