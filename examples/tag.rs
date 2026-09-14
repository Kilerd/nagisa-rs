//! Read one text per stdin line and emit JSON words and POS labels.
use ragisa::Tagger;
use std::io::{BufRead, Write};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let tagger = match std::env::args().nth(1) {
        Some(dir) => Tagger::from_nagisa_dir(dir)?,
        #[cfg(feature = "bundled-model")]
        None => Tagger::new()?,
        #[cfg(not(feature = "bundled-model"))]
        None => return Err("usage: tag <nagisa data dir> (or enable bundled-model)".into()),
    };
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
