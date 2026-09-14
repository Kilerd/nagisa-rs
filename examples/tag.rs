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
        let postags: Vec<_> = result.postags.iter().map(|tag| tag.as_str()).collect();
        serde_json::to_writer(
            &mut out,
            &serde_json::json!({"words": result.words, "postags": postags}),
        )?;
        writeln!(out)?;
    }
    Ok(())
}
