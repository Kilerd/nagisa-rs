//! JSON benchmark worker used by tools/benchmark.py; timings exclude I/O and loading.
use ragisa::{JaSegmenter, TaggedText, Tagger};
use serde::{Deserialize, Serialize};
use std::hint::black_box;
use std::time::Instant;

#[derive(Deserialize)]
struct Request {
    #[serde(default)]
    mode: Mode,
    warmup: usize,
    iterations: usize,
    texts: Vec<String>,
}
#[derive(Default, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Mode {
    #[default]
    Words,
    Tagging,
}
#[derive(Serialize)]
struct Measurement {
    text: String,
    words: Vec<String>,
    postags: Vec<String>,
    samples_us: Vec<f64>,
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = std::env::args().nth(1);
    let request: Request = serde_json::from_reader(std::io::stdin().lock())?;
    if request.iterations == 0 {
        return Err("iterations must be positive".into());
    }
    let start = Instant::now();
    let (load_ms, measurements) = match request.mode {
        Mode::Words => {
            let seg = match dir {
                Some(dir) => JaSegmenter::from_nagisa_dir(dir)?,
                #[cfg(feature = "bundled-model")]
                None => JaSegmenter::new()?,
                #[cfg(not(feature = "bundled-model"))]
                None => return Err("provide a model directory or enable bundled-model".into()),
            };
            let load_ms = start.elapsed().as_secs_f64() * 1e3;
            let measurements = measure(request, |text| TaggedText {
                words: seg.words(text),
                postags: Vec::new(),
            });
            (load_ms, measurements)
        }
        Mode::Tagging => {
            let tagger = match dir {
                Some(dir) => Tagger::from_nagisa_dir(dir)?,
                #[cfg(feature = "bundled-model")]
                None => Tagger::new()?,
                #[cfg(not(feature = "bundled-model"))]
                None => return Err("provide a model directory or enable bundled-model".into()),
            };
            let load_ms = start.elapsed().as_secs_f64() * 1e3;
            (load_ms, measure(request, |text| tagger.tagging(text)))
        }
    };
    serde_json::to_writer(
        std::io::stdout().lock(),
        &serde_json::json!({
            "load_ms": load_ms, "measurements": measurements,
        }),
    )?;
    Ok(())
}

fn measure(request: Request, infer: impl Fn(&str) -> TaggedText) -> Vec<Measurement> {
    let mut measurements = Vec::new();
    for text in request.texts {
        for _ in 0..request.warmup {
            black_box(infer(black_box(&text)));
        }
        let mut samples_us = Vec::with_capacity(request.iterations);
        for _ in 0..request.iterations {
            let start = Instant::now();
            let result = black_box(infer(black_box(&text)));
            samples_us.push(start.elapsed().as_secs_f64() * 1e6);
            drop(result);
        }
        let result = infer(&text);
        measurements.push(Measurement {
            text,
            words: result.words,
            postags: result.postags,
            samples_us,
        });
    }
    measurements
}
