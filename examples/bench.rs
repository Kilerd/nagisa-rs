//! JSON benchmark worker used by tools/benchmark.py; timings exclude I/O and loading.
use nagisa_rs::JaSegmenter;
use serde::{Deserialize, Serialize};
use std::hint::black_box;
use std::time::Instant;

#[derive(Deserialize)]
struct Request {
    warmup: usize,
    iterations: usize,
    texts: Vec<String>,
}
#[derive(Serialize)]
struct Measurement {
    text: String,
    words: Vec<String>,
    samples_us: Vec<f64>,
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = std::env::args()
        .nth(1)
        .ok_or("usage: bench <nagisa data dir>")?;
    let request: Request = serde_json::from_reader(std::io::stdin().lock())?;
    if request.iterations == 0 {
        return Err("iterations must be positive".into());
    }
    let start = Instant::now();
    let seg = JaSegmenter::from_nagisa_dir(dir)?;
    let load_ms = start.elapsed().as_secs_f64() * 1e3;
    let mut measurements = Vec::new();
    for text in request.texts {
        for _ in 0..request.warmup {
            black_box(seg.words(black_box(&text)));
        }
        let mut samples_us = Vec::with_capacity(request.iterations);
        for _ in 0..request.iterations {
            let start = Instant::now();
            let words = black_box(seg.words(black_box(&text)));
            samples_us.push(start.elapsed().as_secs_f64() * 1e6);
            drop(words);
        }
        let words = seg.words(&text);
        measurements.push(Measurement {
            text,
            words,
            samples_us,
        });
    }
    serde_json::to_writer(
        std::io::stdout().lock(),
        &serde_json::json!({
            "load_ms": load_ms, "measurements": measurements,
        }),
    )?;
    Ok(())
}
