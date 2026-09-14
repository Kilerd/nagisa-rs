//! Shared-model batch throughput worker for tools/benchmark_threads.py.
use ragisa::{JaSegmenter, PosTag, TaggedText, Tagger};
use serde::{Deserialize, Serialize};
use std::{hint::black_box, sync::Barrier, time::Instant};

#[derive(Deserialize)]
struct Case {
    text: String,
    words: Vec<String>,
    postags: Vec<String>,
}

#[derive(Deserialize)]
struct Request {
    mode: String,
    threads: Vec<usize>,
    warmup: usize,
    cases: Vec<Case>,
    indices: Vec<usize>,
}

#[derive(Serialize)]
struct Measurement {
    threads: usize,
    elapsed_s: f64,
    verified_lines: usize,
    worker_lines: Vec<usize>,
}

fn measure(request: &Request, infer: impl Fn(&str) -> TaggedText + Sync) -> Vec<Measurement> {
    let expected: Vec<_> = request
        .cases
        .iter()
        .map(|case| {
            let result = TaggedText {
                words: case.words.clone(),
                postags: if request.mode == "tagging" {
                    case.postags
                        .iter()
                        .map(|label| label.parse::<PosTag>().expect("known POS label"))
                        .collect()
                } else {
                    Vec::new()
                },
            };
            assert_eq!(infer(&case.text), result, "serial reference parity");
            result
        })
        .collect();
    request
        .threads
        .iter()
        .map(|&threads| {
            let ready = Barrier::new(threads + 1);
            let start = Barrier::new(threads + 1);
            let done = Barrier::new(threads + 1);
            let validate = Barrier::new(threads + 1);
            std::thread::scope(|scope| {
                let handles: Vec<_> = (0..threads)
                    .map(|worker| {
                        let indices = &request.indices[worker * request.indices.len() / threads
                            ..(worker + 1) * request.indices.len() / threads];
                        let (infer, expected) = (&infer, &expected);
                        let (ready, start, done, validate) = (&ready, &start, &done, &validate);
                        scope.spawn(move || {
                            let mut outputs = Vec::with_capacity(indices.len());
                            for i in 0..request.warmup {
                                black_box(infer(&request.cases[indices[i % indices.len()]].text));
                            }
                            ready.wait();
                            start.wait();
                            for &index in indices {
                                outputs.push(black_box(infer(&request.cases[index].text)));
                            }
                            done.wait();
                            // Keep validation and output destruction outside the batch timer.
                            validate.wait();
                            for (&index, output) in indices.iter().zip(&outputs) {
                                assert_eq!(output, &expected[index], "concurrent reference parity");
                            }
                            outputs.len()
                        })
                    })
                    .collect();
                ready.wait();
                let timer = Instant::now();
                start.wait();
                done.wait();
                let elapsed_s = timer.elapsed().as_secs_f64();
                validate.wait();
                let worker_lines: Vec<_> = handles
                    .into_iter()
                    .map(|handle| handle.join().expect("benchmark worker"))
                    .collect();
                Measurement {
                    threads,
                    elapsed_s,
                    verified_lines: worker_lines.iter().sum(),
                    worker_lines,
                }
            })
        })
        .collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let request: Request = serde_json::from_reader(std::io::stdin().lock())?;
    if request.threads.is_empty()
        || request
            .threads
            .iter()
            .any(|&n| n == 0 || n > request.indices.len())
        || request.indices.iter().any(|&i| i >= request.cases.len())
    {
        return Err("nonempty cases/indices and 1 <= threads <= lines are required".into());
    }
    let dir = std::env::args().nth(1);
    let measurements = match request.mode.as_str() {
        "words" => {
            let segmenter = match dir {
                Some(dir) => JaSegmenter::from_nagisa_dir(dir)?,
                #[cfg(feature = "bundled-model")]
                None => JaSegmenter::new()?,
                #[cfg(not(feature = "bundled-model"))]
                None => return Err("provide a model directory or enable bundled-model".into()),
            };
            measure(&request, |text| TaggedText {
                words: segmenter.words(text),
                postags: Vec::new(),
            })
        }
        "tagging" => {
            let tagger = match dir {
                Some(dir) => Tagger::from_nagisa_dir(dir)?,
                #[cfg(feature = "bundled-model")]
                None => Tagger::new()?,
                #[cfg(not(feature = "bundled-model"))]
                None => return Err("provide a model directory or enable bundled-model".into()),
            };
            measure(&request, |text| tagger.tagging(text))
        }
        _ => return Err("mode must be words or tagging".into()),
    };
    serde_json::to_writer(
        std::io::stdout().lock(),
        &serde_json::json!({
            "measurements": measurements,
        }),
    )?;
    Ok(())
}
