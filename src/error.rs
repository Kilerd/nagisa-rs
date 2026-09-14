//! Errors raised while loading nagisa's data files.

use std::path::PathBuf;

/// Everything that can go wrong loading a nagisa model directory.
#[derive(Debug, thiserror::Error)]
pub enum JaError {
    /// A data file could not be read.
    #[error("cannot read {path}: {source}")]
    Io {
        /// The file that failed.
        path: PathBuf,
        /// The underlying I/O error.
        #[source]
        source: std::io::Error,
    },
    /// The `.dict` file is not gzip-compressed data.
    #[error("cannot inflate {path}: {source}")]
    Gzip {
        /// The file that failed.
        path: PathBuf,
        /// The underlying inflate error.
        #[source]
        source: std::io::Error,
    },
    /// The pickle stream ended in the middle of an opcode.
    #[error("vocabulary pickle is truncated")]
    PickleTruncated,
    /// The pickle stream is not the 5-dict list nagisa writes.
    #[error("vocabulary pickle has an unexpected shape")]
    PickleBadStream,
    /// The pickle stream uses an opcode this reader does not implement.
    #[error("vocabulary pickle uses unsupported opcode {0:#04x}")]
    PickleOpcode(u8),
    /// A vocabulary is missing a key the model needs.
    #[error("vocabulary {vocab} has no {key:?} entry")]
    MissingVocabKey {
        /// Which of the three vocabularies.
        vocab: &'static str,
        /// The key that is missing.
        key: &'static str,
    },
    /// A line of the DyNet text model file could not be understood.
    #[error("{path}: malformed parameter header at byte {offset}: {line:?}")]
    ModelHeader {
        /// The model file.
        path: PathBuf,
        /// Byte offset of the header.
        offset: u64,
        /// The offending line (truncated).
        line: String,
    },
    /// A float in the DyNet text model file could not be parsed.
    #[error("{path}: parameter {name} has a malformed value")]
    ModelValue {
        /// The model file.
        path: PathBuf,
        /// The parameter being read.
        name: String,
    },
    /// The model file does not contain a parameter the segmenter needs.
    #[error("{path}: missing parameter {name}")]
    ModelMissing {
        /// The model file.
        path: PathBuf,
        /// The parameter that is missing.
        name: &'static str,
    },
    /// A parameter has dimensions the segmenter cannot use.
    #[error("{path}: parameter {name} has dims {got:?}, expected {want:?}")]
    ModelShape {
        /// The model file.
        path: PathBuf,
        /// The parameter with the wrong shape.
        name: &'static str,
        /// The dims found in the file.
        got: Vec<usize>,
        /// The dims the segmenter needs.
        want: Vec<usize>,
    },
}
