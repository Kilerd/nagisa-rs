//! Reader for DyNet's text parameter format, as written by
//! `ParameterCollection::save` and read back by `model.populate(...)`.
//!
//! The file is a flat sequence of
//!
//! ```text
//! #Parameter# /birnn/vanilla-lstm-builder/_0 {200,200} 640001 ZERO_GRAD\n
//! <value> <value> ... <value> \n
//! ```
//!
//! blocks (`#LookupParameter#` for lookup tables). The number after the dims is
//! the byte length of the value line *including* its trailing newline, which
//! lets unwanted parameters be skipped without parsing them. Values are written
//! with `%+.8e` followed by a space, i.e. exactly 16 bytes each, in DyNet's
//! storage order: column-major for a matrix, entry-major for a lookup table
//! (so `{dim,count}` is `count` consecutive `dim`-sized rows).
//!
//! Verified against `Parameter.as_array()` for all 28 parameters of
//! `nagisa_v001.model`: max |diff| == 0 for every one.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use crate::error::JaError;

/// One parameter as stored in the file.
pub(crate) struct RawParam {
    /// The `{a,b}` dims from the header.
    pub dims: Vec<usize>,
    /// Values in DyNet storage order.
    pub data: Vec<f32>,
}

/// Read the named parameters out of a DyNet text model file.
///
/// Parameters not in `wanted` are skipped with a seek. The segmentation-only
/// loader uses this to avoid parsing POS parameters.
pub(crate) fn load(path: &Path, wanted: &[&str]) -> Result<HashMap<String, RawParam>, JaError> {
    let file = File::open(path).map_err(|source| JaError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let mut rdr = BufReader::with_capacity(1 << 20, file);
    let mut out: HashMap<String, RawParam> = HashMap::new();
    let mut header = Vec::with_capacity(128);
    let mut offset: u64 = 0;
    let mut values = Vec::new();

    loop {
        header.clear();
        let n = rdr
            .read_until(b'\n', &mut header)
            .map_err(|source| JaError::Io {
                path: path.to_path_buf(),
                source,
            })?;
        if n == 0 {
            break;
        }
        let head_at = offset;
        offset += n as u64;
        let line = std::str::from_utf8(&header)
            .map_err(|_| bad_header(path, head_at, &header))?
            .trim_end_matches('\n');
        let (name, dims, nbytes) =
            parse_header(line).ok_or_else(|| bad_header(path, head_at, &header))?;

        if wanted.contains(&name) {
            values.clear();
            values.resize(nbytes, 0u8);
            rdr.read_exact(&mut values).map_err(|source| JaError::Io {
                path: path.to_path_buf(),
                source,
            })?;
            let count: usize = dims.iter().product();
            let data = parse_values(&values, count).ok_or_else(|| JaError::ModelValue {
                path: path.to_path_buf(),
                name: name.to_owned(),
            })?;
            out.insert(name.to_owned(), RawParam { dims, data });
        } else {
            rdr.seek(SeekFrom::Current(nbytes as i64))
                .map_err(|source| JaError::Io {
                    path: path.to_path_buf(),
                    source,
                })?;
        }
        offset += nbytes as u64;
    }
    Ok(out)
}

fn bad_header(path: &Path, offset: u64, raw: &[u8]) -> JaError {
    JaError::ModelHeader {
        path: path.to_path_buf(),
        offset,
        line: String::from_utf8_lossy(&raw[..raw.len().min(96)]).into_owned(),
    }
}

/// `#Parameter# <name> {200,50} 160001 ZERO_GRAD` -> (name, [200, 50], 160001).
fn parse_header(line: &str) -> Option<(&str, Vec<usize>, usize)> {
    let rest = line
        .strip_prefix("#Parameter# ")
        .or_else(|| line.strip_prefix("#LookupParameter# "))?;
    let mut it = rest.split(' ');
    let name = it.next()?;
    let dims = it.next()?;
    let nbytes = it.next()?.parse::<usize>().ok()?;
    let dims = dims
        .strip_prefix('{')?
        .strip_suffix('}')?
        .split(',')
        .map(|d| d.parse::<usize>().ok())
        .collect::<Option<Vec<usize>>>()?;
    Some((name, dims, nbytes))
}

/// Parse `count` values out of a value line.
///
/// The fast path relies on the fixed `%+.8e ` width (16 bytes); anything else
/// falls back to whitespace splitting. Both use `str::parse::<f32>`, which is
/// correctly rounded, so the 9 significant digits DyNet writes round-trip the
/// original f32 exactly.
fn parse_values(line: &[u8], count: usize) -> Option<Vec<f32>> {
    let body = line.strip_suffix(b"\n")?;
    let mut out = Vec::with_capacity(count);
    if body.len() == count * 16 {
        for rec in body.as_chunks::<16>().0 {
            if rec[15] != b' ' {
                out.clear();
                break;
            }
            let s = std::str::from_utf8(&rec[..15]).ok()?;
            out.push(s.parse::<f32>().ok()?);
        }
        if out.len() == count {
            return Some(out);
        }
        out.clear();
    }
    for tok in body.split(|b| *b == b' ' || *b == b'\t') {
        if tok.is_empty() {
            continue;
        }
        let s = std::str::from_utf8(tok).ok()?;
        out.push(s.parse::<f32>().ok()?);
    }
    (out.len() == count).then_some(out)
}

/// Fetch a parameter and check its dims.
pub(crate) fn take(
    params: &mut HashMap<String, RawParam>,
    path: &Path,
    name: &'static str,
    want: &[usize],
) -> Result<Vec<f32>, JaError> {
    let p = params.remove(name).ok_or(JaError::ModelMissing {
        path: path.to_path_buf(),
        name,
    })?;
    if p.dims != want {
        return Err(JaError::ModelShape {
            path: path.to_path_buf(),
            name,
            got: p.dims,
            want: want.to_vec(),
        });
    }
    Ok(p.data)
}

/// The model file path inside a nagisa `data/` directory.
pub(crate) fn model_path(dir: &Path) -> PathBuf {
    dir.join("nagisa_v001.model")
}

/// Fetch a lookup table and check only its embedding width; the vocabulary
/// size is whatever the file says.
pub(crate) fn take_lookup(
    params: &mut HashMap<String, RawParam>,
    path: &Path,
    name: &'static str,
    dim: usize,
) -> Result<Vec<f32>, JaError> {
    let p = params.remove(name).ok_or(JaError::ModelMissing {
        path: path.to_path_buf(),
        name,
    })?;
    if p.dims.len() != 2 || p.dims[0] != dim {
        return Err(JaError::ModelShape {
            path: path.to_path_buf(),
            name,
            got: p.dims,
            want: vec![dim, 0],
        });
    }
    Ok(p.data)
}
