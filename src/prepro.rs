//! Port of nagisa's `utils.preprocess`, `text.lower()` and `get_chartype`.
//!
//! `nagisa_utils.pyx`:
//! ```python
//! cpdef unicode preprocess(text):
//!     text = utf8rstrip(text)                     # str.rstrip()
//!     text = unicodedata.normalize('NFKC', text)
//!     text = text.replace('İ', 'I')
//!     text = text.replace(' ', '　')
//!     return text
//! ```
//! `Tagger.wakati` then extracts features from `text.lower()` but cuts the
//! words out of the un-lowered `text`.

use crate::unicode_tables::{CASE_IGNORABLE, CASED_STOP, LOWER, LOWER_MAX, LOWER_MIN};

/// CPython's `Py_UNICODE_ISSPACE`, which is what `str.rstrip()` strips.
///
/// This is the Unicode White_Space set plus the four C0 separators
/// U+001C..U+001F, which Unicode does not classify as whitespace; enumerated
/// from CPython 3.11 with `[c for c in range(0x110000) if chr(c).isspace()]`.
fn is_py_space(c: char) -> bool {
    matches!(
        c as u32,
        0x09..=0x0D
            | 0x1C..=0x1F
            | 0x20
            | 0x85
            | 0xA0
            | 0x1680
            | 0x2000..=0x200A
            | 0x2028
            | 0x2029
            | 0x202F
            | 0x205F
            | 0x3000
    )
}

/// `utils.preprocess`: right-strip, NFKC, `İ`->`I`, ASCII space -> U+3000.
pub(crate) fn preprocess(text: &str) -> Vec<char> {
    let stripped: Vec<char> = text.trim_end_matches(is_py_space).chars().collect();
    crate::nfkc::nfkc(&stripped)
        .into_iter()
        .map(|c| match c {
            '\u{0130}' => 'I',
            ' ' => '\u{3000}',
            other => other,
        })
        .collect()
}

fn in_ranges(c: char, ranges: &[(u32, u32)]) -> bool {
    let u = c as u32;
    ranges
        .binary_search_by(|&(lo, hi)| {
            if u < lo {
                std::cmp::Ordering::Greater
            } else if u > hi {
                std::cmp::Ordering::Less
            } else {
                std::cmp::Ordering::Equal
            }
        })
        .is_ok()
}

/// CPython's Final_Sigma context test for the U+03A3 at `i`.
fn final_sigma(chars: &[char], i: usize) -> bool {
    let before = chars[..i]
        .iter()
        .rev()
        .find(|c| !in_ranges(**c, &CASE_IGNORABLE));
    match before {
        Some(c) if in_ranges(*c, &CASED_STOP) => {}
        _ => return false,
    }
    let after = chars[i + 1..]
        .iter()
        .find(|c| !in_ranges(**c, &CASE_IGNORABLE));
    match after {
        Some(c) => !in_ranges(*c, &CASED_STOP),
        None => true,
    }
}

/// CPython's `str.lower()` for a single character, from the generated
/// Unicode 15.0 table rather than `char::to_lowercase`.
///
/// Rust's `std` tracks whatever Unicode release the compiler shipped with
/// (17.0 as of rustc 1.98), which lowercases 55 code points that CPython 3.12
/// leaves alone -- and a different character means a different vocabulary id,
/// hence a different word count. Driving the mapping off nagisa's own
/// interpreter removes that coupling to the toolchain entirely.
fn lower_char(c: char) -> char {
    let u = c as u32;
    if !(LOWER_MIN..=LOWER_MAX).contains(&u) {
        return c;
    }
    match LOWER.binary_search_by_key(&u, |&(from, _)| from) {
        Ok(i) => char::from_u32(LOWER[i].1).unwrap_or(c),
        Err(_) => c,
    }
}

/// Python's `str.lower()`, restricted to the 1:1 case.
///
/// U+0130 is the only code point whose full lowercase mapping is longer than
/// one character, and `preprocess` has already replaced it with `I`, so the
/// character count is preserved; nagisa depends on that (it zips the tag
/// sequence against the *un-lowered* string).
pub(crate) fn lower(chars: &[char]) -> Vec<char> {
    chars
        .iter()
        .enumerate()
        .map(|(i, &c)| {
            if c == '\u{03A3}' {
                if final_sigma(chars, i) {
                    '\u{03C2}'
                } else {
                    '\u{03C3}'
                }
            } else {
                lower_char(c)
            }
        })
        .collect()
}

/// `utils.get_chartype`: hiragana / katakana / kanji / latin / digit / other.
pub(crate) fn chartype(c: char) -> u32 {
    let u = c as u32;
    if (0x3040..=0x309F).contains(&u) {
        0
    } else if (0x30A1..=0x30FA).contains(&u) {
        1
    } else if (0x4E00..=0x9FA5).contains(&u) {
        2
    } else if c.is_ascii_alphabetic() {
        3
    } else if c.is_ascii_digit() {
        4
    } else {
        5
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pre(s: &str) -> String {
        preprocess(s).into_iter().collect()
    }

    #[test]
    fn preprocess_matches_python() {
        assert_eq!(pre("テスト  \t "), "テスト");
        assert_eq!(pre("a b"), "a\u{3000}b");
        assert_eq!(pre("İstanbul"), "Istanbul");
        assert_eq!(pre("①②③"), "123");
        assert_eq!(pre("ﾊﾝｶｸ"), "ハンカク");
        assert_eq!(pre("ｶﾞ"), "ガ");
        assert_eq!(pre("　"), "");
        assert_eq!(pre(""), "");
        assert_eq!(pre("\u{1c}\u{1f}"), "");
    }

    #[test]
    fn final_sigma_rule() {
        let low = |s: &str| -> String {
            lower(&s.chars().collect::<Vec<char>>())
                .into_iter()
                .collect()
        };
        assert_eq!(low("ΑΣ"), "ας");
        assert_eq!(low("ΣΑ"), "σα");
        assert_eq!(low("ΑΣB"), "ασb");
        assert_eq!(low("Σ"), "σ");
        assert_eq!(low("ΑΣ."), "ας.");
        assert_eq!(low("ΑΣ'"), "ας'");
    }

    /// The whole pure-Rust front half of the segmenter -- `preprocess`
    /// (`str.rstrip` + NFKC + `İ`->`I` + space->U+3000), `lower` (including
    /// Final_Sigma) and `get_chartype` -- against values dumped from the
    /// CPython 3.12 reference interpreter, using nagisa's own
    /// `nagisa_utils.preprocess`.
    ///
    /// This one needs **no model files**, so it runs from a bare clone and is
    /// the standing gate for the bug class that a wrong Unicode table
    /// introduces: those three functions are the only place a code point can
    /// turn into a different vocabulary id, and hence into a different word
    /// count. The fixture carries every code point where the Unicode release
    /// the Rust toolchain ships disagrees with CPython 3.12, plus the Python
    /// whitespace/NFKC/case edge cases and a spread of real corpus lines.
    #[test]
    fn preprocessing_matches_cpython312_without_the_model() {
        #[derive(serde::Deserialize)]
        struct Case {
            text: String,
            pre: String,
            low: String,
            ct: String,
        }
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/prepro_cases.jsonl");
        let body = std::fs::read_to_string(&path).expect("read prepro fixtures");
        let mut n = 0usize;
        for (i, line) in body.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            let case: Case = serde_json::from_str(line)
                .unwrap_or_else(|e| panic!("{}:{}: {e}", path.display(), i + 1));
            let pre = preprocess(&case.text);
            assert_eq!(
                pre.iter().collect::<String>(),
                case.pre,
                "preprocess({:?})",
                case.text
            );
            let low = lower(&pre);
            assert_eq!(
                low.iter().collect::<String>(),
                case.low,
                "lower(preprocess({:?}))",
                case.text
            );
            let ct: String = low
                .iter()
                .map(|c| char::from_digit(chartype(*c), 10).expect("char type 0..=5"))
                .collect();
            assert_eq!(ct, case.ct, "char types of {:?}", case.text);
            n += 1;
        }
        assert!(n >= 400, "expected the full fixture, got {n} cases");
    }

    /// Every code point's NFKC and `str.lower()` must agree with the CPython
    /// reference interpreter (3.12 / unidata 15.0.0). The reference file is
    /// `<cp>\t<nfkc code points>\t<lower code points>` for every code point
    /// where either mapping is not the identity, generated by
    /// `docker run python:3.12-slim` (see tools/unicode.py).
    #[cfg_attr(
        not(have_unicode_ref),
        ignore = "needs the CPython 3.12 dump: set NAGISA_RS_UNICODE_REF (see README.md)"
    )]
    #[test]
    fn nfkc_and_lower_match_cpython312() {
        let path = std::env::var_os("NAGISA_RS_UNICODE_REF").expect("unicode reference");
        let body = std::fs::read_to_string(&path).expect("read unicode reference");
        let mut want: std::collections::HashMap<u32, (Vec<u32>, Vec<u32>)> =
            std::collections::HashMap::new();
        let cps = |f: &str| -> Vec<u32> {
            f.split(',')
                .map(|x| u32::from_str_radix(x, 16).expect("hex code point"))
                .collect()
        };
        for line in body.lines() {
            let mut it = line.split('\t');
            let (c, nf, lo) = (
                it.next().expect("cp"),
                it.next().expect("nfkc"),
                it.next().expect("lower"),
            );
            want.insert(
                u32::from_str_radix(c, 16).expect("hex cp"),
                (cps(nf), cps(lo)),
            );
        }
        let mut nfkc_bad = 0usize;
        let mut lower_bad = 0usize;
        for c in 0..0x11_0000u32 {
            let Some(ch) = char::from_u32(c) else {
                continue;
            };
            let (want_nfkc, want_lower) =
                want.get(&c).cloned().unwrap_or_else(|| (vec![c], vec![c]));
            let got_nfkc: Vec<u32> = crate::nfkc::nfkc(&[ch]).iter().map(|x| *x as u32).collect();
            if got_nfkc != want_nfkc {
                nfkc_bad += 1;
                if nfkc_bad <= 20 {
                    eprintln!("NFKC U+{c:04X}: cpython {want_nfkc:04X?} rust {got_nfkc:04X?}");
                }
            }
            // U+0130 is the one code point whose lowercase is two characters;
            // `preprocess` replaces it with `I` before `lower` ever sees it.
            if c == 0x0130 {
                assert_eq!(
                    want_lower.len(),
                    2,
                    "U+0130 should still be the 2-char case"
                );
                continue;
            }
            assert_eq!(want_lower.len(), 1, "U+{c:04X} lowercases to >1 char");
            let got_lower = lower(&[ch])[0] as u32;
            if got_lower != want_lower[0] {
                lower_bad += 1;
                if lower_bad <= 20 {
                    eprintln!(
                        "lower U+{c:04X}: cpython U+{:04X} rust U+{got_lower:04X}",
                        want_lower[0]
                    );
                }
            }
        }
        assert_eq!((nfkc_bad, lower_bad), (0, 0), "NFKC / lower divergences");
    }

    /// NFKC over combining sequences must agree with CPython 3.12 as well, not
    /// just over lone code points: composition and canonical ordering are where
    /// a wrong table shows up. `NAGISA_RS_UNICODE_SWEEPS` points at a
    /// directory holding the two reference dumps generated by
    /// `docker run python:3.12-slim` (see tools/unicode.py):
    /// * `py312_compose.tsv` -- every code point x every mark that is the
    ///   second element of a canonical decomposition (1 112 064 x 85).
    /// * `py312_order.tsv` -- `'a' + m1 + m2` for every pair of the 922 code
    ///   points with a non-zero combining class (850 084 pairs).
    ///
    /// Both list only the sequences NFKC changes.
    #[cfg_attr(
        not(have_unicode_sweeps),
        ignore = "needs the CPython 3.12 sweeps: set NAGISA_RS_UNICODE_SWEEPS (see README.md)"
    )]
    #[test]
    fn nfkc_combining_sequences_match_cpython312() {
        let dir = std::env::var_os("NAGISA_RS_UNICODE_SWEEPS").expect("unicode sweeps");
        let dir = std::path::PathBuf::from(dir);
        let read_cps = |name: &str| -> Vec<u32> {
            std::fs::read_to_string(dir.join(name))
                .expect("read mark list")
                .lines()
                .filter(|l| !l.trim().is_empty())
                .map(|l| u32::from_str_radix(l.trim(), 16).expect("hex"))
                .collect()
        };
        let second = read_cps("py312_second.txt");
        let marks = read_cps("py312_marks.txt");
        let fmt = |v: &[char]| -> String {
            v.iter()
                .map(|c| format!("{:X}", *c as u32))
                .collect::<Vec<_>>()
                .join(",")
        };

        let mut got = String::with_capacity(8 << 20);
        for c in 0..0x11_0000u32 {
            let Some(b) = char::from_u32(c) else { continue };
            for &m in &second {
                let mc = char::from_u32(m).expect("mark");
                let inp = [b, mc];
                let r = crate::nfkc::nfkc(&inp);
                if r != inp {
                    got.push_str(&format!("{c:X},{m:X}\t{}\n", fmt(&r)));
                }
            }
        }
        let want = std::fs::read_to_string(dir.join("py312_compose.tsv")).expect("read compose");
        assert_eq!(got.len(), want.len(), "compose sweep length");
        assert!(got == want, "compose sweep differs from CPython 3.12");

        let mut got = String::with_capacity(8 << 20);
        for &m1 in &marks {
            for &m2 in &marks {
                let inp = [
                    'a',
                    char::from_u32(m1).expect("mark"),
                    char::from_u32(m2).expect("mark"),
                ];
                let r = crate::nfkc::nfkc(&inp);
                if r != inp {
                    got.push_str(&format!("{m1:X},{m2:X}\t{}\n", fmt(&r)));
                }
            }
        }
        let want = std::fs::read_to_string(dir.join("py312_order.tsv")).expect("read order");
        assert_eq!(got.len(), want.len(), "order sweep length");
        assert!(got == want, "order sweep differs from CPython 3.12");
    }

    #[test]
    fn chartypes() {
        assert_eq!(chartype('あ'), 0);
        assert_eq!(chartype('ア'), 1);
        assert_eq!(chartype('漢'), 2);
        assert_eq!(chartype('a'), 3);
        assert_eq!(chartype('7'), 4);
        assert_eq!(chartype('。'), 5);
        assert_eq!(chartype('\u{3000}'), 5);
        assert_eq!(chartype('ー'), 5);
    }
}
