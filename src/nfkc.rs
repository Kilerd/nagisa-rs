//! NFKC, driven by tables generated from the CPython that runs nagisa.
//!
//! `utils.preprocess` calls `unicodedata.normalize('NFKC', text)`. These tables
//! preserve CPython 3.12 / Unicode 15.0.0 behavior independently of Rust's
//! Unicode version: a different normalized character can change vocabulary
//! features and therefore word boundaries.
//!
//! The algorithm is UAX #15: full compatibility decomposition (recursive
//! mappings are pre-expanded in the table, Hangul is algorithmic), canonical
//! ordering, then canonical composition.
//!
//! Verified against CPython 3.12.14 with three exhaustive sweeps (see the
//! crate's `nfkc_and_lower_match_cpython312` test and tools/unicode.py):
//! all 1 112 064 Unicode scalar values, all 1 112 064 x 85 (code point, composing mark)
//! pairs, and all 922 x 922 (mark, mark) orderings after a starter.

use crate::nfkd_tables::{CCC_RANGES, COMPOSE, NFKD_DATA, NFKD_KEYS, NFKD_OFFSETS};

const S_BASE: u32 = 0xAC00;
const L_BASE: u32 = 0x1100;
const V_BASE: u32 = 0x1161;
const T_BASE: u32 = 0x11A7;
const L_COUNT: u32 = 19;
const V_COUNT: u32 = 21;
const T_COUNT: u32 = 28;
const N_COUNT: u32 = V_COUNT * T_COUNT;
const S_COUNT: u32 = L_COUNT * N_COUNT;

/// Canonical combining class.
fn ccc(c: u32) -> u8 {
    CCC_RANGES
        .binary_search_by(|&(lo, hi, _)| {
            if c < lo {
                std::cmp::Ordering::Greater
            } else if c > hi {
                std::cmp::Ordering::Less
            } else {
                std::cmp::Ordering::Equal
            }
        })
        .map_or(0, |i| CCC_RANGES[i].2)
}

fn decompose_into(c: char, out: &mut Vec<char>) {
    let u = c as u32;
    if (S_BASE..S_BASE + S_COUNT).contains(&u) {
        let s = u - S_BASE;
        out.extend(char::from_u32(L_BASE + s / N_COUNT));
        out.extend(char::from_u32(V_BASE + (s % N_COUNT) / T_COUNT));
        if !s.is_multiple_of(T_COUNT) {
            out.extend(char::from_u32(T_BASE + s % T_COUNT));
        }
        return;
    }
    match NFKD_KEYS.binary_search(&u) {
        Ok(i) => {
            let (a, b) = (NFKD_OFFSETS[i] as usize, NFKD_OFFSETS[i + 1] as usize);
            out.extend(NFKD_DATA[a..b].iter().filter_map(|&d| char::from_u32(d)));
        }
        Err(_) => out.push(c),
    }
}

/// Stable sort of each non-starter run by combining class.
fn canonical_order(buf: &mut [char]) {
    for i in 1..buf.len() {
        let cc = ccc(buf[i] as u32);
        if cc == 0 {
            continue;
        }
        let mut j = i;
        while j > 0 {
            let prev = ccc(buf[j - 1] as u32);
            if prev == 0 || prev <= cc {
                break;
            }
            buf.swap(j - 1, j);
            j -= 1;
        }
    }
}

fn compose_pair(a: u32, b: u32) -> Option<char> {
    if (L_BASE..L_BASE + L_COUNT).contains(&a) && (V_BASE..V_BASE + V_COUNT).contains(&b) {
        return char::from_u32(S_BASE + ((a - L_BASE) * V_COUNT + (b - V_BASE)) * T_COUNT);
    }
    if (S_BASE..S_BASE + S_COUNT).contains(&a)
        && (a - S_BASE).is_multiple_of(T_COUNT)
        && (T_BASE + 1..T_BASE + T_COUNT).contains(&b)
    {
        return char::from_u32(a + (b - T_BASE));
    }
    let key = (u64::from(a) << 32) | u64::from(b);
    COMPOSE
        .binary_search_by_key(&key, |&(k, _)| k)
        .ok()
        .and_then(|i| char::from_u32(COMPOSE[i].1))
}

/// `unicodedata.normalize('NFKC', ...)`.
pub(crate) fn nfkc(chars: &[char]) -> Vec<char> {
    if chars.is_empty() {
        return Vec::new();
    }
    let mut buf: Vec<char> = Vec::with_capacity(chars.len() + 8);
    for &c in chars {
        decompose_into(c, &mut buf);
    }
    canonical_order(&mut buf);

    let mut out: Vec<char> = Vec::with_capacity(buf.len());
    out.push(buf[0]);
    let mut starter_pos = 0usize;
    let mut starter_ch = buf[0];
    let mut last_class = i32::from(ccc(buf[0] as u32));
    if last_class != 0 {
        last_class = 256;
    }
    for &ch in &buf[1..] {
        let ch_class = i32::from(ccc(ch as u32));
        if (last_class < ch_class || last_class == 0)
            && let Some(comp) = compose_pair(starter_ch as u32, ch as u32)
        {
            out[starter_pos] = comp;
            starter_ch = comp;
            continue;
        }
        if ch_class == 0 {
            starter_pos = out.len();
            starter_ch = ch;
        }
        last_class = ch_class;
        out.push(ch);
    }
    out
}
