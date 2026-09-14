//! The word-segmentation half of nagisa's model, in f32.
//!
//! `nagisa/model.py::Model.encode_ws` builds, for every character position,
//! the 200-dim vector
//!
//! ```text
//! concat( UNI[u_{i-1..i+1}] , BI[b_{i-1..i+1}] , CTYPE[t_{i-1..i+1}] ,
//!         sum WORD[words starting at i] , sum WORD[words ending at i] )
//! ```
//!
//! runs it through a one-layer bidirectional `dy.LSTMBuilder` (which is
//! `VanillaLSTMBuilder`: gate order i, f, o, g and a constant +1 on the forget
//! gate pre-activation, both confirmed against DyNet numerically), projects the
//! 100-dim output with `w_ws * h + b_ws`, and decodes with the CRF viterbi in
//! `nagisa_utils.np_viterbi` -- which runs in float64 because `npvalue()`
//! widens DyNet's float32 to float64.
//!
//! Matrix-vector products accumulate column by column, which is the order
//! Eigen uses for a column-major matrix and therefore the order DyNet's CPU
//! backend uses.

use crate::hash::FxMap;

pub(crate) const WINDOW: usize = 3;
pub(crate) const DIM_UNI: usize = 32;
pub(crate) const DIM_BI: usize = 16;
pub(crate) const DIM_CTYPE: usize = 8;
pub(crate) const DIM_WORD: usize = 16;
pub(crate) const DIM_INPUT: usize = (DIM_UNI + DIM_BI + DIM_CTYPE) * WINDOW + DIM_WORD * 2;
pub(crate) const DIM_HIDDEN: usize = 100;
pub(crate) const DIM_DIR: usize = DIM_HIDDEN / 2;
pub(crate) const DIM_OUT: usize = 6;
pub(crate) const N_CTYPE: usize = 7;
pub(crate) const CTYPE_PAD: usize = 6;
pub(crate) const MAX_WORD: usize = 8;
const SP_S: usize = 4;
/// Register tile for [`precompute_x2i`]: `RB` gate rows x `TB` steps.
/// `RB` divides 4*DIM_DIR == 200 and `RB * TB` fits in the NEON/AVX register file.
const RB: usize = 20;
const TB: usize = 4;
const SP_E: usize = 5;

/// One direction of the segmentation BiLSTM.
pub(crate) struct Lstm {
    /// `{4h, input}` column-major.
    pub wx: Vec<f32>,
    /// `{4h, h}` column-major.
    pub wh: Vec<f32>,
    /// `{4h}`.
    pub b: Vec<f32>,
}

/// The segmentation model weights.
pub(crate) struct Weights {
    pub uni: Vec<f32>,
    pub bi: Vec<f32>,
    pub word: Vec<f32>,
    pub ctype: Vec<f32>,
    pub fwd: Lstm,
    pub bwd: Lstm,
    /// `{6, 100}` column-major.
    pub w_ws: Vec<f32>,
    pub b_ws: Vec<f32>,
    /// `trans[next][prev]`, widened to f64 like `npvalue()` does.
    pub trans: [[f64; DIM_OUT]; DIM_OUT],
}

/// The three vocabularies plus their OOV ids.
pub(crate) struct Vocab {
    pub uni2id: FxMap<String, u32>,
    pub bi2id: FxMap<String, u32>,
    pub word2id: FxMap<String, u32>,
    pub uni_oov: u32,
    pub bi_oov: u32,
    pub word_oov: u32,
}

/// `res += m * x` for a column-major `m` with `res.len()` rows.
#[inline]
pub(crate) fn gemv_acc(res: &mut [f32], m: &[f32], x: &[f32]) {
    let rows = res.len();
    for (j, &xj) in x.iter().enumerate() {
        let col = &m[j * rows..j * rows + rows];
        for (r, &c) in res.iter_mut().zip(col) {
            *r += xj * c;
        }
    }
}

/// `res += m * x`, same column order as [`gemv_acc`] but with the destination
/// held in registers over the reduction. Requires `res.len() % RB == 0`.
#[inline]
fn gemv_acc_tiled(res: &mut [f32], m: &[f32], x: &[f32]) {
    let rows = res.len();
    for r0 in (0..rows).step_by(RB) {
        let mut acc = [0.0f32; RB];
        acc.copy_from_slice(&res[r0..r0 + RB]);
        for (j, &xj) in x.iter().enumerate() {
            let col = &m[j * rows + r0..j * rows + r0 + RB];
            for (a, &c) in acc.iter_mut().zip(col) {
                *a += xj * c;
            }
        }
        res[r0..r0 + RB].copy_from_slice(&acc);
    }
}

#[inline]
pub(crate) fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// `b + Wx * x_t` for every step at once.
///
/// The input-to-gate product does not depend on the recurrent state, so it can
/// be hoisted out of the sequential loop and tiled. Each output element still
/// accumulates the 200 input dimensions in ascending column order starting from
/// the bias -- exactly the order a per-step column-major `gemv` would use, and
/// therefore bit-identical -- but a `RB x TB` tile of accumulators stays in
/// registers across the whole reduction, so the 160 KB weight matrix is read
/// once per row stripe instead of once per step.
fn precompute_x2i(lstm: &Lstm, xs: &[f32], steps: usize, pre: &mut [f32]) {
    let h4 = DIM_DIR * 4;
    let full = steps - steps % TB;
    for r0 in (0..h4).step_by(RB) {
        let bias: &[f32] = &lstm.b[r0..r0 + RB];
        for t0 in (0..full).step_by(TB) {
            let mut acc = [[0.0f32; RB]; TB];
            for a in &mut acc {
                a.copy_from_slice(bias);
            }
            for j in 0..DIM_INPUT {
                let col = &lstm.wx[j * h4 + r0..j * h4 + r0 + RB];
                let xrow = &xs[t0 * DIM_INPUT + j..];
                for (tt, a) in acc.iter_mut().enumerate() {
                    let xj = xrow[tt * DIM_INPUT];
                    for (dst, &c) in a.iter_mut().zip(col) {
                        *dst += xj * c;
                    }
                }
            }
            for (tt, a) in acc.iter().enumerate() {
                let base = (t0 + tt) * h4 + r0;
                pre[base..base + RB].copy_from_slice(a);
            }
        }
    }
    for t in full..steps {
        let row = &mut pre[t * h4..(t + 1) * h4];
        row.copy_from_slice(&lstm.b);
        gemv_acc(row, &lstm.wx, &xs[t * DIM_INPUT..(t + 1) * DIM_INPUT]);
    }
}

/// Run one direction over `xs` (already in the order to be consumed) and write
/// each step's hidden state into `out[step * DIM_HIDDEN + off ..][..DIM_DIR]`.
fn run_lstm(lstm: &Lstm, xs: &[f32], steps: usize, out: &mut [f32], off: usize, pre: &mut [f32]) {
    let h4 = DIM_DIR * 4;
    precompute_x2i(lstm, xs, steps, pre);
    let mut g = [0.0f32; DIM_DIR * 4];
    let mut c = [0.0f32; DIM_DIR];
    let mut h = [0.0f32; DIM_DIR];
    for t in 0..steps {
        g.copy_from_slice(&pre[t * h4..(t + 1) * h4]);
        if t > 0 {
            gemv_acc_tiled(&mut g, &lstm.wh, &h);
        }
        for k in 0..DIM_DIR {
            let i = sigmoid(g[k]);
            let f = sigmoid(g[DIM_DIR + k] + 1.0);
            let o = sigmoid(g[2 * DIM_DIR + k]);
            let u = g[3 * DIM_DIR + k].tanh();
            c[k] = f * c[k] + i * u;
            h[k] = o * c[k].tanh();
        }
        let row = if off == 0 { t } else { steps - 1 - t };
        out[row * DIM_HIDDEN + off..row * DIM_HIDDEN + off + DIM_DIR].copy_from_slice(&h);
    }
}

/// Character-position features, mirroring `utils.feature_extraction`.
struct Feats {
    uids: Vec<u32>,
    bids: Vec<u32>,
    cids: Vec<u32>,
    starts: Vec<Vec<u32>>,
    ends: Vec<Vec<u32>>,
}

fn extract(vocab: &Vocab, low: &str, offs: &[usize], chars: &[char]) -> Feats {
    let n = chars.len();
    let uni_of = |i: usize| -> u32 {
        *vocab
            .uni2id
            .get(&low[offs[i]..offs[i + 1]])
            .unwrap_or(&vocab.uni_oov)
    };
    let bi_of = |i: usize| -> u32 {
        if i + 1 < n {
            *vocab
                .bi2id
                .get(&low[offs[i]..offs[i + 2]])
                .unwrap_or(&vocab.bi_oov)
        } else {
            let mut key = String::with_capacity(offs[n] - offs[n - 1] + 3);
            key.push_str(&low[offs[n - 1]..offs[n]]);
            key.push_str("<E>");
            *vocab.bi2id.get(&key).unwrap_or(&vocab.bi_oov)
        }
    };

    let mut uids = Vec::with_capacity(n * WINDOW);
    let mut bids = Vec::with_capacity(n * WINDOW);
    let mut cids = Vec::with_capacity(n * WINDOW);
    let types: Vec<u32> = chars.iter().map(|c| crate::prepro::chartype(*c)).collect();
    for i in 0..n {
        for d in 0..WINDOW {
            // context_window pads with id 1 ('pad'), and 6 for the char type.
            let k = (i + d) as isize - (WINDOW / 2) as isize;
            if k < 0 || k as usize >= n {
                uids.push(1);
                bids.push(1);
                cids.push(CTYPE_PAD as u32);
            } else {
                let k = k as usize;
                uids.push(uni_of(k));
                bids.push(bi_of(k));
                cids.push(types[k]);
            }
        }
    }

    let mut starts = Vec::with_capacity(n);
    let mut ends = Vec::with_capacity(n);
    for i in 0..n {
        let mut s = Vec::new();
        for j in i..n.min(i + MAX_WORD) {
            if let Some(id) = vocab.word2id.get(&low[offs[i]..offs[j + 1]]) {
                s.push(*id);
            }
        }
        if s.is_empty() {
            s.push(vocab.word_oov);
        }
        starts.push(s);

        let mut e = Vec::new();
        for len in 1..=MAX_WORD.min(i + 1) {
            if let Some(id) = vocab.word2id.get(&low[offs[i + 1 - len]..offs[i + 1]]) {
                e.push(*id);
            }
        }
        if e.is_empty() {
            e.push(vocab.word_oov);
        }
        ends.push(e);
    }

    Feats {
        uids,
        bids,
        cids,
        starts,
        ends,
    }
}

fn build_inputs(w: &Weights, f: &Feats, n: usize) -> Vec<f32> {
    let mut xs = vec![0.0f32; n * DIM_INPUT];
    for i in 0..n {
        let x = &mut xs[i * DIM_INPUT..(i + 1) * DIM_INPUT];
        let mut at = 0;
        for d in 0..WINDOW {
            let id = f.uids[i * WINDOW + d] as usize;
            x[at..at + DIM_UNI].copy_from_slice(&w.uni[id * DIM_UNI..id * DIM_UNI + DIM_UNI]);
            at += DIM_UNI;
        }
        for d in 0..WINDOW {
            let id = f.bids[i * WINDOW + d] as usize;
            x[at..at + DIM_BI].copy_from_slice(&w.bi[id * DIM_BI..id * DIM_BI + DIM_BI]);
            at += DIM_BI;
        }
        for d in 0..WINDOW {
            let id = f.cids[i * WINDOW + d] as usize;
            x[at..at + DIM_CTYPE]
                .copy_from_slice(&w.ctype[id * DIM_CTYPE..id * DIM_CTYPE + DIM_CTYPE]);
            at += DIM_CTYPE;
        }
        for ids in [&f.starts[i], &f.ends[i]] {
            for &id in ids.iter() {
                let id = id as usize;
                let emb = &w.word[id * DIM_WORD..id * DIM_WORD + DIM_WORD];
                for (dst, &src) in x[at..at + DIM_WORD].iter_mut().zip(emb) {
                    *dst += src;
                }
            }
            at += DIM_WORD;
        }
    }
    xs
}

/// Score every position, i.e. `Model.encode_ws` widened to f64 like
/// `npvalue()` does.
fn observations(w: &Weights, xs: &[f32], n: usize) -> Vec<[f64; DIM_OUT]> {
    let mut hs = vec![0.0f32; n * DIM_HIDDEN];
    let mut pre = vec![0.0f32; n * DIM_DIR * 4];
    run_lstm(&w.fwd, xs, n, &mut hs, 0, &mut pre);
    let mut rev = vec![0.0f32; n * DIM_INPUT];
    for t in 0..n {
        rev[t * DIM_INPUT..(t + 1) * DIM_INPUT]
            .copy_from_slice(&xs[(n - 1 - t) * DIM_INPUT..(n - t) * DIM_INPUT]);
    }
    run_lstm(&w.bwd, &rev, n, &mut hs, DIM_DIR, &mut pre);

    let mut obs = Vec::with_capacity(n);
    let mut o = [0.0f32; DIM_OUT];
    for i in 0..n {
        o.fill(0.0);
        gemv_acc(&mut o, &w.w_ws, &hs[i * DIM_HIDDEN..(i + 1) * DIM_HIDDEN]);
        let mut row = [0.0f64; DIM_OUT];
        for k in 0..DIM_OUT {
            row[k] = f64::from(o[k] + w.b_ws[k]);
        }
        obs.push(row);
    }
    obs
}

/// `nagisa_utils.np_viterbi`, including numpy's first-wins argmax.
fn viterbi(trans: &[[f64; DIM_OUT]; DIM_OUT], obs: &[[f64; DIM_OUT]]) -> Vec<usize> {
    let mut cur = [-1e10f64; DIM_OUT];
    cur[SP_S] = 0.0;
    let mut back: Vec<[usize; DIM_OUT]> = Vec::with_capacity(obs.len());
    for ob in obs {
        let mut bptrs = [0usize; DIM_OUT];
        let mut next = [0.0f64; DIM_OUT];
        for idx in 0..DIM_OUT {
            let mut best = 0usize;
            let mut best_v = cur[0] + trans[idx][0];
            for p in 1..DIM_OUT {
                let v = cur[p] + trans[idx][p];
                if v > best_v {
                    best_v = v;
                    best = p;
                }
            }
            bptrs[idx] = best;
            next[idx] = best_v;
        }
        for k in 0..DIM_OUT {
            cur[k] = next[k] + ob[k];
        }
        back.push(bptrs);
    }

    let mut best = 0usize;
    let mut best_v = cur[0] + trans[SP_E][0];
    for p in 1..DIM_OUT {
        let v = cur[p] + trans[SP_E][p];
        if v > best_v {
            best_v = v;
            best = p;
        }
    }
    let mut path = Vec::with_capacity(back.len() + 1);
    path.push(best);
    for bptrs in back.iter().rev() {
        best = bptrs[best];
        path.push(best);
    }
    path.pop();
    path.reverse();
    path
}

/// `utils.segmenter_for_bmes`: 0=B, 1=M, 2=E, 3=S. A trailing B/M run is
/// dropped, exactly as nagisa drops it.
fn segment(chars: &[char], tags: &[usize]) -> Vec<String> {
    let mut words = Vec::new();
    let mut partial = String::new();
    for (&ch, &tag) in chars.iter().zip(tags) {
        match tag {
            3 => words.push(ch.to_string()),
            2 => {
                partial.push(ch);
                words.push(std::mem::take(&mut partial));
            }
            _ => partial.push(ch),
        }
    }
    words
}

/// Full `Tagger.wakati` for an already-preprocessed character sequence.
pub(crate) fn wakati(
    vocab: &Vocab,
    w: &Weights,
    chars: &[char],
    lower_output: bool,
    dictionary: &crate::dictionary::Dictionary,
) -> Vec<String> {
    if chars.is_empty() {
        return Vec::new();
    }
    let lowered = crate::prepro::lower(chars);
    let mut low = String::with_capacity(lowered.len() * 3);
    let mut offs = Vec::with_capacity(lowered.len() + 1);
    for c in &lowered {
        offs.push(low.len());
        low.push(*c);
    }
    offs.push(low.len());

    let feats = extract(vocab, &low, &offs, &lowered);
    let xs = build_inputs(w, &feats, chars.len());
    let obs = observations(w, &xs, chars.len());
    let mut tags = viterbi(&w.trans, &obs);
    dictionary.apply(chars, &mut tags);
    segment(if lower_output { &lowered } else { chars }, &tags)
}
