//! nagisa 0.2.11 POS inference: word/character/tag features → BiLSTM → softmax.
use crate::infer::{DIM_UNI, DIM_WORD, Lstm, Vocab, Weights, gemv_acc, sigmoid};

pub(crate) const N_TAGS: usize = 24;
pub(crate) const DIM_TAG: usize = 16;
const INPUT: usize = DIM_WORD + DIM_UNI + DIM_TAG;
const HIDDEN: usize = 50;
const CHAR_HIDDEN: usize = DIM_UNI / 2;

pub(crate) struct PosWeights {
    pub tags: Vec<f32>,
    pub char_fwd: Lstm,
    pub char_bwd: Lstm,
    pub fwd: Lstm,
    pub bwd: Lstm,
    pub output: Vec<f32>,
    pub bias: Vec<f32>,
}

/// CPython 3.12 integer-set iteration for POS IDs 0..24. The order matters
/// when summing f32 embeddings. Small sets can have collisions in eight
/// slots; after the first growth all these IDs occupy their own hash slot.
/// See CPython v3.12.14 Objects/setobject.c (insertion, resize and discard).
pub(crate) fn candidate_tags(ids: &[u32], noun: bool) -> Vec<u32> {
    const EMPTY: i32 = -1;
    const DUMMY: i32 = -2;
    fn insert(table: &mut Vec<i32>, used: &mut usize, fill: &mut usize, id: u32) {
        let mask = table.len() - 1;
        let mut at = id as usize & mask;
        let mut perturb = id as usize;
        let mut free = None;
        loop {
            let probes = if at + 9 <= mask { 9 } else { 0 };
            for slot in at..=at + probes {
                if table[slot] == id as i32 {
                    return;
                }
                if table[slot] == DUMMY {
                    free = Some(slot);
                }
                if table[slot] == EMPTY {
                    table[free.unwrap_or(slot)] = id as i32;
                    *used += 1;
                    if free.is_some() {
                        return;
                    }
                    *fill += 1;
                    if *fill * 5 >= mask * 3 {
                        let size = (*used * 4 + 1).next_power_of_two().max(8);
                        let mut grown = vec![EMPTY; size];
                        for &key in table.iter().filter(|&&key| key >= 0) {
                            grown[key as usize] = key;
                        }
                        *table = grown;
                        *fill = *used;
                    }
                    return;
                }
            }
            perturb >>= 5;
            at = (at * 5 + 1 + perturb) & mask;
        }
    }
    let mut table = vec![EMPTY; 8];
    let (mut used, mut fill) = (0, 0);
    for &id in ids {
        insert(&mut table, &mut used, &mut fill, id);
    }
    if noun {
        if let Some(at) = table.iter().position(|&id| id == 0) {
            table[at] = DUMMY;
            used -= 1;
        }
        insert(&mut table, &mut used, &mut fill, 2);
    }
    table
        .into_iter()
        .filter(|&id| id >= 0)
        .map(|id| id as u32)
        .collect()
}

pub(crate) fn is_alnum(word: &str) -> bool {
    !word.is_empty()
        && word.chars().all(|ch| {
            let cp = ch as u32;
            crate::pos_unicode_tables::ALNUM
                .binary_search_by(|&(lo, hi)| {
                    if hi < cp {
                        std::cmp::Ordering::Less
                    } else if lo > cp {
                        std::cmp::Ordering::Greater
                    } else {
                        std::cmp::Ordering::Equal
                    }
                })
                .is_ok()
        })
}

/// A direction's states, stored in the original token order.
fn run(lstm: &Lstm, xs: &[f32], input: usize, hidden: usize, reverse: bool) -> Vec<f32> {
    let steps = xs.len() / input;
    let mut output = vec![0.0; steps * hidden];
    let mut h = vec![0.0; hidden];
    let mut c = vec![0.0; hidden];
    let mut gates = vec![0.0; hidden * 4];
    for step in 0..steps {
        let index = if reverse { steps - 1 - step } else { step };
        gates.copy_from_slice(&lstm.b);
        gemv_acc(
            &mut gates,
            &lstm.wx,
            &xs[index * input..(index + 1) * input],
        );
        if step > 0 {
            gemv_acc(&mut gates, &lstm.wh, &h);
        }
        for k in 0..hidden {
            let i = sigmoid(gates[k]);
            let f = sigmoid(gates[hidden + k] + 1.0);
            let o = sigmoid(gates[2 * hidden + k]);
            let u = gates[3 * hidden + k].tanh();
            c[k] = f * c[k] + i * u;
            h[k] = o * c[k].tanh();
        }
        output[index * hidden..(index + 1) * hidden].copy_from_slice(&h);
    }
    output
}

pub(crate) fn infer(
    vocab: &Vocab,
    shared: &Weights,
    pos: &PosWeights,
    dictionary: &crate::hash::FxMap<String, Vec<u32>>,
    words: &[String],
) -> Vec<usize> {
    let mut inputs = Vec::with_capacity(words.len() * INPUT);
    for word in words {
        // POS features preserve the token's case; only segmentation uses lower().
        let wid = *vocab.word2id.get(word).unwrap_or(&vocab.word_oov) as usize;
        if wid == 0 {
            inputs.extend_from_slice(&[0.0; DIM_WORD]);
        } else {
            inputs.extend_from_slice(&shared.word[wid * DIM_WORD..(wid + 1) * DIM_WORD]);
        }
        let mut chars = Vec::with_capacity(word.chars().count() * DIM_UNI);
        for ch in word.chars() {
            let id = *vocab.uni2id.get(&ch.to_string()).unwrap_or(&vocab.uni_oov) as usize;
            chars.extend_from_slice(&shared.uni[id * DIM_UNI..(id + 1) * DIM_UNI]);
        }
        let forward = run(&pos.char_fwd, &chars, DIM_UNI, CHAR_HIDDEN, false);
        // transduce(chars)[-1] means the last ORIGINAL position. Its backward
        // half has consumed just the final character, not the whole word.
        let backward = run(
            &pos.char_bwd,
            &chars[chars.len() - DIM_UNI..],
            DIM_UNI,
            CHAR_HIDDEN,
            true,
        );
        inputs.extend_from_slice(&forward[forward.len() - CHAR_HIDDEN..]);
        inputs.extend_from_slice(&backward);
        let ids = dictionary.get(word).map_or(&[0][..], Vec::as_slice);
        let mut tags = [0.0; DIM_TAG];
        for id in candidate_tags(ids, is_alnum(word)) {
            if id == 0 {
                continue;
            }
            for (dst, &value) in tags
                .iter_mut()
                .zip(&pos.tags[id as usize * DIM_TAG..][..DIM_TAG])
            {
                *dst += value;
            }
        }
        inputs.extend_from_slice(&tags);
    }
    let forward = run(&pos.fwd, &inputs, INPUT, HIDDEN, false);
    let backward = run(&pos.bwd, &inputs, INPUT, HIDDEN, true);
    let mut result = Vec::with_capacity(words.len());
    for i in 0..words.len() {
        let mut hidden = [0.0; HIDDEN * 2];
        hidden[..HIDDEN].copy_from_slice(&forward[i * HIDDEN..(i + 1) * HIDDEN]);
        hidden[HIDDEN..].copy_from_slice(&backward[i * HIDDEN..(i + 1) * HIDDEN]);
        let mut logits = [0.0; N_TAGS];
        gemv_acc(&mut logits, &pos.output, &hidden);
        for (logit, bias) in logits.iter_mut().zip(&pos.bias) {
            *logit += bias;
        }
        // softmax is monotone; use first-wins argmax, as numpy does.
        let mut best = 0;
        for j in 1..N_TAGS {
            if logits[j] > logits[best] {
                best = j;
            }
        }
        result.push(best);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;

    #[test]
    fn candidates_match_cpython312_set_order() {
        #[derive(Deserialize)]
        struct Case {
            tags: Vec<u32>,
            noun: bool,
            ordered: Vec<u32>,
        }
        let cases: Vec<Case> =
            serde_json::from_str(include_str!("../tests/fixtures/pos_candidates.json")).unwrap();
        assert!(cases.len() >= 5000);
        for case in cases {
            assert_eq!(
                candidate_tags(&case.tags, case.noun),
                case.ordered,
                "tags={:?}, noun={}",
                case.tags,
                case.noun
            );
        }
    }

    #[test]
    fn alnum_matches_python_noun_heuristic() {
        for word in ["猫", "Python", "123", "½", "𑼂"] {
            assert!(is_alnum(word), "{word:?}");
        }
        for word in ["", " ", "abc_", "ᾳ", "🙂", "\u{0345}", "\u{1c89}"] {
            assert!(!is_alnum(word), "{word:?}");
        }
    }
}
