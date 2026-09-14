//! A minimal reader for the pickle stream nagisa ships its vocabularies in.
//!
//! `nagisa_v001.dict` is `cPickle.dump(data, gzip_file, protocol=2)` of the
//! 5-element list `[uni2id, bi2id, word2id, pos2id, word2postags]`. Only the
//! first three are needed for word segmentation, and the fourth and fifth hold
//! ~569 k entries between them, so the reader stops as soon as the third dict
//! is closed (i.e. when the fourth `EMPTY_DICT` shows up). That keeps both the
//! load time and the resident memory down without any conversion step: the
//! file nagisa installs is read as-is.
//!
//! Opcodes present in the file (verified with `pickletools.genops`):
//! PROTO EMPTY_LIST EMPTY_DICT MARK BINUNICODE BINPUT LONG_BINPUT BINGET
//! LONG_BINGET BININT BININT1 BININT2 APPEND APPENDS SETITEMS STOP.

use crate::error::JaError;
use crate::hash::FxMap;

const PROTO: u8 = 0x80;
const EMPTY_LIST: u8 = b']';
const EMPTY_DICT: u8 = b'}';
const MARK: u8 = b'(';
const BINUNICODE: u8 = b'X';
const SHORT_BINUNICODE: u8 = 0x8c;
const BINPUT: u8 = b'q';
const LONG_BINPUT: u8 = b'r';
const MEMOIZE: u8 = 0x94;
const BINGET: u8 = b'h';
const LONG_BINGET: u8 = b'j';
const BININT: u8 = b'J';
const BININT1: u8 = b'K';
const BININT2: u8 = b'M';
const APPEND: u8 = b'a';
const APPENDS: u8 = b'e';
const SETITEM: u8 = b's';
const SETITEMS: u8 = b'u';
const TUPLE: u8 = b't';
const STOP: u8 = b'.';

#[derive(Clone, Copy, PartialEq, Eq)]
enum Item {
    Mark,
    Str(u32),
    Int(i64),
    Dict(u32),
    List,
}

/// The three vocabularies word segmentation needs, in file order.
pub(crate) struct Vocabs {
    pub uni2id: FxMap<String, u32>,
    pub bi2id: FxMap<String, u32>,
    pub word2id: FxMap<String, u32>,
}

struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn byte(&mut self) -> Result<u8, JaError> {
        let b = *self.buf.get(self.pos).ok_or(JaError::PickleTruncated)?;
        self.pos += 1;
        Ok(b)
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], JaError> {
        let end = self.pos.checked_add(n).ok_or(JaError::PickleTruncated)?;
        let s = self
            .buf
            .get(self.pos..end)
            .ok_or(JaError::PickleTruncated)?;
        self.pos = end;
        Ok(s)
    }

    fn u32le(&mut self) -> Result<u32, JaError> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
}

/// Parse the gzip-inflated pickle stream and return the first three dicts.
pub(crate) fn parse_vocabs(buf: &[u8]) -> Result<Vocabs, JaError> {
    let mut r = Reader { buf, pos: 0 };
    let mut stack: Vec<Item> = Vec::with_capacity(64);
    let mut memo: Vec<Item> = Vec::new();
    let mut strings: Vec<String> = Vec::new();
    let mut dicts: Vec<FxMap<String, u32>> = Vec::new();

    let put = |memo: &mut Vec<Item>, idx: usize, v: Item| {
        if memo.len() <= idx {
            memo.resize(idx + 1, Item::Mark);
        }
        memo[idx] = v;
    };

    loop {
        let op = r.byte()?;
        match op {
            PROTO => {
                r.byte()?;
            }
            EMPTY_LIST => stack.push(Item::List),
            EMPTY_DICT => {
                if dicts.len() == 3 {
                    // uni2id / bi2id / word2id are complete; pos2id and
                    // word2postags are not used by the segmenter.
                    break;
                }
                dicts.push(FxMap::default());
                let id = u32::try_from(dicts.len() - 1).map_err(|_| JaError::PickleBadStream)?;
                stack.push(Item::Dict(id));
            }
            MARK => stack.push(Item::Mark),
            BINUNICODE | SHORT_BINUNICODE => {
                let n = if op == BINUNICODE {
                    r.u32le()? as usize
                } else {
                    usize::from(r.byte()?)
                };
                let bytes = r.take(n)?;
                let s = std::str::from_utf8(bytes).map_err(|_| JaError::PickleBadStream)?;
                strings.push(s.to_owned());
                let id = u32::try_from(strings.len() - 1).map_err(|_| JaError::PickleBadStream)?;
                stack.push(Item::Str(id));
            }
            BINPUT | LONG_BINPUT => {
                let idx = if op == BINPUT {
                    usize::from(r.byte()?)
                } else {
                    r.u32le()? as usize
                };
                let top = *stack.last().ok_or(JaError::PickleBadStream)?;
                put(&mut memo, idx, top);
            }
            MEMOIZE => {
                let top = *stack.last().ok_or(JaError::PickleBadStream)?;
                memo.push(top);
            }
            BINGET | LONG_BINGET => {
                let idx = if op == BINGET {
                    usize::from(r.byte()?)
                } else {
                    r.u32le()? as usize
                };
                let v = *memo.get(idx).ok_or(JaError::PickleBadStream)?;
                stack.push(v);
            }
            BININT => {
                let v = r.u32le()? as i32;
                stack.push(Item::Int(i64::from(v)));
            }
            BININT1 => {
                let v = r.byte()?;
                stack.push(Item::Int(i64::from(v)));
            }
            BININT2 => {
                let b = r.take(2)?;
                stack.push(Item::Int(i64::from(u16::from_le_bytes([b[0], b[1]]))));
            }
            APPEND => {
                stack.pop().ok_or(JaError::PickleBadStream)?;
            }
            APPENDS | TUPLE => {
                while stack.pop().ok_or(JaError::PickleBadStream)? != Item::Mark {}
                if op == TUPLE {
                    stack.push(Item::List);
                }
            }
            SETITEM => {
                let v = stack.pop().ok_or(JaError::PickleBadStream)?;
                let k = stack.pop().ok_or(JaError::PickleBadStream)?;
                let d = *stack.last().ok_or(JaError::PickleBadStream)?;
                set_item(&mut dicts, &strings, d, k, v)?;
            }
            SETITEMS => {
                let mark = stack
                    .iter()
                    .rposition(|i| *i == Item::Mark)
                    .ok_or(JaError::PickleBadStream)?;
                let items: Vec<Item> = stack.split_off(mark + 1);
                stack.pop();
                let d = *stack.last().ok_or(JaError::PickleBadStream)?;
                if !items.len().is_multiple_of(2) {
                    return Err(JaError::PickleBadStream);
                }
                for pair in items.as_chunks::<2>().0 {
                    set_item(&mut dicts, &strings, d, pair[0], pair[1])?;
                }
            }
            STOP => break,
            other => return Err(JaError::PickleOpcode(other)),
        }
    }

    if dicts.len() < 3 {
        return Err(JaError::PickleBadStream);
    }
    let word2id = dicts.pop().unwrap_or_default();
    let bi2id = dicts.pop().unwrap_or_default();
    let uni2id = dicts.pop().unwrap_or_default();
    Ok(Vocabs {
        uni2id,
        bi2id,
        word2id,
    })
}

fn set_item(
    dicts: &mut [FxMap<String, u32>],
    strings: &[String],
    target: Item,
    k: Item,
    v: Item,
) -> Result<(), JaError> {
    let Item::Dict(d) = target else {
        return Err(JaError::PickleBadStream);
    };
    let (Item::Str(ks), Item::Int(vi)) = (k, v) else {
        // pos2id / word2postags shapes never reach here, but a value that is
        // not a plain int (e.g. a list) simply is not a vocabulary entry.
        return Err(JaError::PickleBadStream);
    };
    let key = strings.get(ks as usize).ok_or(JaError::PickleBadStream)?;
    let id = u32::try_from(vi).map_err(|_| JaError::PickleBadStream)?;
    let dict = dicts.get_mut(d as usize).ok_or(JaError::PickleBadStream)?;
    dict.insert(key.clone(), id);
    Ok(())
}
