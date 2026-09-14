//! Literal, longest-first, non-overlapping user dictionary matches.
use crate::hash::FxMap;

#[derive(Default)]
struct Node {
    edges: FxMap<char, usize>,
    terminal: bool,
}

#[derive(Default)]
pub(crate) struct Dictionary {
    nodes: Vec<Node>,
}

impl Dictionary {
    pub(crate) fn new<I, S>(words: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let mut dictionary = Self::default();
        for word in words {
            let word = word.as_ref();
            // Match nagisa's length check BEFORE normalization. For example,
            // "Ａ " is accepted, normalizes to "A", and forces a single tag.
            if word.chars().count() <= 1 {
                continue;
            }
            let chars = crate::prepro::preprocess(word);
            // An empty normalized entry cannot describe a word.
            if chars.is_empty() {
                continue;
            }
            if dictionary.nodes.is_empty() {
                dictionary.nodes.push(Node::default());
            }
            let mut node = 0;
            for ch in chars {
                let next = if let Some(&next) = dictionary.nodes[node].edges.get(&ch) {
                    next
                } else {
                    let next = dictionary.nodes.len();
                    dictionary.nodes.push(Node::default());
                    dictionary.nodes[node].edges.insert(ch, next);
                    next
                };
                node = next;
            }
            dictionary.nodes[node].terminal = true;
        }
        dictionary
    }

    pub(crate) fn apply(&self, chars: &[char], tags: &mut [usize]) {
        if self.nodes.is_empty() {
            return;
        }
        let mut start = 0;
        while start < chars.len() {
            let mut node = 0;
            let mut end = None;
            for (offset, ch) in chars[start..].iter().enumerate() {
                let Some(&next) = self.nodes[node].edges.get(ch) else {
                    break;
                };
                node = next;
                if self.nodes[node].terminal {
                    end = Some(start + offset + 1);
                }
            }
            let Some(end) = end else {
                start += 1;
                continue;
            };
            force_span(tags, start, end);
            start = end;
        }
    }
}

/// Replace BMES labels and repair the two adjacent boundaries as nagisa does.
fn force_span(tags: &mut [usize], start: usize, end: usize) {
    if end - start == 1 {
        tags[start] = 3;
    } else {
        tags[start..end].fill(1);
        tags[start] = 0;
        tags[end - 1] = 2;
    }
    if start > 0 {
        tags[start - 1] = match tags[start - 1] {
            0 => 3,
            1 => 2,
            tag => tag,
        };
    }
    if end < tags.len() {
        tags[end] = match tags[end] {
            1 => 0,
            2 => 3,
            tag => tag,
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn longest_leftmost_matches_and_adjacent_matches() {
        let dictionary = Dictionary::new(["東京", "東京都", "京都", "渋谷区"]);
        let mut tags = vec![3; 6];
        dictionary.apply(&"東京都渋谷区".chars().collect::<Vec<_>>(), &mut tags);
        assert_eq!(tags, [0, 1, 2, 0, 1, 2]);
    }

    #[test]
    fn repairs_partial_words_on_both_sides() {
        for (before, after, expected_before, expected_after) in [(0, 1, 3, 0), (1, 2, 2, 3)] {
            let mut tags = vec![before, 3, 3, after];
            force_span(&mut tags, 1, 3);
            assert_eq!(tags, [expected_before, 0, 2, expected_after]);
        }
    }

    #[test]
    fn normalizes_terms_and_treats_metacharacters_literally() {
        let dictionary = Dictionary::new(["", " ", "  ", "東", "Ａ ", "Ｃ＋＋", "(猫)"]);
        let text: Vec<_> = "AC++(猫)".chars().collect();
        let mut tags = vec![1; text.len()];
        dictionary.apply(&text, &mut tags);
        assert_eq!(tags, [3, 0, 1, 2, 0, 1, 2]);
        let mut tags = [1];
        dictionary.apply(&['東'], &mut tags);
        assert_eq!(tags, [1]); // A one-character original entry is ignored.
    }
}
