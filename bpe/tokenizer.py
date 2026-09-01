"""Char-level BPE tokenizer (Sennrich et al. 2016).

Built from scratch for learning.
"""

from collections import Counter

END_OF_WORD = "</w>"


def pretokenize(text: str) -> list[list[str]]:
    """Split raw text into per-word character sequences.

    Each word becomes a list of its characters plus an end-of-word marker.
    The marker keeps BPE merges from crossing word boundaries.

    Example:
        pretokenize("low lower")
        -> [['l', 'o', 'w', '</w>'],
            ['l', 'o', 'w', 'e', 'r', '</w>']]
    """
    return [list(word) + [END_OF_WORD] for word in text.split()]


def count_pairs(corpus: list[list[str]]) -> Counter[tuple[str, str]]:
    """Count how often each adjacent pair of tokens appears across the corpus.

    Each word contributes (len(word) - 1) pairs. If a word appears multiple
    times in the corpus, its pairs are counted multiple times.

    Example:
        count_pairs([['l', 'o', 'w', '</w>'],
                     ['l', 'o', 'w', 'e', 'r', '</w>']])
        -> Counter({('l', 'o'): 2, ('o', 'w'): 2, ('w', '</w>'): 1,
                    ('w', 'e'): 1, ('e', 'r'): 1, ('r', '</w>'): 1})
    """
    pair_counts: Counter[tuple[str, str]] = Counter()
    for word in corpus:
        for i in range(len(word) - 1):
            pair_counts[(word[i], word[i + 1])] += 1
    return pair_counts


def merge_word(word: list[str], pair: tuple[str, str]) -> list[str]:
    """Non-overlapping, left-to-right merge of `pair` inside a single word.

    Every adjacent occurrence of (a, b) is fused into the single token 'ab'.
    Overlapping matches are NOT counted twice — after a merge, we skip
    forward by two positions.

    Example:
        merge_word(['l', 'o', 'w', 'e', 'r', '</w>'], ('l', 'o'))
        -> ['lo', 'w', 'e', 'r', '</w>']

        merge_word(['a', 'a', 'a'], ('a', 'a'))
        -> ['aa', 'a']         # not ['aa', 'aa'] — no overlap
    """
    a, b = pair
    merged_token = a + b
    result: list[str] = []
    i = 0
    while i < len(word):
        if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
            result.append(merged_token)
            i += 2                # skip both tokens we just consumed
        else:
            result.append(word[i])
            i += 1
    return result


def apply_merge(
    corpus: list[list[str]], pair: tuple[str, str]
) -> list[list[str]]:
    """Return a new corpus with every occurrence of `pair` merged.

    Does not mutate the input. Delegates to `merge_word` per word.
    """
    return [merge_word(word, pair) for word in corpus]
