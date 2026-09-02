"""WordPiece tokenizer (BERT lineage).

Where BPE merges the most *frequent* pair, WordPiece merges the pair that
maximizes a likelihood score: freq(ab) / (freq(a) * freq(b)).  This is a
mutual-information-style criterion: it favours pairs whose constituents
have a strong association, regardless of how common each one is alone.

Where BPE encodes by replaying merges in rank order, WordPiece encodes by
greedy longest-prefix matching against the vocab.

Built from scratch for learning.
"""

from collections import Counter

from .pretokenizer import CONTINUATION_PREFIX, pretokenize

Corpus = dict[tuple[str, ...], int]


def build_corpus(text: str) -> Corpus:
    """Pretokenize `text` into a frequency-grouped corpus of WordPiece words.

    Each word is a tuple of character tokens with ## continuation prefixes.

    Example:
        build_corpus("low low")
        -> {('l', '##o', '##w'): 1,
            ('l', '##o', '##w'): 1}   # actually grouped:
        -> {('l', '##o', '##w'): 2}
    """
    corpus: Counter[tuple[str, ...]] = Counter()
    for word in pretokenize(text):
        corpus[tuple(word)] += 1
    return dict(corpus)


def count_pairs(corpus: Corpus) -> Counter[tuple[str, str]]:
    """Count adjacent token pairs across the corpus, weighted by frequency."""
    pair_counts: Counter[tuple[str, str]] = Counter()
    for word, freq in corpus.items():
        for i in range(len(word) - 1):
            pair_counts[(word[i], word[i + 1])] += freq
    return pair_counts


def count_tokens(corpus: Corpus) -> Counter[str]:
    """Count individual token frequencies across the corpus.

    Needed for the WordPiece score denominator: freq(a) * freq(b).
    """
    token_counts: Counter[str] = Counter()
    for word, freq in corpus.items():
        for token in word:
            token_counts[token] += freq
    return token_counts


def score_pairs(
    pair_counts: Counter[tuple[str, str]], token_counts: Counter[str]
) -> dict[tuple[str, str], float]:
    """Compute the WordPiece score for every adjacent pair.

    score(a, b) = freq(a·b) / (freq(a) * freq(b))

    High score means strong association — the pair co-occurs far more than
    you'd expect from the individual frequencies. This is the key difference
    from BPE, which uses raw frequency alone.

    Example: if 'a' appears 1000 times and 'b' appears 1000 times but 'ab'
    only appears 10 times, score = 10 / 1_000_000 = 0.00001 (weak). If 'x'
    and 'y' each appear 5 times and 'xy' appears 5 times, score = 5/25 = 0.2
    (strong — they always appear together).
    """
    return {
        pair: pair_count / (token_counts[pair[0]] * token_counts[pair[1]])
        for pair, pair_count in pair_counts.items()
    }


def merge_word(word: list[str], pair: tuple[str, str]) -> list[str]:
    """Non-overlapping left-to-right merge of `pair` inside a single word.

    The merged token keeps the first token's ## status and strips ##
    from the second:
        ("p", "##l")   → "pl"    (starts word)
        ("##l", "##a") → "##la"  (continues word)

    Same scan-and-skip structure as BPE pieces #1 and #2.
    """
    a, b = pair
    merged = a + b.removeprefix(CONTINUATION_PREFIX)
    result: list[str] = []
    i = 0
    while i < len(word):
        if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
            result.append(merged)
            i += 2
        else:
            result.append(word[i])
            i += 1
    return result


def apply_merge(corpus: Corpus, pair: tuple[str, str]) -> Corpus:
    """Return a new corpus with every occurrence of `pair` merged.

    Frequencies are accumulated (same collision-safety as piece #2).
    """
    new_corpus: Corpus = {}
    for word, freq in corpus.items():
        merged = tuple(merge_word(list(word), pair))
        new_corpus[merged] = new_corpus.get(merged, 0) + freq
    return new_corpus


class WordPieceTokenizer:
    """Placeholder — segments 3+ will fill this in."""

    pass
