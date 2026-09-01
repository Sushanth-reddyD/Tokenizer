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
    """
    a, b = pair
    merged_token = a + b
    result: list[str] = []
    i = 0
    while i < len(word):
        if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
            result.append(merged_token)
            i += 2
        else:
            result.append(word[i])
            i += 1
    return result


def apply_merge(
    corpus: list[list[str]], pair: tuple[str, str]
) -> list[list[str]]:
    """Return a new corpus with every occurrence of `pair` merged."""
    return [merge_word(word, pair) for word in corpus]


class BPETokenizer:
    """Char-level Byte Pair Encoding tokenizer.

    Attributes:
        vocab:  Mapping from token string to integer ID. IDs are assigned in
                the order tokens enter the vocab — first the sorted initial
                alphabet, then each learned merge in learning order.
        merges: Ordered list of (a, b) pairs learned by train(). Position in
                the list is the merge's "rank" — earlier = learned first.
    """

    def __init__(self) -> None:
        self.vocab: dict[str, int] = {}
        self.merges: list[tuple[str, str]] = []

    def train(self, text: str, vocab_size: int) -> None:
        """Learn BPE merges from `text` until vocab reaches `vocab_size`.

        Overwrites any prior training on this instance. Stops early when no
        repeating pair remains (further merges would not compress anything).
        """
        corpus = pretokenize(text)

        # Initial alphabet = every unique character in the corpus (plus </w>,
        # which pretokenize appends to every word). Sorting gives deterministic
        # ID assignment.
        initial_alphabet: set[str] = set()
        for word in corpus:
            initial_alphabet.update(word)
        self.vocab = {tok: i for i, tok in enumerate(sorted(initial_alphabet))}
        self.merges = []

        while len(self.vocab) < vocab_size:
            pair_counts = count_pairs(corpus)
            if not pair_counts:
                break  # every word collapsed to a single token

            max_count = max(pair_counts.values())
            if max_count < 2:
                break  # no repeating pair; further merges don't compress

            # Alphabetical tie-break among pairs at max_count → deterministic.
            best_pair = min(
                pair for pair, count in pair_counts.items() if count == max_count
            )

            corpus = apply_merge(corpus, best_pair)
            merged_token = best_pair[0] + best_pair[1]
            self.vocab[merged_token] = len(self.vocab)
            self.merges.append(best_pair)

    def tokenize(self, text: str) -> list[str]:
        """Segment text into BPE tokens (strings) using learned merges.

        Uses the per-word min-rank algorithm: for each word, repeatedly
        find the adjacent pair with the lowest rank (earliest learned merge)
        and apply it, until no known pair remains.

        Raises KeyError if the text contains characters not in the vocab.
        """
        ranks = {pair: i for i, pair in enumerate(self.merges)}
        tokens: list[str] = []
        for word in pretokenize(text):
            while len(word) > 1:
                best_pair = None
                best_rank = len(self.merges)
                for i in range(len(word) - 1):
                    pair = (word[i], word[i + 1])
                    rank = ranks.get(pair)
                    if rank is not None and rank < best_rank:
                        best_rank = rank
                        best_pair = pair
                if best_pair is None:
                    break
                word = merge_word(word, best_pair)
            tokens.extend(word)
        return tokens

    def encode(self, text: str) -> list[int]:
        """Encode text into a list of integer token IDs.

        Raises KeyError if any character in text was not seen during training.
        """
        return [self.vocab[tok] for tok in self.tokenize(text)]
