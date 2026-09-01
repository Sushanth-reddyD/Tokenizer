"""Char-level BPE tokenizer (Sennrich et al. 2016).

Built from scratch for learning.
"""

import json
from collections import Counter
from pathlib import Path

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


def _count_pairs_grouped(
    corpus: dict[tuple[str, ...], int],
) -> Counter[tuple[str, str]]:
    """Frequency-weighted pair counting over a grouped corpus."""
    pair_counts: Counter[tuple[str, str]] = Counter()
    for word, freq in corpus.items():
        for i in range(len(word) - 1):
            pair_counts[(word[i], word[i + 1])] += freq
    return pair_counts


def _apply_merge_grouped(
    corpus: dict[tuple[str, ...], int], pair: tuple[str, str]
) -> dict[tuple[str, ...], int]:
    """Frequency-weighted merge over a grouped corpus."""
    new_corpus: dict[tuple[str, ...], int] = {}
    for word, freq in corpus.items():
        merged = tuple(merge_word(list(word), pair))
        new_corpus[merged] = new_corpus.get(merged, 0) + freq
    return new_corpus


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

        Internally groups identical words by frequency so that count_pairs
        and apply_merge operate on unique words only (~8× faster on real
        corpora like tinyshakespeare).
        """
        raw_corpus = pretokenize(text)

        initial_alphabet: set[str] = set()
        for word in raw_corpus:
            initial_alphabet.update(word)
        self.vocab = {tok: i for i, tok in enumerate(sorted(initial_alphabet))}
        self.merges = []

        grouped: dict[tuple[str, ...], int] = {}
        for word in raw_corpus:
            key = tuple(word)
            grouped[key] = grouped.get(key, 0) + 1

        while len(self.vocab) < vocab_size:
            pair_counts = _count_pairs_grouped(grouped)
            if not pair_counts:
                break

            max_count = max(pair_counts.values())
            if max_count < 2:
                break

            best_pair = min(
                pair for pair, count in pair_counts.items() if count == max_count
            )

            grouped = _apply_merge_grouped(grouped, best_pair)
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

    def decode(self, ids: list[int]) -> str:
        """Decode a list of token IDs back into a string.

        Reverses encode(): maps IDs to token strings, concatenates them,
        then converts every </w> (or suffix containing it) into a space.
        Trailing space is stripped so decode(encode(text)) == text.

        Raises KeyError if an ID is not in the vocab.
        """
        id_to_token = {i: tok for tok, i in self.vocab.items()}
        text = "".join(id_to_token[i] for i in ids)
        text = text.replace(END_OF_WORD, " ")
        return text.strip()

    def save(self, directory: str | Path) -> None:
        """Persist vocab and merges to a directory.

        Writes two files following the GPT-2 convention:
          vocab.json  — token-to-ID mapping
          merges.txt  — one merge per line, space-separated pair
        """
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "vocab.json", "w") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)
        with open(path / "merges.txt", "w") as f:
            for a, b in self.merges:
                f.write(f"{a} {b}\n")

    @classmethod
    def load(cls, directory: str | Path) -> "BPETokenizer":
        """Load a tokenizer from a directory written by save()."""
        path = Path(directory)
        tok = cls()
        with open(path / "vocab.json") as f:
            tok.vocab = json.load(f)
        with open(path / "merges.txt") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                a, b = line.split(" ", maxsplit=1)
                tok.merges.append((a, b))
        return tok
