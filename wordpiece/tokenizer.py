"""WordPiece tokenizer (BERT lineage).

Where BPE merges the most *frequent* pair, WordPiece merges the pair that
maximizes a likelihood score: freq(ab) / (freq(a) * freq(b)).  This is a
mutual-information-style criterion: it favours pairs whose constituents
have a strong association, regardless of how common each one is alone.

Where BPE encodes by replaying merges in rank order, WordPiece encodes by
greedy longest-prefix matching against the vocab.

Built from scratch for learning.
"""

import json
import re
from collections import Counter
from pathlib import Path

from .pretokenizer import CONTINUATION_PREFIX, pretokenize, split_words

UNK_TOKEN = "[UNK]"

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
    """WordPiece tokenizer (BERT lineage).

    Attributes:
        vocab:  Mapping from token string to integer ID. IDs are assigned in
                order: first the sorted initial alphabet (character tokens with
                ## prefixes), then each learned merge in learning order.

    Unlike byte-level BPE, the vocab maps string→int (same direction as
    piece #1). The tokens are human-readable ("play", "##ing") rather than
    opaque integer IDs — no bytemap needed for serialization.
    """

    def __init__(self) -> None:
        self.vocab: dict[str, int] = {}
        self.special_tokens: dict[str, int] = {}

    def add_special_tokens(self, tokens: list[str]) -> dict[str, int]:
        """Register special tokens with IDs beyond the current vocab.

        Duplicates are silently skipped. Returns the full special_tokens
        mapping.
        """
        for s in tokens:
            if s not in self.special_tokens:
                sid = len(self.vocab)
                self.vocab[s] = sid
                self.special_tokens[s] = sid
        return dict(self.special_tokens)

    def train(self, text: str, vocab_size: int, special_tokens: list[str] | None = None, verbose: bool = False) -> None:
        """Learn WordPiece merges from `text` until the vocab reaches `vocab_size`.

        Overwrites any prior training. The initial vocab is every unique
        character token (including ## variants) found in the corpus, sorted
        alphabetically. Merges are chosen by the WordPiece score criterion:

            score(a, b) = freq(a·b) / (freq(a) * freq(b))

        Ties are broken by taking the lexicographically smallest pair.

        Stops when vocab_size is reached or no adjacent pairs remain.
        """
        corpus = build_corpus(text)

        initial_tokens: set[str] = set()
        for word in corpus:
            initial_tokens.update(word)
        self.vocab = {tok: i for i, tok in enumerate(sorted(initial_tokens))}
        self.special_tokens = {}

        num_merges = vocab_size - len(self.vocab)

        for _ in range(num_merges):
            pair_counts = count_pairs(corpus)
            if not pair_counts:
                break

            token_counts = count_tokens(corpus)
            scores = score_pairs(pair_counts, token_counts)

            max_score = max(scores.values())
            best_pair = min(
                pair for pair, s in scores.items() if s == max_score
            )

            merged_token = best_pair[0] + best_pair[1].removeprefix(
                CONTINUATION_PREFIX
            )
            corpus = apply_merge(corpus, best_pair)
            self.vocab[merged_token] = len(self.vocab)

            if verbose:
                print(
                    f"merge {len(self.vocab) - len(initial_tokens)}/{num_merges}: "
                    f"{best_pair} → {merged_token!r} "
                    f"(score={max_score:.4f})"
                )

        if special_tokens:
            self.add_special_tokens(special_tokens)

    def tokenize(self, text: str, encode_special_tokens: bool = False) -> list[str]:
        """Segment text into WordPiece token strings.

        Uses greedy longest-prefix matching — for each word, repeatedly find
        the longest prefix in the vocab and emit it. If any character in a
        word is not reachable (not even as a single-char token), the entire
        word becomes [UNK].

        When encode_special_tokens is True, registered special tokens are
        recognised as atomic units and the text between them is tokenized
        separately. Same safe-by-default pattern as piece #2.
        """
        if encode_special_tokens and self.special_tokens:
            return self._tokenize_with_specials(text)
        return self._tokenize_ordinary(text)

    def _tokenize_ordinary(self, text: str) -> list[str]:
        """WordPiece tokenization without special-token handling."""
        tokens: list[str] = []
        for word in split_words(text):
            subtokens = self._tokenize_word(word)
            tokens.extend(subtokens)
        return tokens

    def _tokenize_with_specials(self, text: str) -> list[str]:
        """Split text on special token boundaries, tokenize each gap."""
        pattern = "(" + "|".join(
            re.escape(s)
            for s in sorted(self.special_tokens, key=len, reverse=True)
        ) + ")"
        tokens: list[str] = []
        for part in re.split(pattern, text):
            if part in self.special_tokens:
                tokens.append(part)
            elif part:
                tokens.extend(self._tokenize_ordinary(part))
        return tokens

    def _tokenize_word(self, word: str) -> list[str]:
        """Greedy longest-prefix tokenization of a single word."""
        tokens: list[str] = []
        start = 0
        while start < len(word):
            end = len(word)
            matched = None
            while start < end:
                substr = word[start:end]
                if start > 0:
                    substr = CONTINUATION_PREFIX + substr
                if substr in self.vocab:
                    matched = substr
                    break
                end -= 1
            if matched is None:
                return [UNK_TOKEN]
            tokens.append(matched)
            start = end
        return tokens

    def decode(self, ids: list[int]) -> str:
        """Decode a list of token IDs back into a string.

        Reverses encode(): maps IDs to token strings, joins them with ##
        tokens concatenated directly and non-## tokens separated by spaces.

        Like piece #1, the roundtrip is lossy — str.split() collapses
        whitespace and punctuation splitting adds spaces:
            decode(encode("Hello, world!")) → "Hello , world !"
        Piece #2 avoided this because whitespace was in-band.

        Raises KeyError if an ID is not in the vocab.
        """
        id_to_token = {i: tok for tok, i in self.vocab.items()}
        parts: list[str] = []
        for token_id in ids:
            token = id_to_token[token_id]
            if token.startswith(CONTINUATION_PREFIX):
                parts.append(token.removeprefix(CONTINUATION_PREFIX))
            else:
                if parts:
                    parts.append(" ")
                parts.append(token)
        return "".join(parts)

    def encode(self, text: str, encode_special_tokens: bool = False) -> list[int]:
        """Encode text into a list of integer token IDs.

        Wraps tokenize() with a vocab lookup. Raises KeyError if a token
        (including [UNK]) is not in the vocab — add [UNK] to the vocab
        via special tokens to handle unknown words gracefully.
        """
        return [self.vocab[tok] for tok in self.tokenize(text, encode_special_tokens)]

    def save(self, directory: str | Path) -> None:
        """Persist the tokenizer to a directory.

        Writes two files:
          vocab.txt           — one token per line, line number = ID (BERT convention)
          special_tokens.json — {string: ID}, only written when special tokens exist

        BERT's format is simpler than GPT-2's: no bytemap needed because
        WordPiece tokens are already human-readable strings. No merges.txt
        because encoding uses greedy longest-match, not merge replay.
        """
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)

        id_to_token = {i: tok for tok, i in self.vocab.items()}
        with open(path / "vocab.txt", "w") as f:
            for i in range(len(self.vocab)):
                f.write(id_to_token[i] + "\n")

        if self.special_tokens:
            with open(path / "special_tokens.json", "w") as f:
                json.dump(self.special_tokens, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, directory: str | Path) -> "WordPieceTokenizer":
        """Load a tokenizer from a directory written by save()."""
        path = Path(directory)
        tok = cls()

        with open(path / "vocab.txt") as f:
            for i, line in enumerate(f):
                tok.vocab[line.rstrip("\n")] = i

        special_path = path / "special_tokens.json"
        if special_path.exists():
            with open(special_path) as f:
                tok.special_tokens = json.load(f)

        return tok
