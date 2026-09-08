"""Unigram LM tokenizer (piece #4).

The Unigram model (Kudo, 2018) starts with a large seed vocabulary and
iteratively prunes it by removing pieces whose removal increases the
corpus loss the least.  Tokenization uses the Viterbi algorithm to find
the maximum-likelihood segmentation under the current vocabulary.

Built from scratch for learning.
"""

import math

import regex

# Same cl100k_base pattern used by byte-level BPE (piece #2).
_CL100K_PATTERN = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
    r"|[^\r\n\p{L}\p{N}]?\p{L}+"
    r"|\p{N}{1,3}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+"
)

_COMPILED = regex.compile(_CL100K_PATTERN)


def pretokenize(text: str) -> list[str]:
    """Split text into word-level chunks using the cl100k_base regex.

    Lossless: ``"".join(pretokenize(text)) == text`` always holds.
    """
    return [m.group() for m in _COMPILED.finditer(text)]


def build_seed_vocab(text: str, max_seed_size: int = 16) -> dict[str, float]:
    """Build an oversized seed vocabulary from *text*.

    Steps:
      1. Pretokenize into chunks.
      2. For each chunk, extract every substring up to *max_seed_size*.
      3. Count substring frequencies.
      4. Ensure all single characters are present (guarantees coverage).
      5. Return ``{substring: log(count / total)}`` log-probabilities.
    """
    from collections import Counter

    counts: Counter[str] = Counter()
    chunks = pretokenize(text)

    for chunk in chunks:
        n = len(chunk)
        for i in range(n):
            for length in range(1, min(max_seed_size, n - i) + 1):
                counts[chunk[i:i + length]] += 1

    # Guarantee single-character coverage (even chars that only appear
    # inside longer chunks will already be counted, but if the text is
    # empty this is a no-op anyway).
    for chunk in chunks:
        for ch in chunk:
            if ch not in counts:
                counts[ch] = 1  # pragma: no cover — already counted above

    total = sum(counts.values())
    return {piece: math.log(freq / total) for piece, freq in counts.items()}


def viterbi_tokenize(text: str, vocab: dict[str, float]) -> list[str]:
    """Find the maximum-likelihood segmentation of *text* under *vocab*.

    Uses the Viterbi algorithm (dynamic programming).  For each position
    *i* in the string, we find the best previous position *j* such that
    ``text[j:i]`` is in *vocab* and the total log-probability is maximised.
    Then backtrack from the end to recover the segmentation.

    Raises ``ValueError`` if *text* cannot be segmented (a character is
    missing from *vocab*).
    """
    n = len(text)
    if n == 0:
        return []

    NEG_INF = float("-inf")

    # best_score[i] = best log-prob for text[:i].  best_score[0] = 0.
    best_score = [NEG_INF] * (n + 1)
    best_score[0] = 0.0

    # back[i] = the start position of the last piece ending at i.
    back = [0] * (n + 1)

    for i in range(1, n + 1):
        for j in range(i):
            piece = text[j:i]
            if piece in vocab:
                score = best_score[j] + vocab[piece]
                if score > best_score[i]:
                    best_score[i] = score
                    back[i] = j

    if best_score[n] == NEG_INF:
        raise ValueError(
            f"Cannot segment {text!r}: some character not in vocab"
        )

    # Backtrack.
    pieces: list[str] = []
    i = n
    while i > 0:
        pieces.append(text[back[i]:i])
        i = back[i]
    pieces.reverse()
    return pieces


def compute_loss(corpus: list[str], vocab: dict[str, float]) -> float:
    """Total negative log-likelihood of *corpus* under *vocab*.

    Each element of *corpus* is a pretokenized chunk (word).  We tokenize
    it with Viterbi and sum the **negative** log-probabilities of all
    resulting pieces.
    """
    total = 0.0
    for chunk in corpus:
        pieces = viterbi_tokenize(chunk, vocab)
        total += sum(-vocab[p] for p in pieces)
    return total


def compute_piece_scores(
    corpus: list[str], vocab: dict[str, float]
) -> dict[str, float]:
    """For each vocab piece, compute how much loss increases if it is removed.

    Returns ``{piece: delta_loss}``.  Single characters are excluded
    (they can never be removed — coverage guarantee).  A higher score
    means the piece is more important to keep.
    """
    base_loss = compute_loss(corpus, vocab)
    scores: dict[str, float] = {}

    for piece in list(vocab):
        # Never remove single characters.
        if len(piece) == 1:
            continue

        # Temporarily remove the piece and re-score.
        saved = vocab.pop(piece)
        try:
            new_loss = compute_loss(corpus, vocab)
            scores[piece] = new_loss - base_loss
        except ValueError:
            # Removal made some chunk unsegmentable — piece is essential.
            scores[piece] = float("inf")
        finally:
            vocab[piece] = saved

    return scores
