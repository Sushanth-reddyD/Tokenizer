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
