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
