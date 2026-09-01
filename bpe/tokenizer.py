"""Char-level BPE tokenizer (Sennrich et al. 2016).

Built from scratch for learning.
"""

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
