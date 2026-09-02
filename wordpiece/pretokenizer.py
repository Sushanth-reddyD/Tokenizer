"""BERT-style pretokenizer: whitespace split, punctuation split, ## continuation.

Three steps:
  1. Split text on whitespace into raw words.
  2. Split each word further at punctuation boundaries — a punctuation character
     becomes its own token, so "don't" → ["don", "'", "t"].
  3. Convert each word into a list of characters, prefixing every character after
     the first with "##".

The ## prefix marks "this subword continues the previous one, inside the same
original word." It is the opposite convention from piece #1's </w> (which marked
word endings) and piece #2's leading-space (which marked word beginnings).

Example:
    pretokenize("playing well!")
    → [["p", "##l", "##a", "##y", "##i", "##n", "##g"],
       ["w", "##e", "##l", "##l"],
       ["!"]]
"""

import unicodedata

CONTINUATION_PREFIX = "##"


def _is_punctuation(char: str) -> bool:
    """Match BERT's definition of punctuation.

    BERT treats any non-letter, non-digit, non-whitespace character as
    punctuation. This includes ASCII punctuation, Unicode punctuation (e.g.
    full-width comma), and mathematical symbols — anything that is not a
    "normal" word character.
    """
    cp = ord(char)
    if (33 <= cp <= 47) or (58 <= cp <= 64) or (91 <= cp <= 96) or (123 <= cp <= 126):
        return True
    cat = unicodedata.category(char)
    return cat.startswith("P")


def _split_on_punctuation(word: str) -> list[str]:
    """Split a word so that each punctuation character is its own token.

    "don't" → ["don", "'", "t"]
    "hello!" → ["hello", "!"]
    "abc"    → ["abc"]   (no punctuation)
    """
    groups: list[list[str]] = []
    current: list[str] = []
    for char in word:
        if _is_punctuation(char):
            if current:
                groups.append(current)
            groups.append([char])
            current = []
        else:
            current.append(char)
    if current:
        groups.append(current)
    return ["".join(g) for g in groups]


def _to_wordpiece_chars(word: str) -> list[str]:
    """Convert a word to a list of characters with ## continuation prefixes.

    "play" → ["p", "##l", "##a", "##y"]
    "!"    → ["!"]
    ""     → []
    """
    if not word:
        return []
    return [word[0]] + [CONTINUATION_PREFIX + c for c in word[1:]]


def split_words(text: str) -> list[str]:
    """Split text into words (whitespace + punctuation) without ## chars.

    Used by the encoder, which needs raw word strings for the greedy
    longest-prefix algorithm. pretokenize() goes one step further and
    splits each word into ## character tokens — that's for training.

    Example:
        split_words("Hello, world!")  →  ["Hello", ",", "world", "!"]
    """
    words: list[str] = []
    for raw_word in text.strip().split():
        for subword in _split_on_punctuation(raw_word):
            if subword:
                words.append(subword)
    return words


def pretokenize(text: str) -> list[list[str]]:
    """BERT-style pretokenization: whitespace → punctuation → ## chars.

    Returns one list per "word" (where punctuation marks are their own words).
    Each word is a list of characters, with ## prefixes on continuations.

    Example:
        pretokenize("Hello, world!")
        → [["H", "##e", "##l", "##l", "##o"],
           [","],
           ["w", "##o", "##r", "##l", "##d"],
           ["!"]]
    """
    result: list[list[str]] = []
    for raw_word in text.strip().split():
        for subword in _split_on_punctuation(raw_word):
            chars = _to_wordpiece_chars(subword)
            if chars:
                result.append(chars)
    return result
