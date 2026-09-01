"""Regex pre-tokenizer for byte-level BPE (cl100k_base — GPT-4 / GPT-3.5-turbo).

Pre-tokenization splits raw text into chunks *before* BPE runs, so a merge can
never cross a chunk boundary.

Piece #1 did this with ``str.split()`` plus an explicit ``</w>`` marker, which
discarded the original whitespace — newlines and tab runs all collapsed into
single spaces, so decode could only reproduce a normalized form of the input.

This pattern keeps every character instead. A word's leading space is absorbed
into that word's chunk (``" world"`` is one chunk), so joining the chunks back
together reproduces the input byte for byte. That makes the ``</w>`` marker
unnecessary: whitespace is carried in-band by the tokens themselves.

Built from scratch for learning.
"""

import regex

# The cl100k_base pattern, as used by tiktoken for GPT-4 and GPT-3.5-turbo.
# Alternatives are tried left to right, so earlier branches win:
#
#   (?i:'s|'t|'re|'ve|'m|'ll|'d)   English contraction suffixes, case-insensitive
#                                  ("DON'T" splits as " DON" + "'T")
#   [^\r\n\p{L}\p{N}]?\p{L}+       a letter run, optionally preceded by ONE
#                                  non-letter/non-digit char (usually a space)
#   \p{N}{1,3}                     digits, in groups of at most 3. Note there is
#                                  no optional leading space here — unlike words,
#                                  a number never absorbs the space before it
#    ?[^\s\p{L}\p{N}]+[\r\n]*      punctuation/symbol run, optional leading
#                                  space, plus any trailing newlines
#   \s*[\r\n]+                     newline runs (blank lines stay together)
#   \s+(?!\S)                      a whitespace run not followed by a non-space,
#                                  i.e. every space except the one that will
#                                  attach to the next word
#   \s+                            any remaining whitespace
#
# \p{L} and \p{N} are Unicode property escapes. Python's stdlib `re` does not
# support them, which is why this module depends on the `regex` package.
CL100K_PATTERN = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
    r"|[^\r\n\p{L}\p{N}]?\p{L}+"
    r"|\p{N}{1,3}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+"
)

_COMPILED = regex.compile(CL100K_PATTERN)


def split_text(text: str) -> list[str]:
    """Split text into pre-token chunks, preserving every character.

    The split is lossless: ``"".join(split_text(text)) == text`` always holds.

    Example:
        split_text("Hello, world!")
        -> ['Hello', ',', ' world', '!']

        split_text("hello   world")
        -> ['hello', '  ', ' world']
    """
    return [match.group() for match in _COMPILED.finditer(text)]


def pretokenize(text: str) -> list[bytes]:
    """Split text into pre-token chunks and encode each one as UTF-8.

    This is the entry point BPE actually consumes. Working in bytes is what
    makes the tokenizer total: every possible string is a valid byte sequence,
    so unlike piece #1 there is no such thing as an out-of-vocabulary input.

    Example:
        pretokenize("café")
        -> [b'caf\\xc3\\xa9']        # 'é' is two UTF-8 bytes
    """
    return [chunk.encode("utf-8") for chunk in split_text(text)]
