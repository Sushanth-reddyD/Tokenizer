"""GPT-2's reversible map from the 256 byte values to printable characters.

Byte-level BPE computes on raw bytes, but it has to *write* those bytes to
disk. That is where raw bytes fail:

  - merges.txt is space-separated, and byte-level tokens contain literal
    spaces (b" world" is a normal token). "Ġ world" would be ambiguous.
  - merges.txt is line-based, and a token can be b"\\n", which would end the
    record early.
  - vocab.json keys must be strings, and byte 0x0A as a key would embed a
    real newline in that key.

Piece #1 dodged a milder version of this with split(maxsplit=1). That is not
enough here, because in byte-level BPE the space leads the token.

The fix, from Radford et al. (GPT-2): map all 256 byte values bijectively onto
256 characters that are all printable, visible, and non-whitespace. Bytes that
already qualify map to themselves; the 68 that do not are shifted to U+0100
and up. The result is both safe to serialize and legible to read.

This map is applied ONLY at the save/load boundary. train(), tokenize(),
encode() and decode() work on raw ints and bytes and never see it.

Built from scratch for learning.
"""

# The byte values that are already printable and non-whitespace, so they can
# stand for themselves. Each exclusion below prevents a specific file-format bug:
#   32  space      -> would break space-separated merges.txt
#   127 DEL, 0-31  -> control characters, corrupt the file
#   160 NBSP       -> whitespace
#   173 soft hyphen-> renders as nothing, silently unreadable vocab
_SELF_MAPPED_RANGES = (
    (ord("!"), ord("~")),    # 33..126   printable ASCII, minus space and DEL
    (ord("¡"), ord("¬")),    # 161..172  Latin-1 supplement, minus NBSP
    (ord("®"), ord("ÿ")),    # 174..255  ...minus soft hyphen
)


def _build_byte_to_char() -> dict[int, str]:
    """Construct the 256-entry map: 188 self-mapped, 68 shifted to U+0100+."""
    self_mapped = [
        b for start, end in _SELF_MAPPED_RANGES for b in range(start, end + 1)
    ]

    table = {b: chr(b) for b in self_mapped}

    # Every remaining byte gets the next free codepoint at or above U+0100,
    # assigned in ascending byte order. This is what makes byte 32 (space)
    # become 'Ġ' (U+0120): bytes 0..32 are all shifted, so space is the 33rd.
    next_codepoint = 0x100
    for b in range(256):
        if b not in table:
            table[b] = chr(next_codepoint)
            next_codepoint += 1

    return table


_BYTE_TO_CHAR = _build_byte_to_char()
_CHAR_TO_BYTE = {char: b for b, char in _BYTE_TO_CHAR.items()}


def bytes_to_unicode() -> dict[int, str]:
    """Return the byte -> character map (256 entries).

    A fresh copy each call, so a caller mutating it cannot corrupt the
    tokenizer. (GPT-2's original is @lru_cache'd and hands out the shared
    dict — convenient, but a footgun.)

    Example:
        bytes_to_unicode()[32]   -> 'Ġ'   # space
        bytes_to_unicode()[10]   -> 'Ċ'   # newline
        bytes_to_unicode()[97]   -> 'a'   # already printable, unchanged
    """
    return dict(_BYTE_TO_CHAR)


def unicode_to_bytes() -> dict[str, int]:
    """Return the inverse map, character -> byte (256 entries)."""
    return dict(_CHAR_TO_BYTE)


def encode_token(token: bytes) -> str:
    """Render a token's bytes as a printable, whitespace-free string.

    Used by save() when writing vocab.json and merges.txt.

    Example:
        encode_token(b" world")  -> 'Ġworld'
        encode_token(b"\\n\\n")    -> 'ĊĊ'
    """
    return "".join(_BYTE_TO_CHAR[b] for b in token)


def decode_token(text: str) -> bytes:
    """Recover a token's bytes from its printable form.

    Exact inverse of encode_token(). Used by load().

    Raises KeyError if `text` contains a character outside the 256-char map,
    which means the file was not written by encode_token().

    Example:
        decode_token('Ġworld')  -> b' world'
    """
    return bytes(_CHAR_TO_BYTE[char] for char in text)
