import pytest

from bytelevel.bytemap import (
    bytes_to_unicode,
    decode_token,
    encode_token,
    unicode_to_bytes,
)
from bytelevel.pretokenizer import pretokenize, split_text


# ---------- split_text: the lossless invariant ----------

LOSSLESS_CASES = [
    "Hello world",
    "Hello, world!",
    "don't DON'T It'S",
    "1234567",
    "a\n\nb",
    "hello   world",
    "café 🎉 中文",
    "  leading",
    "trailing   ",
    "",
    "x = 1  # note\n",
    "\t\ttabbed\r\n",
]


def test_split_text_is_lossless():
    """Joining the chunks must reproduce the input exactly.

    This is the property piece #1 could not offer, and everything else about
    byte-level BPE's lossless roundtrip rests on it.
    """
    for text in LOSSLESS_CASES:
        assert "".join(split_text(text)) == text, f"lost data on {text!r}"


# ---------- split_text: chunk boundaries ----------

def test_leading_space_attaches_to_word():
    assert split_text("Hello world") == ["Hello", " world"]


def test_punctuation_splits_off_words():
    assert split_text("Hello, world!") == ["Hello", ",", " world", "!"]


def test_contractions_split_case_insensitively():
    assert split_text("don't DON'T It'S") == [
        "don", "'t", " DON", "'T", " It", "'S",
    ]


def test_digits_capped_at_three():
    assert split_text("1234567") == ["123", "456", "7"]


def test_number_does_not_absorb_leading_space():
    """Unlike words, the \\p{N} branch has no optional leading space."""
    assert split_text("x = 1") == ["x", " =", " ", "1"]


def test_newline_runs_stay_together():
    assert split_text("a\n\nb") == ["a", "\n\n", "b"]


def test_whitespace_run_splits_before_final_space():
    """Only the last space of a run attaches to the following word."""
    assert split_text("hello   world") == ["hello", "  ", " world"]


def test_trailing_whitespace_is_its_own_chunk():
    assert split_text("trailing   ") == ["trailing", "   "]


def test_unicode_letters_and_emoji():
    assert split_text("café 🎉 中文") == ["café", " 🎉", " 中文"]


def test_empty_string():
    assert split_text("") == []


# ---------- pretokenize: UTF-8 encoding ----------

def test_pretokenize_encodes_utf8():
    assert pretokenize("café") == [b"caf\xc3\xa9"]


def test_pretokenize_multibyte_chunk():
    """A single emoji is one chunk but four UTF-8 bytes."""
    chunks = pretokenize("🎉")
    assert chunks == [b"\xf0\x9f\x8e\x89"]
    assert len(chunks[0]) == 4


def test_pretokenize_is_lossless():
    for text in LOSSLESS_CASES:
        assert b"".join(pretokenize(text)) == text.encode("utf-8")


def test_pretokenize_never_raises_on_unseen_characters():
    """The whole point of byte-level: no input can be out-of-vocabulary."""
    exotic = "𝕳𝖊𝖑𝖑𝖔 ᚠᚢᚦ ｱｲｳ \x00\x7f"
    assert b"".join(pretokenize(exotic)) == exotic.encode("utf-8")


# ---------- bytemap: shape of the map ----------

def test_map_covers_all_256_bytes():
    table = bytes_to_unicode()
    assert len(table) == 256
    assert set(table) == set(range(256))


def test_map_is_bijective():
    table = bytes_to_unicode()
    assert len(set(table.values())) == 256
    inverse = unicode_to_bytes()
    assert all(inverse[char] == b for b, char in table.items())


def test_split_of_self_mapped_and_shifted():
    """188 bytes are already safe; the other 68 get shifted to U+0100+."""
    table = bytes_to_unicode()
    self_mapped = [b for b, char in table.items() if ord(char) == b]
    shifted = [b for b, char in table.items() if ord(char) != b]
    assert len(self_mapped) == 188
    assert len(shifted) == 68
    assert all(ord(table[b]) >= 0x100 for b in shifted)


# ---------- bytemap: the serialization safety guarantee ----------

def test_no_output_character_is_whitespace():
    """merges.txt is space-separated and line-based. A whitespace character
    anywhere in the map would make the file ambiguous or truncated."""
    assert not any(char.isspace() for char in bytes_to_unicode().values())


def test_every_output_character_is_printable():
    """Control characters would corrupt the file; soft hyphen (173) would
    render as nothing and silently produce an unreadable vocab."""
    assert all(char.isprintable() for char in bytes_to_unicode().values())


# ---------- bytemap: specific mappings ----------

def test_dangerous_bytes_are_shifted():
    table = bytes_to_unicode()
    assert table[0] == "Ā"    # U+0100, NUL
    assert table[9] == "ĉ"    # U+0109, tab
    assert table[10] == "Ċ"   # U+010A, newline
    assert table[32] == "Ġ"   # U+0120, space — the one you see everywhere
    assert table[127] == "ġ"  # U+0121, DEL
    assert table[160] == "ł"  # NBSP
    assert table[173] == "Ń"  # U+0143, soft hyphen — last of the 68


def test_printable_bytes_map_to_themselves():
    table = bytes_to_unicode()
    for char in "!~aZ0123abcÿ":
        assert table[ord(char)] == char


# ---------- bytemap: encode_token / decode_token ----------

def test_encode_token_leading_space():
    """The signature of every GPT-2-family vocab file."""
    assert encode_token(b" world") == "Ġworld"


def test_encode_token_blank_line():
    assert encode_token(b"\n\n") == "ĊĊ"


def test_encode_token_empty():
    assert encode_token(b"") == ""


def test_token_roundtrip_every_single_byte():
    for b in range(256):
        token = bytes([b])
        assert decode_token(encode_token(token)) == token


def test_token_roundtrip_multibyte_utf8():
    for text in LOSSLESS_CASES:
        token = text.encode("utf-8")
        assert decode_token(encode_token(token)) == token


def test_token_roundtrip_all_bytes_at_once():
    token = bytes(range(256))
    assert decode_token(encode_token(token)) == token


def test_encoded_token_never_contains_a_space():
    """Directly locks the merges.txt parsing guarantee for real tokens."""
    for text in LOSSLESS_CASES + ["🎉 中文", "a\tb\nc"]:
        encoded = encode_token(text.encode("utf-8"))
        assert " " not in encoded
        assert "\n" not in encoded


def test_decode_token_rejects_foreign_characters():
    """A character outside the 256-char map means the file was not written
    by encode_token()."""
    with pytest.raises(KeyError):
        decode_token("🎉")


# ---------- bytemap: callers cannot corrupt the tables ----------

def test_returned_maps_are_copies():
    table = bytes_to_unicode()
    table[32] = "CORRUPTED"
    assert bytes_to_unicode()[32] == "Ġ"

    inverse = unicode_to_bytes()
    del inverse["Ġ"]
    assert unicode_to_bytes()["Ġ"] == 32
