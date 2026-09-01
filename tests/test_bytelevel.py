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
